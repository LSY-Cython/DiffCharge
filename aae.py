import torch.nn as nn
import torch
import itertools
from options import Options
import sys
from utils import *
from dataset import creat_dataloader

def reparameterization(mu, std, opt):
    sampled_z = torch.randn(mu.size(0), opt.latent_size, device=opt.device)
    z = sampled_z * std + mu
    return z

class Encoder(nn.Module):
    def __init__(self, opt):
        super(Encoder, self).__init__()
        self.opt = opt
        self.encoder = nn.Sequential(
            nn.Linear(opt.seq_len, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 512),
            nn.LeakyReLU(0.2),
        )
        self.mu = nn.Linear(512, opt.latent_size)
        self.std = nn.Linear(512, opt.latent_size)

    def forward(self, x):
        x = self.encoder(x)
        mu = self.mu(x)
        std = self.std(x)
        z = reparameterization(mu, std, self.opt)
        return z, x

class Decoder(nn.Module):
    def __init__(self, opt):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(opt.latent_size, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, opt.seq_len),
            nn.ReLU()
        )

    def forward(self, z):
        x = self.decoder(z)
        return x

class Discriminator(nn.Module):
    def __init__(self, opt):
        super(Discriminator, self).__init__()
        self.netD = nn.Sequential(
            nn.Linear(opt.latent_size, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, z):
        y = self.netD(z)
        return y

class AAE:
    def __init__(self, opt, dataloader):
        super(AAE, self).__init__()
        self.opt = opt
        self.dataloader = dataloader
        self.encoder = Encoder(opt).to(opt.device)
        self.decoder = Decoder(opt).to(opt.device)
        self.netD = Discriminator(opt).to(opt.device)
        self.rec_loss_func = nn.L1Loss()
        self.enc_loss_func = nn.MSELoss()
        self.adversarial_loss = torch.nn.BCELoss()
        self.optimizer_AE = torch.optim.Adam(itertools.chain(self.encoder.parameters(), self.decoder.parameters()),
                                             lr=opt.lr_AE, betas=(0.5, 0.999))
        self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr_D, betas=(0.5, 0.999))

    def forward_AE(self):
        self.z_real, self.h_real = self.encoder(self.x_real)
        self.x_rec = self.decoder(self.z_real)
        self.z_rec, self.h_rec = self.encoder(self.x_rec)
        self.input_x, self.target_x = self.x_rec[:, 0:-1], self.x_rec[:, 1:]

    def step_AE(self):
        self.forward_AE()
        self.rec_loss = self.rec_loss_func(self.x_real, self.x_rec)
        self.enc_loss = self.enc_loss_func(self.h_real, self.h_rec)
        self.gen_loss = self.adversarial_loss(self.netD(self.z_real), self.valid)
        self.ae_loss = self.opt.alpha*self.rec_loss + (1-self.opt.alpha)*self.gen_loss + self.opt.beta*self.enc_loss
        self.optimizer_AE.zero_grad()
        self.ae_loss.backward()
        self.optimizer_AE.step()

    def step_D(self, z):
        real_loss = self.adversarial_loss(self.netD(z), self.valid)
        fake_loss = self.adversarial_loss(self.netD(self.z_real.detach()), self.fake)
        self.adv_loss = 0.5 * (real_loss + fake_loss)
        self.optimizer_D.zero_grad()
        self.adv_loss.backward()
        self.optimizer_D.step()

    def train(self):
        epoch_rec_loss, epoch_enc_loss, epoch_G_loss, epoch_D_loss = [], [], [], []
        for epoch in range(self.opt.n_epochs):
            batch_rec_loss, batch_enc_loss, batch_G_loss, batch_D_loss = [], [], [], []
            for i, data in enumerate(self.dataloader):
                self.x_real = data["current"].to(self.opt.device).squeeze(2)
                batch_size = self.x_real.shape[0]
                self.valid = torch.ones((batch_size, 1), dtype=torch.float32).to(self.opt.device)
                self.fake = torch.zeros((batch_size, 1), dtype=torch.float32).to(self.opt.device)
                z = torch.randn(batch_size, self.opt.latent_size, device=self.opt.device)
                self.step_AE()
                self.step_D(z)
                batch_rec_loss.append(self.rec_loss.item())
                batch_enc_loss.append(self.enc_loss.item())
                batch_G_loss.append(self.gen_loss.item())
                batch_D_loss.append(self.adv_loss.item())
            epoch_rec_loss.append(np.mean(batch_rec_loss))
            epoch_enc_loss.append(np.mean(batch_enc_loss))
            epoch_G_loss.append(np.mean(batch_G_loss))
            epoch_D_loss.append(np.mean(batch_D_loss))
            print(f"epoch={epoch}/{self.opt.n_epochs}, rec_loss={epoch_rec_loss[-1]}, enc_loss={epoch_enc_loss[-1]}, "
                  f"G_loss={epoch_G_loss[-1]}, D_loss={epoch_D_loss[-1]}")
            model_para = {"encoder": self.encoder.state_dict(), "decoder": self.decoder.state_dict()}
            save_path = f"weights/{self.opt.model_name}/{self.opt.level}/epoch{epoch}.pt"
            torch.save(model_para, save_path)
        plot_training_loss(epoch_rec_loss, epoch_enc_loss, epoch_G_loss, epoch_D_loss,
                           model_name=f"{self.opt.model_name}",
                           labels=["Reconstruction Loss", "Encoding Loss", "G Loss", "D Loss"])

    def test(self, pt_file, sample_num):
        weight = torch.load(pt_file, map_location=self.opt.device)
        self.encoder.load_state_dict(weight["encoder"])
        self.decoder.load_state_dict(weight["decoder"])
        self.encoder.eval()
        self.decoder.eval()
        z = torch.randn(sample_num, self.opt.latent_size)
        x_gen = self.decoder(z).squeeze().detach().numpy()
        for i in range(sample_num):
            x = x_gen[i]
            x_reverse = np.flipud(x)
            valid_index = np.where(x_reverse!=0)[0][0]
            x_valid = np.flipud(x_reverse[valid_index:])
            img_path = f"generation/{self.opt.model_name}/{self.opt.level}-alpha{opt.alpha}/{i}"
            plot_driver_generation(x*32, x_valid*32, img_path)
        # for i, data in enumerate(self.dataloader):
        #     self.x_real = data["pilot"].to(self.opt.device).squeeze(2)
        #     duration = data["duration"]
        #     self.forward_AE()
        #     real = self.x_real.detach().squeeze().numpy()[0:duration]
        #     rec = self.x_rec.detach().squeeze().numpy()[0:duration]
        #     z = self.z_real.detach().squeeze().numpy()
        #     img_path = f"reconstruction/{i}.png"
        #     plot_reconstruction(real, rec, z, img_path)

if __name__ == "__main__":
    isTrain = False
    model_name = "aae"
    opt = Options(model_name, isTrain)
    modules = sys.modules
    data_file = f"ACN-data/{opt.level}_generation_dataset.json"
    data_loader = creat_dataloader(data_file, opt.level, opt.batch_size, opt.shuffle, isTrain)
    model = AAE(opt, data_loader)
    if isTrain:
        model.train()
    else:
        best_epoch = 152  # alpha0.5——192, alpha0.75——152
        pt_file = f"weights/aae/driver/epoch{best_epoch}.pt"
        model.test(pt_file, sample_num=1500)
