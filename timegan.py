import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from utils import *
from options import Options
import sys
from utils import *
from dataset import creat_dataloader

class CondRNN(nn.Module):
    def __init__(self, input_size, hidden_size, condition_size):
        super(CondRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Linear(condition_size, hidden_size)
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True)

    def forward(self, x, condition):  # (batch_size, 720, input_size), (batch_size, 3)
        # h0 = c0 = self.embedding(condition).view(1, x.shape[0], self.hidden_size)  # (batch_size, 1, hidden_size)
        # output, (_, _) = self.rnn(x, (h0, c0))  # (batch_size, 720, hidden_size)
        output, (_, _) = self.rnn(x)
        return output

class Generator(nn.Module):
    def __init__(self, opt):
        super(Generator, self).__init__()
        self.projector = nn.LSTM(1, opt.latent_size, num_layers=1, batch_first=True)
        self.netG = CondRNN(opt.latent_size, opt.hidden_G, opt.condition_size)
        self.fc = nn.Sequential(
            nn.Linear(opt.hidden_G, opt.input_size),
            nn.ReLU()
        )

    def forward(self, noise, condition):
        x, (_, _) = self.projector(noise)
        output = self.netG(x, condition)  # (batch_size, 720, input_size)
        generation = self.fc(output)
        return generation

class Discriminator(nn.Module):
    def __init__(self, opt):
        super(Discriminator, self).__init__()
        self.netD = CondRNN(opt.input_size, opt.hidden_D, opt.condition_size)
        self.fc = nn.Sequential(
            nn.Linear(opt.hidden_D, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x, condition):
        output = self.netD(x, condition)
        discrimination = self.fc(output)  # (batch_size, 720, 1)
        return discrimination.squeeze(2)

class TimeGAN:
    def __init__(self, opt, dataloader):
        super(TimeGAN, self).__init__()
        self.opt = opt
        self.dataloader = dataloader
        self.netG = Generator(opt).to(opt.device)
        self.netD = Discriminator(opt).to(opt.device)
        self.adversarial_loss_func = nn.BCELoss()
        self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr_G, betas=(0.5, 0.999))
        self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr_D, betas=(0.5, 0.999))

    def sample(self, batch_size):
        noise = torch.randn(batch_size, self.opt.seq_len, device=self.opt.device)
        # noise_norm = F.normalize(noise, p=2, dim=2)
        condition = torch.FloatTensor(batch_size, 3).uniform_(0, 1).to(self.opt.device)
        return noise.unsqueeze(2), condition

    def forward_G(self):
        self.noise, self.c_gen = self.sample(self.batch_size)
        self.c_gen = self.c_real
        self.x_gen = self.netG(self.noise, self.c_gen)
        self.y_gen = self.netD(self.x_gen, self.c_gen)

    def train(self):
        netG_loss_set, netD_loss_set = [], []
        for epoch in range(self.opt.n_epochs):
            for i, data in enumerate(self.dataloader):
                # prepare data
                self.x_real = data["current"].to(self.opt.device)
                self.c_real = None
                self.batch_size = self.x_real.shape[0]
                self.valid = torch.ones((self.batch_size, self.opt.seq_len), dtype=torch.float32).to(self.opt.device)
                self.fake = torch.zeros((self.batch_size, self.opt.seq_len), dtype=torch.float32).to(self.opt.device)
                # train discriminator
                self.optimizer_D.zero_grad()
                self.forward_G()
                self.y_real = self.netD(self.x_real, self.c_real)
                self.real_loss = self.adversarial_loss_func(self.y_real, self.valid)
                self.fake_loss = self.adversarial_loss_func(self.y_gen, self.fake)
                self.netD_loss = 0.5 * (self.real_loss + self.fake_loss)
                self.netD_loss.backward()
                self.optimizer_D.step()
                netD_loss_set.append(self.netD_loss.item())
                # train generator
                self.optimizer_G.zero_grad()
                self.forward_G()
                self.netG_loss = self.adversarial_loss_func(self.y_gen, self.valid)
                self.netG_loss.backward()
                self.optimizer_G.step()
                netG_loss_set.append(self.netG_loss.item())
                print(f"epoch={epoch}/{self.opt.n_epochs}, iteration={i}, "
                      f"netG_loss={netG_loss_set[-1]}, netD_loss={netD_loss_set[-1]}")
            model_para = {"G": self.netG.state_dict(), "D": self.netD.state_dict()}
            save_path = f"weights/{self.opt.model_name}/{self.opt.level}/epoch{epoch}.pt"
            torch.save(model_para, save_path)
        plot_labels = ["Generator Loss", "Discriminator Loss"]
        plot_training_loss(netG_loss_set, netD_loss_set, model_name=self.opt.model_name, labels=plot_labels)

    def test(self, weight_path, sample_num):
        with torch.no_grad():
            netG_weight = torch.load(weight_path, map_location=self.opt.device)["G"]
            self.netG.load_state_dict(netG_weight)
            self.netG.eval()
        noise, c_gen = self.sample(sample_num)
        x_gen = self.netG(noise, c_gen)
        for i in range(sample_num):
            x, c = x_gen[i].detach().numpy(), c_gen[i].detach().numpy()
            x_reverse = np.flipud(x)
            valid_index = np.where(x_reverse!=0)[0][0]
            x_valid = np.flipud(x_reverse[valid_index:])
            img_path = f"generation/timegan/driver/{i}"
            plot_driver_generation(x*32, x_valid*32, img_path)

if __name__ == "__main__":
    isTrain = False
    model_name = "timegan"
    opt = Options(model_name, isTrain)
    modules = sys.modules
    data_file = f"ACN-data/{opt.level}_generation_dataset.json"
    data_loader = creat_dataloader(data_file, opt.level, opt.batch_size, opt.shuffle, isTrain)
    model = TimeGAN(opt, data_loader)
    if isTrain:
        model.train()
    else:
        best_epoch = 30
        pt_file = f"weights/timegan/driver/epoch{best_epoch}.pt"
        model.test(pt_file, sample_num=1500)