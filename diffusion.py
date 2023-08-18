import torch.utils.data
import scipy.signal as sig
from network import *
from utils import plot_driver_generation, plot_station_generation, plot_training_loss
from maxsam import *

class DDPM:
    def __init__(self, opt, data_loader):
        super().__init__()
        if opt.network == "attention":
            self.eps_model = Attention(opt).to(opt.device)
        else:
            self.eps_model = CNN(opt).to(opt.device)
        self.opt = opt
        self.n_steps = opt.n_steps
        if opt.schedule == "linear":
            self.beta = torch.linspace(opt.beta_start, opt.beta_end, opt.n_steps, device=opt.device)
        else:
            self.beta = torch.linspace(opt.beta_start**0.5, opt.beta_end**0.5, opt.n_steps, device=opt.device)**2
        self.alpha = 1.0 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        # self.sigma2 = self.beta
        self.sigma2 = torch.cat((torch.tensor([self.beta[0]], device=opt.device), self.beta[1:]*(1-self.alpha_bar[0:-1])/(1-self.alpha_bar[1:])))
        self.optimizer = torch.optim.Adam(self.eps_model.parameters(), lr=opt.init_lr)
        self.data_loader = data_loader
        self.loss_func = nn.MSELoss()
        p1, p2 = int(0.75 * opt.n_epochs), int(0.9 * opt.n_epochs)
        self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[p1, p2], gamma=0.1)

    def gather(self, const, t):
        return const.gather(-1, t).view(-1, 1, 1)

    def q_xt_x0(self, x0, t):
        alpha_bar = self.gather(self.alpha_bar, t)
        mean = (alpha_bar**0.5)*x0
        var = 1 - alpha_bar
        return mean, var

    def q_sample(self, x0, t, eps):
        mean, var = self.q_xt_x0(x0, t)
        return mean + (var**0.5)*eps

    def p_sample(self, xt, c, t):
        eps_theta = self.eps_model(xt, c, t)
        alpha_bar = self.gather(self.alpha_bar, t)
        alpha = self.gather(self.alpha, t)
        eps_coef = (1 - alpha)/(1 - alpha_bar)**0.5
        mean = (xt - eps_coef*eps_theta)/(alpha**0.5)
        var = self.gather(self.sigma2, t)
        if (t == 0).all():
            z = torch.zeros(xt.shape, device=xt.device)
        else:
            z = torch.randn(xt.shape, device=xt.device)
        return mean + (var**0.5)*z

    def cal_loss(self, x0, c):  # (B, L, 1)
        batch_size = x0.shape[0]
        t = torch.randint(0, self.n_steps, (batch_size,), device=x0.device, dtype=torch.long)
        noise = torch.randn_like(x0)
        xt = self.q_sample(x0, t, eps=noise)
        eps_theta = self.eps_model(xt, c, t)
        return self.loss_func(noise, eps_theta)

    def sample(self, weight_path, n_samples, condition):
        c = torch.from_numpy(condition).type(torch.float32)
        c = c.view(1, -1).to(self.opt.device)
        with torch.no_grad():
            weight = torch.load(weight_path, map_location=self.opt.device)
            self.eps_model.load_state_dict(weight)
            self.eps_model.eval()
            if self.opt.level == "station":
                if condition[0] == 1:
                    max_sampler = pmax_sample("ACN-data/caltech/station/2019", n_samples)
                else:
                    max_sampler = pmax_sample("ACN-data/jpl/station/2019", n_samples)
            else:
                max_sampler = cmax_sample("ACN-data/jpl/driver", n_samples)
            for i in range(n_samples):
                x = torch.randn([1, self.opt.seq_len, self.opt.input_dim])
                for j in range(0, self.n_steps, 1):
                    t = torch.ones(1, dtype=torch.long)*(self.n_steps-j-1)
                    x = self.p_sample(x, c, t)
                x = x.squeeze().detach().numpy()
                x = sig.medfilt(x, 5)
                gen = x * max_sampler[i] / np.max(x)  # scale to 1.0
                path = f"generation/{self.opt.model_name}/{self.opt.network}/{self.opt.level}/{self.opt.cond_flag}/{i}"
                if self.opt.level == "station":
                    gen_filt = self.station_postprocess(gen)
                    plot_station_generation(gen_filt, path)
                else:
                    gen_filt1, gen_filt2 = self.driver_postprocess(gen)
                    plot_driver_generation(gen_filt1, gen_filt2, path)

    def driver_postprocess(self, x):
        x = sig.medfilt(x, kernel_size=5)
        low_index = np.where(x < 0)[0]
        high_index = np.where(x > 32)[0]
        x[low_index], x[high_index] = 0.0, 32
        x_filt1 = x
        # identify zero padding
        try:
            zero_index = np.where(x < 0.5)[0]
            invalid_index = zero_index[np.where(zero_index > 50)[0][0]]
            x_filt2 = x[0:invalid_index+1]
        except:
            x_filt2 = x_filt1
        return x_filt1, x_filt2

    def station_postprocess(self, x):
        # x[np.where(x < 0)[0]] = 0.0
        # x = sig.medfilt(x, kernel_size=5)
        # try:
        #     low_index = np.where(x < 10)[0][-1]
        #     if low_index > 200:
        #         x[low_index:] = 0
        # except:
        #     pass
        return x

    def train(self):
        epoch_loss = []
        for epoch in range(self.opt.n_epochs):
            batch_loss = []
            for i, data in enumerate(self.data_loader):
                if self.opt.level == "station":
                    x0 = data["power"].to(self.opt.device)
                else:
                    x0 = data["current"].to(self.opt.device)
                c = data["condition"].to(self.opt.device)
                self.optimizer.zero_grad()
                loss = self.cal_loss(x0, c)
                loss.backward()
                self.optimizer.step()
                batch_loss.append(loss.item())
            epoch_loss.append(np.mean(batch_loss))
            print(f"epoch={epoch}/{self.opt.n_epochs}, loss={epoch_loss[-1]}")
            self.lr_scheduler.step()
            save_path = f"weights/{self.opt.model_name}/{self.opt.network}/{self.opt.level}/{self.opt.cond_flag}/epoch{epoch}.pt"
            torch.save(self.eps_model.state_dict(), save_path)
        plot_training_loss(epoch_loss, model_name=f"{self.opt.model_name}_{self.opt.network}_{self.opt.level}", labels=["Diffusion Loss"])