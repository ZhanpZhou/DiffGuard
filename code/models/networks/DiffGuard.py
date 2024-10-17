import math
from inspect import isfunction
import numpy as np
import torch
import torch.nn as nn
import tqdm
from . import BACKBONES

class DiffGuard(nn.Module):

    def __init__(self, backbone, input_h, input_w, noise_channel, w_guide=0.1, p_uncond=0.1, **kwargs):
        super(DiffGuard, self).__init__()
        self.denoise_fn = BACKBONES[backbone](input_h=input_h, input_w=input_w, **kwargs)
        self.input_h = input_h
        self.input_w = input_w
        self.noise_channel = noise_channel
        self.w_guide = w_guide
        self.p_uncond = p_uncond
        self.beta_schedule = {
            "train": {
                "schedule": "linear",
                "n_timestep": 2000,
                "linear_start": 1e-6,
                "linear_end": 0.01
            },
            "test": {
                "schedule": "linear",
                "n_timestep": 1000,
                "linear_start": 1e-4,
                "linear_end": 0.09
            }
        }

    def set_new_noise_schedule(self, device=torch.device('cuda'), phase='train'):
        betas = make_beta_schedule(**self.beta_schedule[phase])
        self.betas = betas.detach().cpu().numpy() if isinstance(betas, torch.Tensor) else betas
        alphas = 1. - self.betas
        self.sqrt_alphas = torch.FloatTensor(np.sqrt(alphas))
        self.sqrt_betas = torch.FloatTensor(np.sqrt(self.betas))

        timesteps, = self.betas.shape
        self.num_timesteps = int(timesteps)
        
        gammas = np.cumprod(alphas, axis=0)
        gammas_prev = np.append(1., gammas[:-1])

        # calculations for diffusion p(x_t | x_{t-1}) and others
        self.gammas = torch.FloatTensor(gammas)
        self.sqrt_recip_gammas = torch.FloatTensor(np.sqrt(1. / gammas))
        self.sqrt_recipm1_gammas = torch.FloatTensor(np.sqrt(1. / gammas - 1))

        # calculations for posterior p(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - gammas_prev) / (1. - gammas)
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.posterior_log_variance_clipped = torch.FloatTensor(np.log(np.maximum(posterior_variance, 1e-20)))
        self.posterior_mean_coef1 = torch.FloatTensor(betas * np.sqrt(gammas_prev) / (1. - gammas))
        self.posterior_mean_coef2 = torch.FloatTensor((1. - gammas_prev) * np.sqrt(alphas) / (1. - gammas))

    def sample_t(self, batch_size):
        return torch.randint(1, self.num_timesteps, (batch_size,)).long()

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_gammas, t, x_t.shape).to(x_t.device) * x_t -
            extract(self.sqrt_recipm1_gammas, t, x_t.shape).to(x_t.device) * noise
        )

    def p_posterior(self, x_0_hat, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape).to(x_0_hat.device) * x_0_hat +
            extract(self.posterior_mean_coef2, t, x_t.shape).to(x_t.device) * x_t
        )
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_log_variance_clipped

    def p_mean_variance(self, x_t, t, clip_denoised: bool, x_cond=None, y=None):
        noise_level = extract(self.gammas, t, x_shape=(1, 1)).to(x_t.device)
        if x_cond is not None:
            x_0_hat = self.predict_start_from_noise(
                    x_t, t=t, noise=self.denoise_fn(torch.cat((x_cond, x_t), dim=1), noise_level, y=y).to(x_t.device))
        else:
            x_0_hat = self.predict_start_from_noise(
                    x_t, t=t, noise=self.denoise_fn(x_t, noise_level, y=y).to(x_t.device))

        if clip_denoised:
            x_0_hat.clamp_(-1., 1.)

        model_mean, posterior_log_variance = self.p_posterior(x_0_hat=x_0_hat, x_t=x_t, t=t)
        return model_mean, posterior_log_variance

    def sample_from_p(self, x_0, sample_gammas, noise=None):
        """
        Sample from p(x_t | x_0) for a batch of x_0.
        """
        noise = default(noise, lambda: torch.randn_like(x_0))
        return sample_gammas.sqrt() * x_0 + (1 - sample_gammas).sqrt() * noise

    def sample_previous(self, x_t, t, clip_denoised=True, x_cond=None, y=None):
        """
        Sample x_{t-1} from x_t using DDPM.
        """
        model_mean, model_log_variance = self.p_mean_variance(x_t=x_t, t=t, clip_denoised=clip_denoised, x_cond=x_cond, y=y)
        if self.w_guide and y is not None:
            # classifier-free guidance
            model_mean_null, model_log_variance_null = self.p_mean_variance(x_t=x_t, t=t, clip_denoised=clip_denoised, x_cond=x_cond, y=torch.zeros_like(y))
            model_mean += self.w_guide * (model_mean - model_mean_null)
        noise = torch.randn_like(x_t) if any(t>0) else torch.zeros_like(x_t)
        noise = noise.to(x_t.device)
        model_log_variance = model_log_variance.to(x_t.device)
        return model_mean + noise * (0.5 * model_log_variance).exp()

    def sample_next(self, x_t, t):
        """
        Sample x_{t+1} from x_t using DDPM.
        """
        noise = torch.randn_like(x_t).to(x_t.device)
        return (
            extract(self.sqrt_alphas, t, x_t.shape).to(x_t.device) * x_t +
            extract(self.sqrt_betas, t, x_t.shape).to(x_t.device) * noise
        )
    
    def train(self, mode=True):
        super().train(mode)
        if mode:
            self.set_new_noise_schedule(phase='train')
        else:
            self.set_new_noise_schedule(phase='test')
    
    def eval(self):
        super().eval()
        self.set_new_noise_schedule(phase='test')

    def forward(self, x_0, x_cond=None, y=None, mask=None, noise=None, restoration=False, x_t=None, num_samples=10, save_num=8):
        if not restoration:
            # sampling from p(gammas)
            b, *_ = x_0.shape
            t = self.sample_t(b).to(x_0.device)
            gamma_t1 = extract(self.gammas, t-1, x_shape=(1, 1))
            sqrt_gamma_t2 = extract(self.gammas, t, x_shape=(1, 1))
            sample_gammas = (sqrt_gamma_t2-gamma_t1) * torch.rand((b, 1)).to(sqrt_gamma_t2.device) + gamma_t1
            sample_gammas = sample_gammas.view(b, -1).to(x_0.device)

            noise = default(noise, lambda: torch.randn_like(x_0))
            x_noisy = self.sample_from_p(
                x_0=x_0, sample_gammas=sample_gammas.view(-1, 1, 1, 1).to(x_0.device), noise=noise)

            if self.p_uncond and y is not None:
                y *= broadcast_to(torch.rand((y.shape[0],)) > self.p_uncond, y)

            if x_cond is not None:
                if mask is not None:
                    noise_hat = self.denoise_fn(torch.cat((x_cond, x_noisy*mask+(1.-mask)*x_0), dim=1), sample_gammas, y=y)
                else:
                    noise_hat = self.denoise_fn(torch.cat((x_cond, x_noisy), dim=1), sample_gammas, y=y)
            else:
                noise_hat = self.denoise_fn(x_noisy, sample_gammas, y=y)
            return noise, noise_hat
        else:
            return self.restoration(x_cond=x_cond, x_t=x_t, x_0=x_0, mask=mask, num_samples=num_samples, save_num=save_num)
    
    @torch.no_grad()
    def restoration(self, num_samples, x_cond=None, x_0=None, mask=None, y=None, save_num=2, **kwargs):
        """
        Sample x_0 from x_t using DDPM.
        """
        assert self.num_timesteps > save_num, 'num_timesteps must greater than save_num'
        sample_inter = self.num_timesteps // save_num
        
        x_t = torch.randn((num_samples, self.noise_channel, self.input_h, self.input_w)).cuda(0)
        ret_arr = [(0, x_t)]
        for i in tqdm.tqdm(reversed(range(0, self.num_timesteps))):
            t = torch.full((num_samples,), i, device=x_t.device, dtype=torch.long)
            x_t = self.sample_previous(x_t, t, x_cond=x_cond, y=y) # Sample x_t_next from x_t
            if mask is not None:
                x_t = x_0*(1.-mask) + mask*x_t
            if i % sample_inter == 0:
                ret_arr.append((self.num_timesteps-i, x_t))
        return x_t, ret_arr

    def compute_alpha(self, beta, t):
        beta = torch.cat((torch.zeros(1).to(beta.device), beta), dim=0)
        a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
        return a

# gaussian diffusion trainer class
def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def extract(a, t, x_shape=(1,1,1,1)):
    b, *_ = t.shape
    out = a.gather(-1, t.to(a.device))
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

# beta_schedule function
def _warmup_beta(linear_start, linear_end, n_timestep, warmup_frac):
    betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    warmup_time = int(n_timestep * warmup_frac)
    betas[:warmup_time] = np.linspace(
        linear_start, linear_end, warmup_time, dtype=np.float64)
    return betas

def make_beta_schedule(schedule, n_timestep, linear_start=1e-6, linear_end=1e-2, cosine_s=8e-3):
    if schedule == 'quad':
        betas = np.linspace(linear_start ** 0.5, linear_end ** 0.5,
                            n_timestep, dtype=np.float64) ** 2
    elif schedule == 'linear':
        betas = np.linspace(linear_start, linear_end,
                            n_timestep, dtype=np.float64)
    elif schedule == 'warmup10':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.1)
    elif schedule == 'warmup50':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.5)
    elif schedule == 'const':
        betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    elif schedule == 'jsd':  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1. / np.linspace(n_timestep,
                                 1, n_timestep, dtype=np.float64)
    elif schedule == "cosine":
        timesteps = (
            torch.arange(n_timestep + 1, dtype=torch.float64) /
            n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * math.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = betas.clamp(max=0.999)
    else:
        raise NotImplementedError(schedule)
    return betas

def broadcast_to(arr, x, dtype=None, device=None, ndim=None):
    if x is not None:
        dtype = dtype or x.dtype
        device = device or x.device
        ndim = ndim or x.ndim
    out = torch.as_tensor(arr, dtype=dtype, device=device)
    return out.reshape((-1,) + (1,) * (ndim - 1))