import torch
import torch.nn as nn

class DDPMScheduler(nn.Module):
    def __init__(self, num_train_timesteps, beta_1, beta_T, mode="linear", sigma_type="small"):
        super().__init__()
        self.num_train_timesteps = num_train_timesteps
        self.timesteps = torch.arange(num_train_timesteps - 1, -1, -1)

        if mode == "linear":
            betas = torch.linspace(beta_1, beta_T, num_train_timesteps)
        elif mode == "quadratic":
            betas = torch.linspace(beta_1**0.5, beta_T**0.5, num_train_timesteps)**2
        else:
            raise ValueError("mode must be 'linear' or 'quadratic'")

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        sigma = torch.zeros_like(betas)
        if sigma_type == "small":
            # FIX: formula correcta — sigma[0] = 0, sigma[t] para t>=1
            sigma[1:] = torch.sqrt(
                betas[1:] * (1 - alphas_cumprod[:-1]) / (1 - alphas_cumprod[1:])
            )
        elif sigma_type == "large":
            sigma = torch.sqrt(betas)
        else:
            raise ValueError("sigma_type must be 'small' or 'large'")

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sigmas", sigma)

    def uniform_sample_t(self, batch_size, device):
        t = torch.randint(0, self.num_train_timesteps, (batch_size,), device=device)
        return t

    def add_noise(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)

        alpha_hat_t = self.alphas_cumprod[t].view(-1, 1, 1, 1)
        x_t = torch.sqrt(alpha_hat_t) * x0 + torch.sqrt(1 - alpha_hat_t) * noise

        return x_t, noise

    def step(self, x_t, t, eps_theta):
        """
        Un paso de reverse diffusion: x_t -> x_{t-1}.
        FIX: eliminado el código duplicado que sobreescribía la protección t==0.
        """
        alpha_t     = self.alphas[t].view(-1, 1, 1, 1)
        alpha_hat_t = self.alphas_cumprod[t].view(-1, 1, 1, 1)
        sigma_t     = self.sigmas[t].view(-1, 1, 1, 1)

        # Media del paso reverso
        mu_t = (1 / torch.sqrt(alpha_t)) * (
            x_t - (1 - alpha_t) / torch.sqrt(1 - alpha_hat_t) * eps_theta
        )

        # Añadir ruido solo si t > 0
        noise = torch.randn_like(x_t)
        x_prev = mu_t + sigma_t * noise
        x_prev[t == 0] = mu_t[t == 0]   # sin ruido en el último paso

        return x_prev
