import torch
import torch.nn as nn

class DDPMScheduler(nn.Module):
    def __init__(self, num_train_timesteps, beta_1, beta_T, mode="linear", sigma_type="small"):
        super().__init__()
        self.num_train_timesteps = num_train_timesteps
        self.timesteps = torch.arange(num_train_timesteps - 1, -1, -1)

        # --- Scheduler ---
        if mode == "linear":
            betas = torch.linspace(beta_1, beta_T, num_train_timesteps)
        elif mode == "quadratic":
            betas = torch.linspace(beta_1**0.5, beta_T**0.5, num_train_timesteps)**2
        else:
            raise ValueError("mode must be 'linear' or 'quadratic'")

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        # Sigma
        sigma = torch.zeros_like(betas)
        if sigma_type == "small":
            sigma[1:] = torch.sqrt(betas[1:] * (1 - alphas_cumprod[:-1]) / (1 - alphas_cumprod[1:]))
        elif sigma_type == "large":
            sigma = torch.sqrt(betas)
        else:
            raise ValueError("sigma_type must be 'small' or 'large'")

        # Guardar como buffers para que estén en el mismo device que el módulo
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sigmas", sigma)

        # self.betas = None
        # self.alphas = None
        # self.alphas_cumprod = None
        # self.sigmas = None

    # --- Para elegir t aleatorio en un batch ---
    def uniform_sample_t(self, batch_size, device):
        t = torch.randint(0, self.num_train_timesteps, (batch_size,), device=device)
        return t

    def add_noise(self, x0, t, noise=None):
        """
        Aplica forward diffusion a x0 en el timestep t.

        Args:
            x0: tensor de imágenes [B, C, H, W]
            t: tensor de timesteps [B]
            noise: opcional, tensor de ruido [B, C, H, W]

        Returns:
            x_t: imágenes ruidosas
            noise: el ruido usado
        """
        if noise is None:
            noise = torch.randn_like(x0)

        # Obtener alpha_hat[t] y convertir a shape [B,1,1,1] para broadcasting
        alpha_hat_t = self.alphas_cumprod[t].view(-1, 1, 1, 1)

        # x_t = sqrt(alpha_hat) * x0 + sqrt(1-alpha_hat) * noise
        x_t = torch.sqrt(alpha_hat_t) * x0 + torch.sqrt(1 - alpha_hat_t) * noise

        return x_t, noise

    def step(self, x_t, t, eps_theta):
        """
        Un paso de reverse diffusion para ir de x_t a x_{t-1}.
        
        Args:
            x_t: tensor [B, C, H, W] imagen ruidosa en timestep t
            t: tensor [B] de timesteps
            eps_theta: tensor [B, C, H, W] predicción de ruido por el modelo
        
        Returns:
            x_prev: tensor [B, C, H, W] imagen para timestep t-1
        """
        # Broadcasting
        alpha_t = self.alphas[t].view(-1, 1, 1, 1)
        alpha_hat_t = self.alphas_cumprod[t].view(-1, 1, 1, 1)
        sigma_t = self.sigmas[t].view(-1, 1, 1, 1)
        
        # Calcular mu_t
        mu_t = (x_t - torch.sqrt(1 - alpha_t) / torch.sqrt(1 - alpha_hat_t) * eps_theta) / torch.sqrt(alpha_t)
        
        # Generar ruido solo si t > 0
        noise = torch.randn_like(x_t)
        x_prev = mu_t + sigma_t * noise
        x_prev[t == 0] = mu_t[t == 0]  # si t==0, no agregamos ruido
        x_prev[t != 0] = torch.clamp(x_prev[t != 0], -1., 1.)
        
        return x_prev
