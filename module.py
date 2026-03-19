import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from scheduler import DDPMScheduler

class DiffusionModule(nn.Module):

    def __init__(self, network, scheduler: DDPMScheduler):
        super().__init__()
        self.network = network
        self.scheduler = scheduler

    def get_loss(self, x0, class_label=None, noise=None):
        device = x0.device
        batch_size = x0.shape[0]

        t = self.scheduler.uniform_sample_t(batch_size, device)

        if noise is None:
            noise = torch.randn_like(x0)

        x_t, noise = self.scheduler.add_noise(x0, t, noise)

        if self.network.use_cfg and class_label is not None:
            noise_pred = self.network(x_t, t, cls=class_label)
        else:
            noise_pred = self.network(x_t, t)

        loss = F.mse_loss(noise_pred, noise)
        return loss

    @torch.no_grad()
    def sample(self, batch_size, class_label=None, guidance_scale=1.0):
        device = next(self.network.parameters()).device
        image_size = self.network.image_size

        x = torch.randn(batch_size, 3, image_size, image_size).to(device)

        # FIX: usar TODOS los timesteps — quitar [::10]
        timesteps = self.scheduler.timesteps.to(device)

        for t in tqdm(timesteps):
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)

            if self.network.use_cfg and class_label is not None:
                eps_theta = self.network(x, t_batch, cls=class_label, scale=guidance_scale)
            else:
                eps_theta = self.network(x, t_batch)

            x = self.scheduler.step(x, t_batch, eps_theta)

        x = torch.clamp(x, -1., 1.)
        return x

    def save_model(self, file_path):
        torch.save({
            "network_state": self.network.state_dict(),
            "scheduler": self.scheduler.__dict__
        }, file_path)
        print(f"Model saved to {file_path}")

    def load_model(self, file_path, map_location=None):
        data = torch.load(file_path, map_location=map_location)
        self.network.load_state_dict(data["network_state"])
        print(f"Model loaded from {file_path}")
