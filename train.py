import os
import math
import json
import argparse
import numpy as np
from itertools import chain
from pathlib import Path
from datetime import datetime
from multiprocessing.pool import Pool

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import torchvision.transforms as tv_transforms
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

from dotmap import DotMap
from pytorch_lightning import seed_everything

from dataset import DataModule, get_data_iterator, tensor_to_pil_image
from scheduler import DDPMScheduler
from unet import UNet
from module import DiffusionModule


def evaluation_dataset(data_root='data', batch_size=32, num_workers=4, image_resolution=32):
    dataset = DataModule(data_root, 'val', batch_size, num_workers, image_resolution)
    eval_dir = dataset.root.parent / 'eval'
    eval_dir.mkdir(exist_ok=True)
    
    for path in dataset.fnames:
        img = Image.open(path).resize((64, 64))
        img.save(eval_dir / path.name)
        print(f'Processed {path.name}')
    
    print(f'Constructed eval directory at {eval_dir}')

def main(args):
    config = DotMap(vars(args))
    config.device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"

    save_dir = Path("results")
    save_dir.mkdir(exist_ok=True)

    ckpt_path = save_dir / "last.ckpt"

    print(f"save_dir: {save_dir}")
    seed_everything(config.seed)

    image_resolution = args.image_resolution
    ds_module = DataModule("./data", "train", config.batch_size, num_workers=4, image_resolution=image_resolution)
    train_dl = ds_module.dataloader()
    train_it = get_data_iterator(train_dl)

    var_scheduler = DDPMScheduler(config.num_diffusion_train_timesteps, config.beta_1, config.beta_T, mode="linear")

    network = UNet(
        num_timesteps=config.num_diffusion_train_timesteps,
        image_size=image_resolution,
        base_channels=128,
        channel_mults=[1, 2, 2, 2],
        attn_levels=[1],
        num_res_blocks=4,
        dropout=0.1,
        use_cfg=args.use_cfg,
        cfg_dropout=args.cfg_dropout,
        num_classes=getattr(ds_module, "num_classes", None),
    )

    ddpm = DiffusionModule(network, var_scheduler).to(config.device)

    optimizer = torch.optim.Adam(ddpm.network.parameters(), lr=2e-4)

    scheduler_lr = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda t: min((t + 1) / config.warmup_steps, 1.0)
    )

    # Auto training for server fails
    step = 0
    losses = []

    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=config.device)
        ddpm.network.load_state_dict(checkpoint["network"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        step = checkpoint.get("step", 0)
        print(f"Resuming from step {step}")

    with tqdm(initial=step, total=config.train_num_steps) as pbar:
        while step < config.train_num_steps:

            if step % config.log_interval == 0:
                ddpm.eval()

                if len(losses) > 0:
                    plt.plot(losses)
                    plt.savefig(save_dir / "loss.png")
                    plt.close()

                samples = ddpm.sample(4)
                imgs = tensor_to_pil_image(samples)

                for i, img in enumerate(imgs):
                    img.save(save_dir / f"step={step}-{i}.png")

                # Save data checkpoint
                torch.save({
                    "network": ddpm.network.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": step
                }, ckpt_path)

                ddpm.train()

            img, label = next(train_it)
            img, label = img.to(config.device), label.to(config.device)

            loss = ddpm.get_loss(img, class_label=label) if args.use_cfg else ddpm.get_loss(img)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler_lr.step()

            losses.append(loss.item())

            step += 1
            pbar.set_description(f"Loss: {loss.item():.4f}")
            pbar.update(1)

    # Generate 10 images
    print("Generating final samples...")
    ddpm.eval()
    samples = ddpm.sample(10)
    imgs = tensor_to_pil_image(samples)

    for i, img in enumerate(imgs):
        img.save(save_dir / f"final_{i}.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--train_num_steps", type=int, default=100000,
                        help="number of training steps")
    parser.add_argument("--warmup_steps", type=int, default=200)
    parser.add_argument("--log_interval", type=int, default=2000)
    parser.add_argument("--max_num_images_per_cat", type=int, default=3000,
                        help="max images per category")
    parser.add_argument("--num_diffusion_train_timesteps", type=int, default=1000,
                        help="diffusion steps")
    parser.add_argument("--beta_1", type=float, default=1e-4)
    parser.add_argument("--beta_T", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--image_resolution", type=int, default=64)
    parser.add_argument("--sample_method", type=str, default="ddpm")
    parser.add_argument("--use_cfg", action="store_true")
    parser.add_argument("--cfg_dropout", type=float, default=0.1)
    parser.add_argument("--eval", action="store_true", help="run evaluation to prepare eval images")
    args = parser.parse_args()

    
    evaluation_dataset(data_root="data", batch_size=args.batch_size, num_workers=4, image_resolution=args.image_resolution)
    main(args)