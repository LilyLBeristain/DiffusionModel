import argparse
import torch
import numpy as np
from pathlib import Path
from torch_fidelity import calculate_metrics

# Import your diffusion model
from module import DiffusionModule
from scheduler import DDPMScheduler
from unet import UNet
from dataset import tensor_to_pil_image


# Sample images from the model
def sample_images(args):
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ----------------------------
    # Initialize network (same as training)
    # ----------------------------
    network = UNet(
        num_timesteps=1000,
        image_size=64,
        base_channels=128,
        channel_mults=[1, 2, 2, 2],
        attn_levels=[1],
        num_res_blocks=4,
        dropout=0.1,
        use_cfg=args.use_cfg
    )

    scheduler = DDPMScheduler(
        1000,
        beta_1=1e-4,
        beta_T=0.02,
        mode="linear"
    )

    ddpm = DiffusionModule(network, scheduler).to(device)

    # ----------------------------
    # Load trained weights
    # ----------------------------
    ddpm.load_model(args.ckpt_path, map_location=device)

    ddpm.eval()

    total_samples = args.num_samples
    num_batches = int(np.ceil(total_samples / args.batch_size))
    print(f"Generating {total_samples} images in {num_batches} batches...")

    for i in range(num_batches):

        sidx = i * args.batch_size
        eidx = min((i + 1) * args.batch_size, total_samples)

        B = eidx - sidx

        if args.use_cfg:
            labels = torch.randint(1, 4, (B,)).to(device)
            samples = ddpm.sample(B, class_label=labels, guidance_scale=args.cfg_scale)
        else:
            samples = ddpm.sample(B)

        pil_images = tensor_to_pil_image(samples)

        for j, img in zip(range(sidx, eidx), pil_images):
            img.save(save_dir / f"{j}.png")
            print(f"Saved image {j}")

    print("Sampling complete.")


# Compute FID using torch-fidelity
def calculate_fid(paths, device=None):

    print(f"Computing FID between:\n  {paths[0]}\n  {paths[1]}")

    results = calculate_metrics(
        input1=paths[0],
        input2=paths[1],
        fid=True,
        verbose=False,
        device=device
    )

    return results.get("frechet_inception_distance", None)


def main():

    parser = argparse.ArgumentParser(
        description="Unified Diffusion Sampling & FID Calculation"
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Sampling
    sample_parser = subparsers.add_parser("sample", help="Generate images")

    sample_parser.add_argument("--batch_size", type=int, default=64)
    sample_parser.add_argument("--gpu", type=int, default=0)

    sample_parser.add_argument(
        "--ckpt_path",
        type=str,
        default="results/diffusion-ddpm-02-05-010238/last.ckpt"
    )

    sample_parser.add_argument("--save_dir", type=str, default="generated_images")

    sample_parser.add_argument("--use_cfg", action="store_true")

    sample_parser.add_argument("--cfg_scale", type=float, default=7.5)

    sample_parser.add_argument("--num_samples", type=int, default=500)

    # FID Calculation
    fid_parser = subparsers.add_parser("fid", help="Compute FID")

    fid_parser.add_argument("--save_path", type=str, default="generated_images")

    fid_parser.add_argument("--gt_path", type=str, default="./data/eval")

    fid_parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    if args.command == "sample":
        sample_images(args)

    elif args.command == "fid":
        fid_value = calculate_fid(
            [args.save_path, args.gt_path],
            device=args.device
        )

        print("FID:", fid_value)


if __name__ == "__main__":
    main()