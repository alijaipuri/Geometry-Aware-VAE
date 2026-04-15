import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np
import json
import yaml
from models import GaussianVAE, SphericalVAE
from utils import (get_dataloader, load_checkpoint, compute_silhouette,
                   compute_ssim, plot_reconstructions, plot_interpolation)


def evaluate_model(model_type='gaussian', dataset='mnist', config_path='configs/config.yaml'):
    with open(config_path) as f:
        config = yaml.safe_load(f)
    cfg = config.get(model_type, config['gaussian'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    val_loader, in_channels = get_dataloader(dataset, 64, train=False)
    img_size = 32 if dataset == 'cifar10' else 28

    if model_type == 'gaussian':
        model = GaussianVAE(latent_dim=cfg['latent_dim'], in_channels=in_channels, img_size=img_size).to(device)
    else:
        model = SphericalVAE(latent_dim=cfg['latent_dim'], in_channels=in_channels, img_size=img_size).to(device)

    ckpt_path = f'results/models/{model_type}_{dataset}_best.pth'
    if os.path.exists(ckpt_path):
        load_checkpoint(model, None, ckpt_path, device)
        print(f"Loaded checkpoint: {ckpt_path}")
    else:
        print(f"No checkpoint found at {ckpt_path}, using random weights")

    model.eval()
    all_z, all_labels, all_originals, all_recons = [], [], [], []
    ssim_scores = []

    with torch.no_grad():
        for i, (x, y) in enumerate(val_loader):
            if i >= 30:
                break
            x = x.to(device)
            if model_type == 'gaussian':
                recon, mu, logvar, z = model(x)
            else:
                recon, mu, kappa, z = model(x)
            all_z.append(z.cpu().numpy())
            all_labels.append(y.numpy())
            all_originals.append(x.cpu().numpy())
            all_recons.append(recon.cpu().numpy())
            for j in range(min(x.size(0), 10)):
                s = compute_ssim(x[j].cpu().numpy().squeeze(), recon[j].cpu().numpy().squeeze())
                ssim_scores.append(s)

    all_z = np.concatenate(all_z)
    all_labels = np.concatenate(all_labels)
    all_originals = np.concatenate(all_originals)
    all_recons = np.concatenate(all_recons)

    sil = compute_silhouette(all_z, all_labels)
    ssim_mean = float(np.mean(ssim_scores))

    print(f"[{model_type.upper()} | {dataset}] Silhouette={sil:.4f} | SSIM={ssim_mean:.4f}")

    # Plot reconstructions
    orig_imgs = [all_originals[i].transpose(1, 2, 0) for i in range(10)]
    recon_imgs = [all_recons[i].transpose(1, 2, 0) for i in range(10)]
    plot_reconstructions(orig_imgs, recon_imgs,
                         f'{model_type.capitalize()} VAE Reconstructions ({dataset})',
                         f'results/plots/{model_type}_{dataset}_recons.png')

    # Interpolation
    if len(all_z) >= 2:
        z_start = torch.tensor(all_z[0:1], dtype=torch.float32)
        z_end = torch.tensor(all_z[-1:], dtype=torch.float32)
        plot_interpolation(model, z_start, z_end, device,
                           f'results/plots/{model_type}_{dataset}_interpolation.png',
                           spherical=(model_type == 'spherical'))

    metrics = {'silhouette': sil, 'ssim': ssim_mean}
    os.makedirs('results/logs', exist_ok=True)
    with open(f'results/logs/{model_type}_{dataset}_eval.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    return metrics


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='gaussian')
    parser.add_argument('--dataset', default='mnist')
    args = parser.parse_args()
    evaluate_model(args.model, args.dataset)
