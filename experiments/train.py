import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.optim as optim
import yaml
import argparse
import json
from tqdm import tqdm
import numpy as np

from models import GaussianVAE, SphericalVAE
from utils import (set_seed, get_dataloader, save_checkpoint,
                   gaussian_vae_loss, spherical_vae_loss,
                   plot_loss_curves, plot_latent_space,
                   plot_reconstructions, plot_tsne)


def train_epoch(model, loader, optimizer, device, beta, model_type):
    model.train()
    total_loss = recon_sum = kl_sum = 0
    for x, _ in loader:
        x = x.to(device)
        optimizer.zero_grad()
        if model_type == 'gaussian':
            recon, mu, logvar, z = model(x)
            loss, recon_l, kl_l = gaussian_vae_loss(recon, x, mu, logvar, beta)
        else:
            recon, mu, kappa, z = model(x)
            loss, recon_l, kl_l = spherical_vae_loss(recon, x, kappa, model, beta)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        recon_sum += recon_l.item()
        kl_sum += kl_l.item()
    n = len(loader)
    return total_loss / n, recon_sum / n, kl_sum / n


def val_epoch(model, loader, device, beta, model_type):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            if model_type == 'gaussian':
                recon, mu, logvar, z = model(x)
                loss, _, _ = gaussian_vae_loss(recon, x, mu, logvar, beta)
            else:
                recon, mu, kappa, z = model(x)
                loss, _, _ = spherical_vae_loss(recon, x, kappa, model, beta)
            total_loss += loss.item()
    return total_loss / len(loader)


def collect_latents(model, loader, device, model_type, max_batches=20):
    model.eval()
    all_z, all_labels = [], []
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            if i >= max_batches:
                break
            x = x.to(device)
            if model_type == 'gaussian':
                _, mu, _, z = model(x)
            else:
                _, mu, _, z = model(x)
            all_z.append(z.cpu().numpy())
            all_labels.append(y.numpy())
    return np.concatenate(all_z), np.concatenate(all_labels)


def run_training(model_type='gaussian', dataset='mnist', config_path='configs/config.yaml'):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    cfg = config.get(model_type, config['gaussian'])
    seed = cfg.get('seed', 42)
    set_seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Training] model={model_type} | dataset={dataset} | device={device}")

    train_loader, in_channels = get_dataloader(dataset, cfg['batch_size'], train=True)
    val_loader, _ = get_dataloader(dataset, cfg['batch_size'], train=False)
    img_size = 32 if dataset == 'cifar10' else 28

    if model_type == 'gaussian':
        model = GaussianVAE(latent_dim=cfg['latent_dim'], in_channels=in_channels, img_size=img_size).to(device)
    else:
        model = SphericalVAE(latent_dim=cfg['latent_dim'], in_channels=in_channels, img_size=img_size).to(device)

    optimizer = optim.Adam(model.parameters(), lr=cfg['lr'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['epochs'])

    train_losses, val_losses = [], []
    best_val = float('inf')

    for epoch in range(1, cfg['epochs'] + 1):
        tr_loss, recon_l, kl_l = train_epoch(model, train_loader, optimizer, device, cfg['beta'], model_type)
        val_loss = val_epoch(model, val_loader, device, cfg['beta'], model_type)
        scheduler.step()

        train_losses.append(tr_loss)
        val_losses.append(val_loss)

        print(f"Epoch {epoch:03d} | Train={tr_loss:.4f} | Recon={recon_l:.4f} | KL={kl_l:.4f} | Val={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(model, optimizer, epoch, val_loss,
                            f'results/models/{model_type}_{dataset}_best.pth')

    # Save final
    save_checkpoint(model, optimizer, cfg['epochs'], train_losses[-1],
                    f'results/models/{model_type}_{dataset}_final.pth')

    # Plots
    plot_loss_curves(train_losses, val_losses,
                     f'{model_type.capitalize()} VAE Loss ({dataset})',
                     f'results/plots/{model_type}_{dataset}_loss.png')

    latent_z, labels = collect_latents(model, val_loader, device, model_type)

    plot_latent_space(latent_z, labels,
                      f'{model_type.capitalize()} VAE Latent Space ({dataset})',
                      f'results/plots/{model_type}_{dataset}_latent.png')

    plot_tsne(latent_z, labels,
              f'{model_type.capitalize()} VAE t-SNE ({dataset})',
              f'results/plots/{model_type}_{dataset}_tsne.png')

    # Save metrics
    metrics = {'train_losses': train_losses, 'val_losses': val_losses, 'best_val': best_val}
    os.makedirs('results/logs', exist_ok=True)
    with open(f'results/logs/{model_type}_{dataset}_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"[Done] Best val loss: {best_val:.4f}")
    return model, metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='gaussian', choices=['gaussian', 'spherical'])
    parser.add_argument('--dataset', default='mnist')
    args = parser.parse_args()
    run_training(args.model, args.dataset)
