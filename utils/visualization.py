import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import os
from sklearn.manifold import TSNE
import seaborn as sns


COLORS = plt.cm.tab10.colors


def plot_latent_space(latent_vecs, labels, title, save_path, dim=2):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    vecs = latent_vecs[:, :2] if latent_vecs.shape[1] >= 2 else latent_vecs
    scatter = ax.scatter(vecs[:, 0], vecs[:, 1], c=labels, cmap='tab10', alpha=0.6, s=5)
    plt.colorbar(scatter, ax=ax)
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel('z₁')
    ax.set_ylabel('z₂')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_tsne(latent_vecs, labels, title, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    n = min(3000, len(latent_vecs))
    idx = np.random.choice(len(latent_vecs), n, replace=False)
    vecs = latent_vecs[idx]
    lbls = labels[idx]
    perp = min(30, n // 5)
    tsne = TSNE(n_components=2, perplexity=perp, random_state=42)
    reduced = tsne.fit_transform(vecs)
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(reduced[:, 0], reduced[:, 1], c=lbls, cmap='tab10', alpha=0.6, s=8)
    plt.colorbar(scatter, ax=ax)
    ax.set_title(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_reconstructions(originals, reconstructions, title, save_path, n=10):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, axes = plt.subplots(2, n, figsize=(n * 1.5, 3))
    for i in range(n):
        img_orig = originals[i].squeeze()
        img_recon = reconstructions[i].squeeze()
        cmap = 'gray' if img_orig.ndim == 2 else None
        axes[0, i].imshow(img_orig, cmap=cmap)
        axes[0, i].axis('off')
        axes[1, i].imshow(img_recon, cmap=cmap)
        axes[1, i].axis('off')
    axes[0, 0].set_ylabel('Original', fontsize=10)
    axes[1, 0].set_ylabel('Reconstructed', fontsize=10)
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_interpolation(model, z_start, z_end, device, save_path, steps=10, spherical=False):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.eval()
    interps = []
    for alpha in np.linspace(0, 1, steps):
        z = (1 - alpha) * z_start + alpha * z_end
        if spherical:
            import torch.nn.functional as F
            z = F.normalize(z, dim=1)
        with torch.no_grad():
            img = model.decode(z.to(device))
        interps.append(img.cpu().squeeze())

    fig, axes = plt.subplots(1, steps, figsize=(steps * 1.5, 2))
    for i, img in enumerate(interps):
        img_np = img.numpy()
        cmap = 'gray' if img_np.ndim == 2 else None
        axes[i].imshow(img_np, cmap=cmap)
        axes[i].axis('off')
    fig.suptitle('Latent Space Interpolation', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_loss_curves(train_losses, val_losses, title, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(train_losses, label='Train Loss', color='steelblue', linewidth=2)
    ax.plot(val_losses, label='Val Loss', color='coral', linewidth=2, linestyle='--')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_comparison_table(results_dict, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('off')
    rows = list(results_dict.keys())
    cols = list(next(iter(results_dict.values())).keys())
    cell_data = [[f"{results_dict[r][c]:.4f}" if isinstance(results_dict[r][c], float) else str(results_dict[r][c])
                  for c in cols] for r in rows]
    table = ax.table(cellText=cell_data, rowLabels=rows, colLabels=cols,
                     cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.4, 2)
    plt.title('Model Comparison Table', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
