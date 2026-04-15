import torch
import torch.nn.functional as F


def gaussian_vae_loss(recon_x, x, mu, logvar, beta=0.1):
    recon_loss = F.mse_loss(recon_x, x, reduction='mean')
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    total = recon_loss + beta * kl_loss
    return total, recon_loss, kl_loss


def spherical_vae_loss(recon_x, x, kappa, model, beta=0.1):
    recon_loss = F.mse_loss(recon_x, x, reduction='mean')
    kl_loss = model.kl_loss(kappa)
    total = recon_loss + beta * kl_loss
    return total, recon_loss, kl_loss
