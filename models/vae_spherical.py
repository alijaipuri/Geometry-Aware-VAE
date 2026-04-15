import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SphericalVAE(nn.Module):
    """
    VAE with spherical (hypersphere) latent space.
    Latent vectors are normalized to lie on S^(d-1).
    Uses von Mises-Fisher inspired reparameterization.
    """

    def __init__(self, latent_dim=10, in_channels=1, img_size=28):
        super().__init__()
        self.latent_dim = latent_dim
        self.img_size = img_size
        self.in_channels = in_channels

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
        )

        self.flatten_dim = self._get_flatten_dim(in_channels, img_size)

        # mu direction (will be normalized) + kappa (concentration)
        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_kappa = nn.Linear(self.flatten_dim, 1)  # scalar concentration

        # Decoder
        self.fc_decode = nn.Linear(latent_dim, self.flatten_dim)
        self.decoder_shape = self._get_decoder_shape(in_channels, img_size)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(32, in_channels, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )

    def _get_flatten_dim(self, in_channels, img_size):
        dummy = torch.zeros(1, in_channels, img_size, img_size)
        out = self.encoder(dummy)
        return int(out.view(1, -1).shape[1])

    def _get_decoder_shape(self, in_channels, img_size):
        dummy = torch.zeros(1, in_channels, img_size, img_size)
        out = self.encoder(dummy)
        return out.shape[1:]

    def encode(self, x):
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        mu = self.fc_mu(h)
        mu_norm = F.normalize(mu, p=2, dim=1)          # project to unit sphere
        kappa = F.softplus(self.fc_kappa(h)) + 1.0    # concentration >= 1
        return mu_norm, kappa

    def reparameterize_vmf(self, mu, kappa):
        """
        Approximate vMF reparameterization using the Wood (1994) trick.
        Samples w ~ distribution on [-1,1] then combines with tangent vector.
        """
        d = self.latent_dim
        batch_size = mu.size(0)

        # Sample w (distance along mean direction)
        w = self._sample_weight(kappa, d, mu.device)  # (B, 1)

        # Sample unit vector in tangent space (perpendicular to mu)
        eps = torch.randn(batch_size, d, device=mu.device)
        e = F.normalize(eps - (eps * mu).sum(dim=1, keepdim=True) * mu, dim=1)

        # Combine
        z = w * mu + (1 - w ** 2).clamp(min=1e-10).sqrt() * e
        return F.normalize(z, dim=1)

    def _sample_weight(self, kappa, d, device):
        """Rejection sampling for vMF weight."""
        batch_size = kappa.size(0)
        kappa_val = kappa.squeeze(1).detach()

        # Use the approximation for numerical stability
        c = (d - 1.0) / 2.0
        b = (-2.0 * kappa_val + (4.0 * kappa_val ** 2 + (d - 1) ** 2).sqrt()) / (d - 1)
        x0 = (1.0 - b) / (1.0 + b)
        m = (d - 1.0) / 2.0
        c2 = kappa_val * x0 + (d - 1.0) * torch.log(1.0 - x0 ** 2)

        w = torch.zeros(batch_size, device=device)
        mask = torch.ones(batch_size, dtype=torch.bool, device=device)
        max_iter = 100

        for _ in range(max_iter):
            if not mask.any():
                break
            n = mask.sum().item()
            z = torch.distributions.Beta(m, m).sample((n,)).to(device)
            z_clamp = z.clamp(1e-6, 1 - 1e-6)
            u = torch.rand(n, device=device)
            W_prop = (1.0 - (1.0 + b[mask]) * z_clamp) / (1.0 - (1.0 - b[mask]) * z_clamp)
            accept = kappa_val[mask] * W_prop + (d - 1) * torch.log(1.0 - x0[mask] * W_prop) - c2[mask]
            accepted = torch.log(u) <= accept
            w_indices = mask.nonzero(as_tuple=True)[0]
            accepted_global = w_indices[accepted]
            w[accepted_global] = W_prop[accepted].float()
            mask[accepted_global] = False

        return w.unsqueeze(1)

    def decode(self, z):
        h = self.fc_decode(z)
        h = h.view(h.size(0), *self.decoder_shape)
        return self.decoder(h)

    def forward(self, x):
        mu, kappa = self.encode(x)
        z = self.reparameterize_vmf(mu, kappa)
        recon = self.decode(z)
        recon = F.interpolate(recon, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False)
        return recon, mu, kappa, z

    def sample(self, num_samples, device):
        # Uniform sampling on sphere
        z = torch.randn(num_samples, self.latent_dim, device=device)
        z = F.normalize(z, dim=1)
        return self.decode(z)

    def kl_loss(self, kappa):
        """
        KL divergence between vMF(mu, kappa) and uniform on sphere.
        Approximation: KL ≈ kappa * I_{d/2}(kappa) / I_{d/2-1}(kappa) - log C_d(kappa)
        """
        d = self.latent_dim
        kappa_val = kappa.squeeze(1)
        # Simplified approximation for large d
        kl = kappa_val - (d / 2.0 - 1.0) * torch.log(kappa_val + 1e-8)
        return kl.mean()
