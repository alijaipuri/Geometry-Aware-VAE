import torch
import torch.nn as nn
import torch.nn.functional as F


class GaussianVAE(nn.Module):
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

        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)

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
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.fc_decode(z)
        h = h.view(h.size(0), *self.decoder_shape)
        return self.decoder(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        # Resize to match input exactly
        recon = F.interpolate(recon, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False)
        return recon, mu, logvar, z

    def sample(self, num_samples, device):
        z = torch.randn(num_samples, self.latent_dim).to(device)
        return self.decode(z)
