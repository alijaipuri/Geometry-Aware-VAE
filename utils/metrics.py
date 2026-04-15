import numpy as np
import torch
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import LabelEncoder


def compute_silhouette(latent_vectors, labels):
    if len(np.unique(labels)) < 2:
        return 0.0
    try:
        score = silhouette_score(latent_vectors, labels, sample_size=min(5000, len(labels)))
        return float(score)
    except Exception:
        return 0.0


def compute_ssim(img1, img2):
    """Simplified SSIM between two numpy image arrays [0,1]."""
    C1 = (0.01) ** 2
    C2 = (0.03) ** 2
    mu1 = img1.mean()
    mu2 = img2.mean()
    sigma1 = img1.var()
    sigma2 = img2.var()
    sigma12 = ((img1 - mu1) * (img2 - mu2)).mean()
    ssim = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
           ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1 + sigma2 + C2))
    return float(ssim)


def compute_fid_score(real_features, fake_features):
    """Simplified FID using feature statistics."""
    mu1, mu2 = real_features.mean(0), fake_features.mean(0)
    cov1 = np.cov(real_features.T)
    cov2 = np.cov(fake_features.T)
    diff = mu1 - mu2
    covmean = np.sqrt(cov1 @ cov2 + 1e-6 * np.eye(cov1.shape[0]))
    fid = diff @ diff + np.trace(cov1 + cov2 - 2 * covmean)
    return float(fid)
