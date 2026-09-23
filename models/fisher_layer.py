"""Trainable raw Fisher encoding, Deep FisherNet equations (8)-(10)."""
import math
import torch
from torch import nn


class FisherLayer(nn.Module):
    """[B,M,D] + optional boolean [B,M] mask -> [B,2*K*D].

    Output order is all first-order components followed by all second-order
    components. No power/L2 normalization is applied inside this layer.
    w and b are directly trainable; positivity is only required for the GMM
    standard deviations used at initialization, not for learned w.
    """

    def __init__(self, num_components=32, feature_dim=256):
        super().__init__()
        if num_components <= 0 or feature_dim <= 0:
            raise ValueError('num_components and feature_dim must be positive')
        self.num_components = num_components
        self.feature_dim = feature_dim
        self.w = nn.Parameter(torch.ones(num_components, feature_dim))
        # Break component symmetry for synthetic experiments. Real training
        # should initialize from a GMM fitted on training descriptors.
        self.b = nn.Parameter(torch.empty(num_components, feature_dim))
        nn.init.normal_(self.b, std=0.01)

    @torch.no_grad()
    def initialize_from_gmm(self, means, stds):
        """Initialize w=1/stds, b=-means. Pass stddev, NOT covariance."""
        means = torch.as_tensor(means, device=self.b.device, dtype=self.b.dtype)
        stds = torch.as_tensor(stds, device=self.w.device, dtype=self.w.dtype)
        if means.shape != self.b.shape or stds.shape != self.w.shape:
            raise ValueError('means and stds must have shape [K,D]')
        if not torch.isfinite(means).all() or not torch.isfinite(stds).all() or (stds <= 0).any():
            raise ValueError('GMM means must be finite; stds must be finite and positive')
        inverse = stds.reciprocal()
        if not torch.isfinite(inverse).all():
            raise ValueError('inverse standard deviations overflow the parameter dtype')
        self.w.copy_(inverse)
        self.b.copy_(-means)
        return self

    def forward(self, x, mask=None, assignment_temperature=1.):
        if not math.isfinite(assignment_temperature) or assignment_temperature<=0:
            raise ValueError('assignment temperature must be finite and positive')
        if x.ndim != 3 or x.shape[2] != self.feature_dim or min(x.shape[:2]) == 0:
            raise ValueError('x must be nonempty [B,M,D] with matching D')
        if x.device != self.w.device or x.dtype != self.w.dtype:
            raise ValueError('x and layer must have matching device and floating dtype')
        if mask is None:
            mask = torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
        if mask.shape != x.shape[:2] or mask.dtype != torch.bool or mask.device != x.device:
            raise ValueError('mask must be boolean [B,M] on the input device')
        counts = mask.sum(dim=1)
        if (counts == 0).any():
            raise ValueError('each image needs at least one valid patch')
        # Remove invalid padding before arithmetic, including NaN/Inf padding.
        clean = torch.where(mask[..., None], x, torch.zeros_like(x))
        if not torch.isfinite(clean).all():
            raise ValueError('valid patch descriptors must be finite')
        z = self.w[None, None] * (clean[:, :, None, :] + self.b[None, None])
        squared = z.square()
        gamma = torch.softmax(-0.5 * squared.sum(dim=-1) / assignment_temperature, dim=-1)
        weighted = gamma[..., None] * mask[:, :, None, None]
        denominator = counts[:, None, None].to(x.dtype)
        first = (weighted * z).sum(dim=1) / denominator
        second = (weighted * (squared - 1)).sum(dim=1) / (denominator * math.sqrt(2))
        return torch.cat((first.flatten(1), second.flatten(1)), dim=1)
