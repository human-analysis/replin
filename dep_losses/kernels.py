import math
import torch

"""
Standard kernel functions for use in dependency loss calculations.
"""

class Gaussian:
    def __init__(self, sigma=None):
        self.sigma = sigma

    def __call__(self, x, y=None):
        if y is None:
            if self.sigma is None:
                dist = torch.cdist(x, x, p=2) ** 2
                dist_u = torch.triu(dist, diagonal=1)
                sigma = torch.sqrt(0.5 * torch.median(dist_u[dist_u > 0]))
            else:
                sigma = self.sigma
        else:
            dist = torch.cdist(x, y, p=2) ** 2
            if self.sigma is None:
                sigma = torch.sqrt(0.5 * torch.median(dist[dist > 0]))
            else:
                sigma = self.sigma

        kernel = torch.exp(-dist / (2*sigma**2))
        return kernel

class RFFGaussian:
    def __init__(self, sigma=None, rff_dim=200, sigma_numel_max=9000):
        self.sigma = sigma
        self.rff_dim = rff_dim
        self.numel_max = sigma_numel_max
        self.w = None
        self.b = None

    def __call__(self, x):
        dim_x = x.shape[1]
        device = x.device
        if self.sigma is None:
            n = min(self.numel_max, x.shape[0])
            rand = torch.randperm(n, device=device)
            x_samp = x[rand, :]
            x_samp = x_samp[0: n, :]
            dist = torch.cdist(x_samp, x_samp, p=2) ** 2
            dist = torch.triu(dist, diagonal=1)
            if (dist > 0).sum():
                sigma = torch.sqrt(0.5 * torch.median(dist[dist > 0]))
            else:
                sigma = 1
        else:
            sigma = self.sigma

        mu_x = torch.zeros(dim_x, device=x.device, dtype=x.dtype)
        sigma_x = torch.eye(dim_x, device=x.device, dtype=x.dtype) / (sigma ** 2)
        px = torch.distributions.MultivariateNormal(mu_x, sigma_x)
        self.w = px.sample((self.rff_dim,))
        p = torch.distributions.uniform.Uniform(torch.tensor([0.0], device=x.device, dtype=x.dtype), 2 * torch.tensor([math.pi], device=x.device, dtype=x.dtype))
        self.b = p.sample((self.rff_dim,)).squeeze(1)
        phi_x = torch.sqrt(2. / torch.tensor([self.rff_dim], device=x.device)) * torch.cos(torch.mm(x, self.w.t()) + self.b)

        return phi_x

class Linear:
    def __init__(self, *args, **kwargs):
        # self.numel_max = sigma_numel_max
        pass

    def __call__(self, x, y=None):
        if y is None:
            kernel = torch.mm(x, x.t())
        else:
            kernel = torch.mm(y, y.t())

        return kernel
