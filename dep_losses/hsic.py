import torch
import torch.nn as nn

from .kernels import RFFGaussian, Linear
from .utils import mean_center

# "Measuring Statistical Dependence with Hilbert-Schmidt Norms" by
# Gretton et al., 2005

class HSIC(nn.Module):
    def __init__(self, k1_type="RFFGaussian", k2_type="RFFGaussian"):
        super().__init__()
        self.eps = 1e-4
        if k1_type == "RFFGaussian":
            self.kernel_s = RFFGaussian(rff_dim=200)
        elif k1_type == "Linear":
            self.kernel_s = Linear()
        else:
            raise ValueError("k1_type must be either RFFGaussian or Linear")
        if k2_type == "RFFGaussian":
            self.kernel_z = RFFGaussian(rff_dim=200)
        elif k2_type == "Linear":
            self.kernel_z = Linear()
        else:
            raise ValueError("k2_type must be either RFFGaussian or Linear")

    def forward(self, z1, z2):
        phi_z2 = self.kernel_s(z2)
        phi_z2 = mean_center(phi_z2, dim=0)

        phi_z1 = self.kernel_z(z1)
        phi_z1 = mean_center(phi_z1, dim=0)

        n, _ = phi_z1.shape  # batch size
        hsic_zz = torch.norm(torch.mm(phi_z1.t(), phi_z1), p='fro') / n
        hsic_ss = torch.norm(torch.mm(phi_z2.t(), phi_z2), p='fro') / n
        hsic_zs = torch.norm(torch.mm(phi_z1.t(), phi_z2), p='fro') / n

        out = hsic_zs ** 2 / (hsic_zz * hsic_ss)

        return out
