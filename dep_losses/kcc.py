import torch
import torch.nn as nn

from .kernels import RFFGaussian, Linear
from .utils import mean_center

# "Kernel Independent Component Analysis" by Bach and Jordan, JMLR 2002

class KCC(nn.Module):
    def __init__(self, k1_type="RFFGaussian", k2_type="RFFGaussian", lam=0.01):
        super().__init__()
        self.lam = lam
        if k1_type == "RFFGaussian":
            self.kernel_s = RFFGaussian(rff_dim=100)
        elif k1_type == "Linear":
            self.kernel_s = Linear()
        else:
            raise ValueError("k1_type must be either RFFGaussian or Linear")
        if k2_type == "RFFGaussian":
            self.kernel_z = RFFGaussian(rff_dim=100)
        elif k2_type == "Linear":
            self.kernel_z = Linear()
        else:
            raise ValueError("k2_type must be either RFFGaussian or Linear")

    def forward(self, z, s):
        if len(z.shape) == 1:
            z = z.reshape(-1, 1)
        if len(s.shape) == 1:
            s = s.reshape(-1, 1)
        z = z.type(torch.float)
        s = s.type(torch.float)
        phi_z = self.kernel_z(z)
        phi_s = self.kernel_s(s)

        phi_z = mean_center(phi_z, dim=0)
        phi_s = mean_center(phi_s, dim=0)

        n = phi_z.shape[0]

        m_z = phi_z.shape[1]
        m_s = phi_s.shape[1]

        # st_time = time.time()
        c_zz = torch.mm(phi_z.t(), phi_z) / n
        c_ss = torch.mm(phi_s.t(), phi_s) / n
        c_zs = torch.mm(phi_z.t(), phi_s) / n

        psd_z = torch.linalg.inv(c_zz + self.lam * torch.eye(m_z, device=c_zz.device))
        psd_s = torch.linalg.inv(c_ss + self.lam * torch.eye(m_s, device=c_ss.device))
        zeros_z = torch.zeros(m_z, m_z, device=c_zs.device)
        zeros_s = torch.zeros(m_s, m_s, device=c_zs.device)

        a_zs = torch.cat((zeros_z, torch.mm(psd_z, c_zs)), dim=1)
        a_sz = torch.cat((torch.mm(psd_s, c_zs.t()), zeros_s), dim=1)
        b_zs = torch.cat((a_zs, a_sz), dim=0)

        a_zz = torch.cat((zeros_z, torch.mm(psd_z, c_zz)), dim=1)
        a_zzt = torch.cat((torch.mm(psd_z, c_zz.t()), zeros_z), dim=1)
        b_zz = torch.cat((a_zz, a_zzt), dim=0)

        a_ss = torch.cat((zeros_s, torch.mm(psd_s, c_ss)), dim=1)
        a_sst = torch.cat((torch.mm(psd_s, c_ss.t()), zeros_s), dim=1)
        b_ss = torch.cat((a_ss, a_sst), dim=0)

        kcc_zs = torch.max(torch.real(torch.linalg.eigvals(b_zs))) / n
        kcc_zz = torch.max(torch.linalg.eigvalsh(b_zz)) / n
        kcc_ss = torch.max(torch.linalg.eigvalsh(b_ss)) / n

        return kcc_zs / torch.sqrt(kcc_zz*kcc_ss)
