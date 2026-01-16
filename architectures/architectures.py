import torch
import torch.nn as nn
import numpy as np

class MLP(nn.Module):
    def __init__(self, sizes, act="ReLU", seed=42, hc_inp=None):
        # hc_inp is hard-coded input size for the model. This is for
        # ease in setting the input size of the model according to the
        # causal graph.
        super().__init__()
        if hc_inp is not None:
            sizes[0] = hc_inp
        l = len(sizes)
        layers = []
        for _ in range(l-1):
            layers.append(nn.Linear(sizes[_], sizes[_+1]))
            if _ < l-2:
                if act == "LeakyReLU":
                    layers.append(nn.LeakyReLU(0.1))
                else:
                    layers.append(getattr(nn, act)())
        self.layers = nn.Sequential(*layers)
        self.seed = seed
        self.out_dim = sizes[-1]
        self.init_layers()

    def init_layers(self):
        rng = torch.get_rng_state()
        torch.manual_seed(self.seed)
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)
        torch.set_rng_state(rng)

    def forward(self, x):
        x = self.layers(x)
        return x

    def get_out_dim(self):
        return self.out_dim

class MultiMLP(nn.Module):
    def __init__(self, *sizes, act="ReLU", seed=42, hc_inp=None):
        # hc_inp is hard-coded input size for the model. This is for
        # ease in setting the input size of the model according to the
        # causal graph.
        super().__init__()
        self.inp_sizes = [_[0] for _ in sizes]
        self.inp_sizes = np.cumsum(self.inp_sizes)
        self.out_dim = sizes[0][-1]

        self.mlp_list = nn.ModuleList([MLP(_, act, (i+1)*seed, hc_inp) for i, _ in enumerate(sizes)])
        for i, mlp in enumerate(self.mlp_list):
            mlp.init_layers()

    def forward(self, x):
        out = torch.zeros(x.shape[0], self.out_dim, device=x.device)
        for i, mlp in enumerate(self.mlp_list):
            if i == 0:
                out += mlp(x[:, :self.inp_sizes[i]])
            else:
                out += mlp(x[:, self.inp_sizes[i-1]:self.inp_sizes[i]])
        return out

    def get_out_dim(self):
        return self.out_dim

class Linear(nn.Module):
    def __init__(self, inp, out, bias=True, seed=42, hc_inp=None):
        # hc_inp is hard-coded input size for the model. This is for
        # ease in setting the input size of the model according to the
        # causal graph.
        super().__init__()
        if hc_inp is not None:
            inp = hc_inp
        self.layer = nn.Linear(inp, out, bias=bias)
        self.seed = seed
        self.out_dim = out
        self.init_layers()

    def init_layers(self):
        rng = torch.get_rng_state()
        torch.manual_seed(self.seed)
        nn.init.xavier_normal_(self.layer.weight)
        if self.layer.bias is not None:
            nn.init.zeros_(self.layer.bias)
        torch.set_rng_state(rng)

    def forward(self, x):
        x = self.layer(x)
        return x

    def get_out_dim(self):
        return self.out_dim

class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, *args):
        if len(args) == 1:
            return args[0]
        return args

    def get_out_dim(self):
        return None
