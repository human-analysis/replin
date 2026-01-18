import torch
import torch.nn as nn

import architectures as arch
import datasets

class ERM(nn.Module):
    def __init__(self, model_args):
        super().__init__()
        self.num_vars = model_args.num_vars
        self.seed = model_args.seed
        dataset = getattr(datasets, model_args.dataset)
        self.data_props = dataset.get_all_properties()
        self.num_classes = self.data_props.num_classes
        self.normalize_feats = True if model_args.norm_features == 1 else False
        if self.normalize_feats:
            print("** Normalizing features")

        if isinstance(self.num_classes, int):
            self.num_classes = [self.num_classes for _ in range(self.num_vars)]

        if model_args.inp_dim is not None:
            self.inp_dim = model_args.inp_dim
        else:
            self.inp_dim = self.data_props.data_dim

        # Architectures that learn features from the input directly
        self.features = nn.ModuleList()
        assert model_args.features is not None, "Feature architecture not specified"
        for i in range(self.num_vars):
            if len(model_args.features["args"]) == 0:
                model_args.features["args"] = [self.inp_dim]
                fd = self.inp_dim
            else:
                fd = model_args.features["args"][-1]
            feat_model_class = getattr(arch, model_args.features["name"])
            self.features.append(feat_model_class([self.inp_dim,
                                                   *model_args.features["args"]],
                                                   seed=self.seed+2+i))

        # To predict output from the features
        self.head = nn.ModuleList()
        for i in range(self.num_vars):
            self.head.append(arch.MLP([fd, self.num_classes[i]], seed=self.seed+i))
        
        rng = torch.get_rng_state()
        torch.manual_seed(self.seed*10+1203) # random
        self._init_modules()
        torch.set_rng_state(rng)

    def _init_modules(self):
        for f in self.features:
            f.init_layers()
        for h in self.head:
            h.init_layers()

    def forward(self, x, y, int_pos):
        feats = [_(x) for _ in self.features]
        if self.normalize_feats:
            feats = [_/torch.norm(_, dim=-1, keepdim=True) for _ in feats]
        out = [_(f) for f, _ in zip(feats, self.head)]

        return out, feats
