import torch

class BaseDataset:
    # Base dataset class if you want to add more datasets
    def __init__(self, args, seed=42, split="train", only_one_type=None):
        self.args = args
        self.split = split
        self.only_one_type = only_one_type
        if args.data_seed is not None:
            seed = args.data_seed
        if split == "train":
            self.seed = seed-1
        elif split == "val":
            self.seed = seed
        elif split == "test":
            self.seed = seed+1
        else:
            raise ValueError(f"Unknown value for split. Got {split}")

        self.beta = args.beta
        self.num_points = args.num_points # number of data points
        self.num_vars = args.num_vars # number of variables

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx], self.inter[idx]

    def __len__(self):
        return len(self.data)

    def to(self, device):
        self.data = self.data.to(device)
        self.labels = self.labels.to(device)
        self.inter = self.inter.to(device)

    def _verify_intervention_values(self, values):
        if values is None:
            return tuple([None] * self.num_vars)
        assert isinstance(values, (list, tuple))
        assert len(values) == self.num_vars, f"Either provide a list with {self.num_vars} values or None"

        atleast_one_not_none = False
        num_points = None
        for v in values:
            assert v is None or isinstance(v, torch.Tensor), \
                    f"Each value must be a torch.Tensor or None. Got {type(v)}"
            if v is not None:
                atleast_one_not_none = True
                if num_points is None:
                    num_points = v.shape[0]
                else:
                    assert num_points == v.shape[0], "All tensors must have the same number of data points"
        assert atleast_one_not_none, "Not all values can be None"

        return values

    def _generate_labels(self, start=None, num_points=None, device=None):
        raise NotImplementedError

    def _label_to_data(self, label):
        raise NotImplementedError

    def _generate_data(self):
        raise NotImplementedError

    def get_adj_matrix(self):
        return self.adj_matrix
