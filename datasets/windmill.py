import math
import torch

from .base import BaseDataset
from .factory import labels_from_2var_graph

class DummyArgs(object):
    def __init__(self):
        pass

class Windmill(BaseDataset):
    @staticmethod
    def get_all_properties():
        props = DummyArgs()
        adj_matrix = torch.zeros(2, 2)
        adj_matrix[0, 1] = 1
        props.adj_matrix = adj_matrix
        props.data_dim = 2
        props.num_classes = 2
        props.type = "discrete"

        return props

    def __init__(self, args, seed=42, split="train", only_one_type=None):
        super().__init__(args, seed, split, only_one_type)

        # Dataset specific arguments. See App. I in the paper.
        data_args = args.data_args
        self.r_max = data_args.r_max # maximum radius
        self.num_arms = data_args.num_arms # number of arms per class
        self.th_wid = 2*math.pi/(2*self.num_arms)*0.9 # fractional width of each arm
        self.max_th_offset = data_args.max_th_offset # maximum theta offset based on r.
        self.offset_wavelength = data_args.offset_wavelength # adjust the complexity of the windmill arms

        self._generate_data()

    def _generate_labels(self, start=None, num_points=None, device=None):
        a, b = self._verify_intervention_values(start)
        if num_points is None:
            num_points = self.num_points

        a, b = labels_from_2var_graph(a, b, num_points, device)

        return (a, b)

    def _label_to_data(self, label):
        A, B = label
        num_p = A.shape[0]

        # Convert A, B to X.
        th_A0 = torch.linspace(0, 2*math.pi, self.num_arms+1)[:-1]
        th_A1 = torch.linspace(0, 2*math.pi, self.num_arms+1)[:-1] + math.pi/self.num_arms
        # Choose a random arm for A=0 from possible arms. Likewise for A=1.
        th_A0 = th_A0[torch.randint(self.num_arms, (num_p,))]
        th_A1 = th_A1[torch.randint(self.num_arms, (num_p,))]

        # beta distribution with alpha=1, beta=3
        beta_dist = torch.distributions.beta.Beta(1, 2.5)

        # Sample r according to B. If B=0, sample a small r, else sample a large r.
        # r ranges from 0 to r_max
        B0_r = beta_dist.sample(torch.Size([num_p])) * self.r_max/2.
        B1_r = self.r_max - beta_dist.sample(torch.Size([num_p])) * self.r_max/2.
        r = B * B0_r + (1-B) * B1_r

        # Sample theta according to A.
        # Choose the theta arm according to A and then sample from this arm using a uniform distribution.

        # First we will have a cartwheel
        theta = torch.rand(num_p)*self.th_wid + th_A0*(1-A) + th_A1*A - self.th_wid/2.

        # Add an offset to theta according to r.
        th_offset_mod = torch.sin((r/self.r_max)*self.offset_wavelength*math.pi)
        th_offset = self.max_th_offset*th_offset_mod
        theta += th_offset

        # Convert to Cartesian coordinates
        x1 = r*torch.cos(theta)
        x2 = r*torch.sin(theta)

        data = torch.stack([x1, x2], dim=1)
        labels = torch.stack([A, B], dim=1).type(torch.long)

        return data, labels

    def _generate_data(self):
        rng = torch.get_rng_state()
        torch.manual_seed(self.seed)
        if self.split == "train":
            num_obs_points = int(self.beta*self.num_points)
            num_int_points = self.num_points-int(self.beta*self.num_points)
        else:
            num_obs_points = self.num_points//2
            num_int_points = self.num_points//2

        if self.only_one_type is None or self.split in ["val", "test"]:
            # Generate observational data
            labels = self._generate_labels(num_points=num_obs_points)
            obs_data, obs_labels = self._label_to_data(labels)
            obs_inter = torch.zeros(num_obs_points, 2)

            # Generate interventional data
            b_int = torch.bernoulli(0.5*torch.ones(num_int_points))
            labels = self._generate_labels(start=[None, b_int])
            int_data, int_labels = self._label_to_data(labels)
            int_inter = torch.zeros(num_int_points, 2)
            int_inter[:, 1] = 1

            self.data = torch.cat([obs_data, int_data], dim=0)
            self.labels = torch.cat([obs_labels, int_labels], dim=0)
            self.inter = torch.cat([obs_inter, int_inter], dim=0)

        elif self.only_one_type == "obs":
            labels = self._generate_labels(num_points=num_obs_points)
            self.data, self.labels = self._label_to_data(labels)
            self.inter = torch.zeros(num_obs_points, 2)

        elif self.only_one_type == "int":
            b_int = torch.bernoulli(0.5*torch.ones(num_int_points))
            labels = self._generate_labels(start=[None, b_int])
            self.data, self.labels = self._label_to_data(labels)
            self.inter = torch.zeros(num_int_points, 2)
            self.inter[:, 1] = 1

        # Mix data to get both observational and interventional
        # data in each batch (approximately).
        perm = torch.randperm(self.data.shape[0])
        self.data = self.data[perm]
        self.labels = self.labels[perm]
        self.inter = self.inter[perm]

        torch.set_rng_state(rng)

        # Save the adjacency matrix
        self.adj_matrix = torch.zeros(2, 2)
        self.adj_matrix[0, 1] = 1
