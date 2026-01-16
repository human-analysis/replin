import datetime
import os
import json
import shutil
import torch
from argparse import Namespace
from copy import deepcopy

# NOTE: datasets is also the name of a pip package associated with
# hugging face. So if you get an error, it might be due to that.
import datasets


def ch(x):
    return chr(65+x)

class AlternateDataloader:
    def __init__(self, dataloader_tuple, args):
        self.num_loaders = len(dataloader_tuple)
        self.batch_idx = 0
        self.data_iterators = [iter(dataloader) for dataloader in dataloader_tuple]
        self.dataloader_tuple = dataloader_tuple
        self.beta = args.beta
        if self.beta < 1:
            # This is super-sampling
            self.num_batches = self.num_loaders * max([len(dataloader)
                                                       for dataloader in dataloader_tuple])
        else:
            self.num_batches = len(dataloader_tuple[0])

    def reset_iterators(self):
        self.batch_idx = 0
        self.data_iterators = [iter(dataloader) for dataloader in self.dataloader_tuple]

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        return self

    def __next__(self):
        if self.batch_idx >= self.num_batches:
            self.reset_iterators()
            raise StopIteration
    
        if self.beta == 1:
            iter_idx = 0
        else:
            iter_idx = self.batch_idx % self.num_loaders

        try:
            batch = next(self.data_iterators[iter_idx])
        except StopIteration:
            # When you call 'iter()' on a generator, it returns an
            # iterator and is same as invoking __iter__ method.
            # Shuffling in the dataloader is carried in the __iter__
            # method via the __iter__ method of its underlying sampler.
            # Check
            # https://discuss.pytorch.org/t/about-the-details-of-shuffle-in-dataloader/39487/
            # for more details.
            self.data_iterators[iter_idx] = iter(self.dataloader_tuple[iter_idx])
            batch = next(self.data_iterators[iter_idx])

        self.batch_idx += 1

        # We return (self.batch_idx - 1) % self.num_loaders since we already
        # incremented self.batch_idx
        return (self.batch_idx-1) % self.num_loaders, batch

    def get_adj_matrix(self):
        return self.dataloader_tuple[0].dataset.adj_matrix

# Pass the same seed to train, val and test dataloaders. The seeds are
# adjusted internally in the dataset class based on the split.
def create_val_test_dataloaders(args):
    # Load the data directly to GPU for faster evaluation
    print("Loading eval data directly to GPU")
    device = "cuda"

    print("** Creating validation dataset")
    val_dataset = getattr(datasets, args.dataset)(args, split="val", seed=args.seed,
                                                  only_one_type=None)
    print("** Creating test dataset")
    test_dataset = getattr(datasets, args.dataset)(args, split="test", seed=args.seed,
                                                   only_one_type=None)

    # num_workers must be zero if the data is directly loaded to GPU
    val_dataset.to(device)
    test_dataset.to(device)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size,
                                                    shuffle=False, num_workers=0,
                                                    drop_last=False)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size,
                                                    shuffle=False, num_workers=0,
                                                    drop_last=False)

    return val_dataloader, test_dataloader

def create_train_dataloader(args, only_one_type=None):
    # Load the data directly to GPU for faster evaluation
    print("Loading train data directly to GPU")
    device = "cuda"

    train_dataset = getattr(datasets, args.dataset)(args, split="train", seed=args.seed,
                                                    only_one_type=only_one_type)

    # num_workers must be zero if the data is directly loaded to GPU
    train_dataset.to(device)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                                    shuffle=True, num_workers=0,
                                                    drop_last=False)

    return train_dataloader

def convert_namespace_to_dict(args):
    args_dict = vars(args)
    for k, v in args_dict.items():
        if isinstance(v, Namespace):
            args_dict[k] = convert_namespace_to_dict(v)

    return args_dict

def save_args_src(args, log_folder):
    args_copy = deepcopy(args)
    args_dict = convert_namespace_to_dict(args_copy)

    # args as a JSON file
    with open(os.path.join(log_folder, "args.json"), 'w') as fp:
        json.dump(args_dict, fp, sort_keys=True, indent=4)

    # Get your current location
    src_dir = os.getcwd()
    # Name of zip folder
    src_name = "src"
    # Create a temporary folder to keep the source
    dst_dir = os.path.join(log_folder, src_name)

    os.makedirs(dst_dir, exist_ok=True)

    curr_dirname = os.getcwd().split('/')[-1]
    # file formats that you want
    file_formats = ('.py', '.ini')

    # Create a new folder, and copy the required files to it.
    for root, subdirs, files in os.walk(src_dir):
        for f in files:
            if f.endswith(file_formats) and log_folder not in root:
                root_ = root.split(curr_dirname)[1][1:]
                src_path = os.path.join(root, f)
                dst_path = os.path.join(dst_dir, root_)
                os.makedirs(dst_path, exist_ok=True)
                # Copy the required files
                shutil.copy(src=src_path, dst=dst_path)

    # Write the time stamp
    with open(os.path.join(dst_dir, 'timestamp'), 'w') as f:
        now = datetime.datetime.now()
        ts = "{:04d}-{:02d}-{:02d}, {:02d}:{:02d}:{:02d}".format(now.year, now.month,
                                                                now.day, now.hour,
                                                                now.minute, now.second)
        f.write(ts+'\n')

    # Zip the folder
    shutil.make_archive(dst_dir, format='zip', root_dir=log_folder, base_dir=src_name)
    # Remove the temporary folder
    shutil.rmtree(dst_dir)
