import argparse
import configparser
import json
import os
from argparse import Namespace
from pprint import pprint

def parse_args():
    parser = argparse.ArgumentParser(description='Representation Learning from Interventional Data')

    # the following two parameters can only be provided at the command line.
    parser.add_argument("--c", dest="config", help="Specify a config file")
    parser.add_argument('--data_conf', help="Dataset config", default=None)

    # Read just --c and --data_conf from command line, and ignore other
    # args
    args, remaining_argv = parser.parse_known_args()

    ## General arguments
    parser.add_argument('--ckpt_folder', type=str, default="ckpts",
                        help="Folder to save checkpoints. The base folder is set in the code. See below.")
    parser.add_argument('--eval_ckpt', type=str, default=None,
                        help="Path to a checkpoint to be evaluated")
    parser.add_argument('--expt_name', type=str, default="",
                        help="Name of the experiment if you want to label individual runs")
    parser.add_argument('--logs_folder', type=str, default=None,
                        help="Folder to save logs")
    parser.add_argument('--seed', type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument('--data_seed', type=int, default=None,
                        help="Random seed for data generation")
    parser.add_argument('--no_eval', type=int, default=0,
                        help="If 1, do not run the final testing after training")

    ## Model arguments
    parser.add_argument('--model_name', type=str,
                        help="Name of the model to be used (ERM or RepLIn)")
    parser.add_argument('--features', type=json.loads, default=None,
                        help="JSON dict of feature architecture name and its arguments")
    parser.add_argument('--norm_features', type=int, default=0,
                        help="If 1, normalize features to unit hypersphere")

    ## Data arguments
    parser.add_argument('--batch_size', type=int,
                        help="Batch size for training")
    parser.add_argument('--beta', type=float, default=0.5,
                        help="Proportion of interventional data in the dataset")
    parser.add_argument('--data_args', type=json.loads,
                        help="JSON dict of data arguments.")
    parser.add_argument('--dataset', type=str,
                        help="Name of the dataset to be used")
    parser.add_argument('--data_loc', type=str, default=None,
                        help="Location of the data")
    parser.add_argument('--inp_dim', type=int, default=None,
                        help="Input feature dimension")
    parser.add_argument('--num_points', type=int,
                        help="Number of data points in the dataset")
    parser.add_argument('--num_vars', type=int,
                        help="Number of variables in the dataset")
    parser.add_argument('--workers', type=int, default=10,
                        help="Number of data loader workers")

    ## Training arguments
    parser.add_argument('--epochs', type=int,
                        help="Number of training epochs")
    parser.add_argument('--eval_dep_fn', type=str, default="KCC",
                        help="Dependency function to be used during evaluation")
    parser.add_argument('--eval_every', type=int,
                        help="Evaluate every N epochs")
    parser.add_argument('--loss_function', type=str, default="CELossOnList",
                        help="Predictive loss function to be used during training")
    parser.add_argument('--lr', type=float,
                        help="Learning rate for optimizer")
    parser.add_argument('--weight_decay', type=float,
                        help="Weight decay for optimizer")
    parser.add_argument('--trn_dep_fn', type=str, default="HSIC",
                        help="Dependence function to be used during training")
    parser.add_argument('--trn_self_dep_fn', type=str, default="HSIC",
                        help="Self-dependence function to be used during training")
    parser.add_argument('--linear_dep', type=int, default=0,
                        help="If 1, use linear kernels in dep function")
    parser.add_argument('--linear_selfdep', type=int, default=1,
                        help="If 1, use linear kernels in self-dep function")
    parser.add_argument('--optimizer', type=json.loads,
                        default={"name": "Adam", "args": {}},
                        help="JSON dict of optimizer name and its arguments")
    parser.add_argument('--scheduler', type=json.loads, default=None,
                        help="JSON dict of scheduler name and its arguments")
    parser.add_argument('--start_dep', type=float, default=0,
                        help="Proportion of training epochs until which lam_dep is 0")
    parser.add_argument('--end_dep', type=float, default=1,
                        help="Proportion of training epochs by which lam_dep reaches its max value")
    parser.add_argument('--mixed_dataset', action='store_true',
                        help="Mix interventional and observational data")
    
    ## Loss weights
    parser.add_argument('--lam_dep', type=float, default=0,
                        help="Weight for dependence loss. Default means ERM.")
    parser.add_argument('--lam_pred', type=float, default=1,
                        help="Weight for prediction loss.")
    parser.add_argument('--lam_self', type=float, default=0,
                        help="Weight for self-dependence loss. Default means ERM.")

    # Read arguments from config file
    config_file_copy = args.config
    if os.path.exists(args.config):
        config = configparser.ConfigParser()
        config.read([args.config])
        defaults = dict(config.items("Arguments"))
        # if data_conf is passed from terminal, it overwrites the
        # data_conf in config file
        if args.data_conf is not None:
            defaults.pop("data_conf")
        parser.set_defaults(**defaults)
    else:
        print(f"!! Couldn't find the provided config file. {args.config}")

    # If data args were not found in the general config.
    if args.data_conf is None:
        args.data_conf = defaults.get("data_conf")

    ########
    # Helper functions for parsing the values from config
    def islist(s:str):
        return (s[0] == '[') and (s[-1] == ']')

    def isfloat(s:str):
        s = s.replace(".", "")
        return s.isdigit()

    def parse_list(s:str):
        # TODO: Parse nested lists
        # TODO: List of non-numbers
        if s[0] == '[' and s[1] == '[':
            raise ValueError("Nested lists not supported")
        
        s = s[1:-1] # remove the brackets

        if any([_.isalpha() for _ in s]):
            raise ValueError("List of non-numbers not supported")
    
        s = s.replace(" ", "")
        s = s.split(",")
        if isfloat(s[0]) and s[0].count(".") > 0:
            return [float(x) for x in s]
        elif s[0].isdigit():
            s = [int(x) for x in s]
        else:
            raise ValueError(f"Got unknown content type. {s[0]}")
        return s
    ########

    # Read data args from data config file
    if os.path.exists(args.data_conf):
        config = configparser.ConfigParser()
        config.read([args.data_conf])
        data_defaults = dict(config.items("Arguments"))
        dataset = data_defaults.pop("dataset")
        num_vars = data_defaults.pop("num_vars")
        num_points = data_defaults.pop("num_points")

        # Convert all elements to float or int if possible
        for k, v in data_defaults.items():
            try:
                if v.isdigit():
                    data_defaults[k] = int(v)
                elif islist(v):
                    data_defaults[k] = parse_list(v)
                elif isfloat(v):
                    data_defaults[k] = float(v)
                else:
                    raise ValueError(f"Unknown type. Got {type(v)} for {k}")
            except ValueError:
                pass
        parser.set_defaults(data_args=data_defaults, dataset=dataset,
                            num_vars=num_vars, num_points=num_points)
        data_conf_copy = args.data_conf

    # Read remaining args from command line
    args = parser.parse_args(remaining_argv)
    combined_data_args = data_defaults | args.data_args
    args.data_args = Namespace(**combined_data_args)
    args.config = config_file_copy
    args.data_conf = data_conf_copy

    # Verify the args and preprocess the args accordingly
    args = preprocess_args(args)

    pprint(args.__dict__, indent=2, sort_dicts=True)

    return args

def preprocess_args(args):
    """
    Verify the args and preprocess the args accordingly
    """
    # Convert all string boolean args to bool
    def convert_str_to_bool(d):
        for k, v in d.items():
            if isinstance(v, str):
                if v.lower() == "true":
                    d[k] = True
                elif v.lower() == "false":
                    d[k] = False
            elif isinstance(v, dict):
                d[k] = convert_str_to_bool(v)

        return d

    # The code was written with beta being the proportion of
    # observational data. In the paper, beta is defined as the
    # proportion of interventional data. So we need to adjust it here.
    args.beta = 1 - args.beta

    # the base folder where you want to save logs and checkpoints. This
    # removes the need to specify full paths for logs and checkpoints.
    base_folder = "."
    
    if args.logs_folder is None:
        args.logs_folder = f"logs_{args.dataset}"

    args.logs_folder = os.path.join(base_folder, args.logs_folder)
    args.ckpt_folder = os.path.join(base_folder, args.ckpt_folder)
    os.makedirs(base_folder, exist_ok=True)

    args.__dict__ = convert_str_to_bool(args.__dict__)

    # When mixed_dataset is used, both observational and interventional
    # data appears as observational data. So we need to adjust the
    # number of epochs accordingly to ensure that the model sees enough
    # interventional data.
    if args.mixed_dataset:
        print("** Adjusting the number of epochs for mixed batch training.")
        args.epochs = int(2*args.beta * args.epochs)

    if getattr(args.data_args, "task_type", None) is not None:
        args.task_type = args.data_args.task_type

    return args
