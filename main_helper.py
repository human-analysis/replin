import os
from datetime import datetime

import torch

import methods
import losses
import dep_losses
from main_utils import AlternateDataloader
from logger import Logger

from main_utils import save_args_src

def get_dataloaders(args):
    from main_utils import create_train_dataloader, \
                           create_val_test_dataloaders

    # Create datasets and dataloaders for 2 variable datasets
    print("Creating datasets and dataloaders:", args.dataset, "with beta =", args.beta)
    if args.mixed_dataset:
        # Obs data contain both obs and int data
        print("** Creating mixed dataset")
        obs_train_dataloader = create_train_dataloader(args, only_one_type=None)
    else:
        # Obs data contain only obs data
        print("** Creating observational dataset")
        obs_train_dataloader = create_train_dataloader(args, only_one_type="obs")
 
    # Get training datasets with only interventional data is beta < 1
    # and dataset is not mixed.
    if args.beta < 1 and not args.mixed_dataset:
        print("** Creating interventional dataset")
        int_train_dataloader = create_train_dataloader(args, only_one_type="int")
        dataloader_tuple = tuple([obs_train_dataloader, int_train_dataloader])   
    else:
        dataloader_tuple = tuple([obs_train_dataloader])

    # Get validation and test dataloaders
    val_dataloader, test_dataloader = create_val_test_dataloaders(args)
    # AlternateDataloader alternates between batches with only obs and
    # int data.
    train_dataloader = AlternateDataloader(dataloader_tuple, args)

    return train_dataloader, val_dataloader, test_dataloader

def get_model(args):
    # Create model
    print("Creating model:", args.model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = getattr(methods, args.model_name)(args)
    
    # Load a checkpoint if provided
    if args.eval_ckpt is not None:
        print("Loading checkpoint:", args.eval_ckpt)
        model.load_state_dict(torch.load(args.eval_ckpt))
        args.epochs = 1

    model.to(device)

    return model

def get_loss_functions(args):
    # Create loss functions

    # Main predictive loss function
    if getattr(losses, args.loss_function, None) is None:
        raise ValueError("Loss function not found:", args.loss_function)
    criterion = getattr(losses, args.loss_function)(args)

    # Dependence loss function for training.
    trn_dep_fn_name = getattr(dep_losses, args.trn_dep_fn)
    if args.linear_dep == 1 and args.trn_dep_fn in ["HSIC", "KCC"]:
        print("!! Enforcing linear independence in dep function")
        trn_dep_fn = trn_dep_fn_name(k1_type="Linear", k2_type="Linear")
    else:
        trn_dep_fn = trn_dep_fn_name()
    
    # Self-dependence loss function for training. We use linear kernels
    # for this.
    trn_self_dep_fn_name = getattr(dep_losses, args.trn_self_dep_fn)
    if args.linear_selfdep == 1 and args.trn_self_dep_fn in ["HSIC", "KCC"]:
        print("!! Enforcing linear independence in self-dep fn")
        trn_self_dep_fn = trn_self_dep_fn_name(k1_type="Linear", k2_type="Linear")
    else:
        trn_self_dep_fn = trn_self_dep_fn_name()

    # Dependence loss function for evaluation.
    eval_dep_fn_name = getattr(dep_losses, args.eval_dep_fn)
    eval_dep_fn = eval_dep_fn_name()

    return criterion, trn_dep_fn, trn_self_dep_fn, eval_dep_fn

def get_optimizer(args, model):
    optimizer_class = getattr(torch.optim, args.optimizer["name"])
    optimizer = optimizer_class(model.parameters(), lr=args.lr,
                                **args.optimizer["args"])
    if args.scheduler is not None:
        if len(args.scheduler) != 0:
            scheduler_class = getattr(torch.optim.lr_scheduler, args.scheduler["name"])
            scheduler = scheduler_class(optimizer,
                                        **args.scheduler["args"])
        else:
            scheduler = None
    else:
        scheduler = None

    return optimizer, scheduler

def get_logger(args, avg_K):
    # Setup logging
    if args.expt_name != "":
        expt_name = args.expt_name
    else:
        # put the timestamp as the experiment name
        expt_name = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Name the log folder
    log_name = args.model_name
    if args.mixed_dataset:
        log_name += "-mixed"
    log_name += f"-{args.dataset}-{args.seed}-{expt_name}"

    log_folder = os.path.join(args.logs_folder, log_name)
    print("Logs folder name:", os.path.basename(log_folder))
    os.makedirs(log_folder, exist_ok=True)
    # The code supports wandb and tensorboard loggers.
    logger = Logger(log_name, avg_K, log_folder, args, logger_type="tensorboard")

    # Save the args and the source code as a zipped folder in the log
    # folder before you start training so that you can reproduce the run
    # later even if the code changes. Only python and .ini files are
    # saved. So it will not occupy too much space.
    save_args_src(args, log_folder)

    return logger, log_folder

def save_tensor(tensor, folder, name):
    torch.save(tensor, os.path.join(folder, name))
