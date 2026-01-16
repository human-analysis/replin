import os
import sys
import torch
import numpy as np
import json

def write_to_json(data_dict, filename):
    key_list = list(data_dict.keys())
    for key in key_list:
        v = data_dict[key]
        if isinstance(v, torch.Tensor):
            data_dict[key] = v.item()
    with open(filename, 'w') as f:
        json.dump(data_dict, f, sort_keys=True, indent=4)

    return 0

class Logger:
    def __init__(self, name, K, logfolder, args, logger_type="wandb"):
        self.K = K
        self.logfolder = logfolder
        self.json_file = os.path.join(logfolder, "results.json")
        self.data_dict = {} # dictionary of lists

        if logger_type == "wandb":
            # Setup wandb logging
            import wandb
            wandb.init(project="RepLIn", dir=logfolder,
                    name=name, config=args)
        elif logger_type == "tensorboard":
            os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
            import tensorflow as tf
            tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir=logfolder)

        self.logger_type = logger_type
    
    def save_weights(self, weights, global_step):
        torch.save(weights, os.path.join(self.logfolder, f"ckpt_{global_step}.ckpt"))

    def __call__(self, key, value, global_step=0):
        if isinstance(value, (list, dict, tuple)):
            raise ValueError("Pass a scalar value to the logger. Got ", type(value))

        existing_keys = list(self.data_dict.keys())
        if isinstance(value, torch.Tensor):
            try:
                value = value.item()
            except ValueError as e:
                if e == "only one element tensors can be converted to Python scalars":
                    print("Passed a non-scalar tensor to the logger for key", key)
                raise ValueError(e)

        if self.logger_type == "wandb":
            wandb.log({key: value}, step=global_step)
        elif self.logger_type == "tensorboard":
            self.writer.add_scalar(key, value, global_step=global_step)

        if key in existing_keys:
            if len(self.data_dict[key]) >= self.K:
                self.data_dict[key].pop(0)
            self.data_dict[key].append(value)
        else:
            self.data_dict[key] = [value]

    def write(self):
        # Write the values to the json file
        avg_results = {k: np.mean(v) if isinstance(v, list) else v for k, v in self.data_dict.items()}
        write_to_json(avg_results, self.json_file)

        return 0

    def print(self):
        # Print the values in self.data_dict.
        avg_results = {k: np.mean(v) if isinstance(v, list) else v for k, v in self.data_dict.items()}

        json.dump(avg_results, sys.stdout, sort_keys=True, indent=4)

        return 0
