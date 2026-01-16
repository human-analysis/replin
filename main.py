import random
import numpy as np
import torch

from args import parse_args
from main_helper import get_dataloaders, get_model, \
                        get_loss_functions, get_optimizer, \
                        get_logger
from trainer import train
from metrics import calc_all_metrics

@torch.no_grad()
def test(model, test_dataloader, eval_dep_fn,
         device, logger, global_step):
    model.eval()

    preds = []
    labels = []
    feats = []
    int_pos_all = []

    for batch in test_dataloader:
        x, y, int_pos = batch
        if not x.is_cuda:
            x = x.to(device)
            y = y.to(device)

        outs_, feats_ = model(x, y, int_pos)

        if len(preds) == 0:
            feats = [[f.detach()] for f in feats_]
            preds = [[torch.argmax(o.detach(), 1)] for o in outs_]
        else:
            for f_idx, f in enumerate(feats_):
                feats[f_idx].append(f.detach())
            for o_idx, o in enumerate(outs_):
                preds[o_idx].append(torch.argmax(o.detach(), 1))
        
        labels.append(y)
        int_pos_all.append(int_pos)

    preds = [torch.cat(p, dim=0) for p in preds]
    feats = [torch.cat(f, dim=0) for f in feats]
    labels = torch.cat(labels, dim=0).detach().type(torch.long)
    int_pos_all = torch.cat(int_pos_all, dim=0).detach()

    calc_all_metrics(labels, feats, preds,
                     int_pos_all, eval_dep_fn, logger,
                     global_step, desc="test")

if __name__ == "__main__":
    # Collect all arguments in a namespace-like variable
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set seed everywhere
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # The metrics may be noisy. So we average over the past K steps.
    avg_K = 5

    # Build your model
    model = get_model(args)

    # Get traing and eval loss functions.
    criterion, trn_dep_fn, trn_self_dep_fn, \
        eval_dep_fn = get_loss_functions(args)
    
    # Get optimizer and scheduler
    optimizer, scheduler = get_optimizer(args, model)

    # Get dataloaders
    train_dataloader, val_dataloader, test_dataloader = get_dataloaders(args)

    # Get logger. Tensorboard or WandB
    logger, log_folder = get_logger(args, avg_K)

    # Start training
    global_step = train(args, model, train_dataloader, val_dataloader,
                        criterion, logger, log_folder, trn_dep_fn,
                        trn_self_dep_fn, eval_dep_fn, optimizer, device,
                        scheduler, avg_K)

    if args.no_eval == 0:
        # Training is done, now evaluate on test set
        test(model, test_dataloader, eval_dep_fn, device,
            logger, global_step)
    
    logger.save_weights(model.state_dict(), global_step)
