import os
import torch
from tqdm import trange, tqdm

from validater import validate
from main_utils import ch


def train_one_epoch(model, train_dataloader, epoch,
                    criterion, trn_dep_fn, trn_self_dep_fn,
                    optimizer, logger, device,
                    start_dep, end_dep, args, global_step):
    model.train()

    # Adjacency matrix to find parents of the intervened nodes
    adj_matrix = train_dataloader.get_adj_matrix()

    # Warmup lam_dep
    if epoch > start_dep:
        lam_dep = args.lam_dep*((epoch - start_dep)/(end_dep - start_dep))
        lam_dep = min(lam_dep, args.lam_dep)
    else:
        lam_dep = 0

    for _, batch in tqdm(train_dataloader, total=len(train_dataloader), leave=False):
        x, y, int_pos = batch

        if not x.is_cuda:
            x = x.to(device)
            y = y.to(device)
            int_pos = int_pos.to(device)

        out, feats = model(x, y, int_pos)

        # We now return the pred losses for each variable separately.
        loss_pred_ = criterion(out, y) # predictive loss
        loss_pred = torch.mean(torch.stack(loss_pred_))
        if torch.sum(int_pos) == 0:
            logger("trn_pred_obs/total", loss_pred.item(), global_step)
            for l_idx, l in enumerate(loss_pred_):
                logger(f"trn_pred_obs/{ch(l_idx)}", l.item(), global_step)
        else:
            # Variable "v" identifies the intervened variable in this batch
            v = torch.mean(int_pos, 0).argmax().item()
            logger(f"trn_pred_int{ch(v)}/total", loss_pred.item(), global_step)
            for l_idx, l in enumerate(loss_pred_):
                logger(f"trn_pred_int{ch(v)}/{ch(l_idx)}", l.item(), global_step)

        # Get the training accuracy on this batch
        with torch.no_grad():
            for o_idx, o in enumerate(out):
                pred = torch.argmax(o, dim=1)
                perf = torch.mean((pred == y[:, o_idx]).type(torch.float))
                if torch.sum(int_pos) == 0:
                    logger(f"trn_perf_obs/{ch(o_idx)}", perf, global_step)
                else:
                    v = torch.mean(int_pos, 0).argmax().item()
                    logger(f"trn_perf_int{ch(v)}/{ch(o_idx)}", perf, global_step)

        # Additional independence losses, only for RepLIn
        if args.model_name == "RepLIn":
            # Compute self-dependence loss for both obs and int
            if args.lam_self > 0:
                if torch.sum(int_pos) == 0:
                    prefix = "obs"
                else:
                    prefix = f"int{ch(torch.mean(int_pos, 0).argmax().item())}"
                loss_self = 0
                for f_idx, feat in enumerate(feats):
                    i_fi_hsic = trn_self_dep_fn(feat, y[:, f_idx].reshape(-1, 1).type(torch.float))
                    logger(f"trn_self/{prefix}_{ch(f_idx)}F{ch(f_idx)}", 1.-i_fi_hsic.item(), global_step)
                    loss_self = loss_self + i_fi_hsic

                loss_self = 1 - loss_self/len(feats)
                logger(f"trn_self/{prefix}_total", loss_self.item(), global_step)
            else:
                loss_self = torch.zeros_like(loss_pred)

            # Compute dependence loss only on interventional data.
            if torch.sum(int_pos) > 1 and epoch >= start_dep:
                """
                We will intervene on only one variable in each batch.
                The code is however general in case we decide to extend
                to multiple interventions per batch.
                """
                # Get list of intervened variables in this batch
                int_vars = torch.nonzero(torch.sum(int_pos, 0)).reshape(-1)

                # Iterate through the list of intervened variables
                loss_dep = torch.zeros_like(loss_pred)
                for v in int_vars:
                    # Get interventional feature
                    feat_int = feats[v][int_pos[:, v] == 1]

                    # Get parents of v.
                    par_v = torch.nonzero(adj_matrix[:, v] == 1)
                    if len(par_v) == 0:
                        continue

                    # Collect parent features where v is intervened
                    loss_dep_ = 0
                    abs_dep = 0
                    for p in par_v:
                        feat_par = feats[p][int_pos[:, v] == 1]

                        # Compute dependence loss
                        loss_dep_temp = trn_dep_fn(feat_int, feat_par)
                        loss_dep_ = loss_dep_ + loss_dep_temp
                        abs_dep = abs_dep + loss_dep_temp

                    loss_dep_ = loss_dep_ / len(par_v)
                    abs_dep = abs_dep / len(par_v)
                    logger(f"trn_dep_loss/scaled_{ch(v)}", lam_dep*loss_dep_.item(), global_step)
                    logger(f"trn_dep_loss/abs_{ch(v)}", abs_dep.item(), global_step)

                    loss_dep = loss_dep + loss_dep_

                loss_dep = lam_dep*loss_dep/len(int_vars)
                logger("trn_dep_loss/total", loss_dep.item(), global_step)
            else:
                loss_dep = torch.zeros_like(loss_pred)
        else:
            # placeholder zero losses for ERM
            loss_dep = torch.zeros_like(loss_pred)
            loss_self = torch.zeros_like(loss_pred)

        # Compute the final loss and backpropagate
        loss = args.lam_pred*loss_pred + \
                lam_dep*loss_dep + \
                args.lam_self*loss_self

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        logger("trn_loss", loss.item(), global_step)

        # Stop training if you get NaN losses
        if torch.isnan(loss):
            if torch.isnan(loss_pred):
                raise ValueError("Loss pred is NaN")
            if torch.isnan(loss_dep):
                raise ValueError("Loss dep is NaN")

        global_step += 1

    return global_step

def train(args, model, train_dataloader, val_dataloader, criterion,
          logger, log_folder, trn_dep_fn, trn_self_dep_fn,
          eval_dep_fn, optimizer, device, scheduler, avg_K):
    # Setup training args
    global_step = 0
    start_dep = args.start_dep*args.epochs
    end_dep = args.end_dep*args.epochs

    # Print the number of trainable and non-trainable params just to be
    # sure that it's doing what we expect.
    num_trainable = 0
    num_nontrainable = 0
    for p in model.parameters():
        if p.requires_grad:
            num_trainable += p.numel()
        else:
            num_nontrainable += p.numel()

    print("Number of trainable params", num_trainable)
    print("Number of non-trainable params", num_nontrainable)

    # Validate once before we start training.
    with torch.no_grad():
        if args.no_eval == 0:
            validate(model, val_dataloader, criterion,
                    logger, log_folder,
                    0, eval_dep_fn, args, device,
                    global_step)
        torch.save(model.state_dict(),
                    os.path.join(log_folder, "latest_model.pt"))

    # Start training
    for epoch in trange(args.epochs):
        if epoch == args.epochs-1:
            logger.print()

        # We don't train if we are only evaluating a checkpoint
        if args.eval_ckpt is not None:
            exit()

        # Training starts
        global_step = train_one_epoch(model, train_dataloader, epoch,
                                      criterion, trn_dep_fn,
                                      trn_self_dep_fn, optimizer,
                                      logger, device, start_dep,
                                      end_dep, args, global_step)
        logger("epoch", epoch, global_step)

        # Finished one epoch
        if scheduler is not None:
            scheduler.step()
            logger("lr", scheduler.get_last_lr()[0], global_step)

        # In addition to periodic validation according to --eval_every,
        # we also validate for the last avg_K epochs to get a better
        # estimate of the validation results in the JSON file.
        # eval_or_not decides whether to evaluate at this epoch.
        eval_or_not = (epoch+1) % args.eval_every == 0 or \
                       args.eval_ckpt is not None or \
                       epoch >= args.epochs-avg_K

        if eval_or_not and args.no_eval == 0:
            with torch.no_grad():
                validate(model, val_dataloader, criterion,
                         logger, log_folder,
                         epoch, eval_dep_fn, args, device,
                         global_step)
        torch.save(model.state_dict(),
                    os.path.join(log_folder, "latest_model.pt"))

    return global_step