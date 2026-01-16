import torch
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

from main_helper import save_tensor
from metrics import calc_all_metrics
from main_utils import ch

def plot_2d_decision_boundary(model, device, x_min, x_max, mesh_size=100):
    X1 = torch.linspace(x_min, x_max, mesh_size, device=device)
    X2 = torch.linspace(x_min, x_max, mesh_size, device=device)

    X1_, X2_ = torch.meshgrid(X1, X2, indexing='xy')

    X1_ = X1_.reshape(-1, 1)
    X2_ = X2_.reshape(-1, 1)

    mesh_inp = torch.cat([X1_, X2_], dim=1)

    mesh_outA = []
    mesh_outB = []

    batchsize = 10

    for i in range(mesh_inp.shape[0]//batchsize):
        mesh_out_, mesh_feats = model(mesh_inp[i*batchsize:(i+1)*batchsize], None, None)
        mesh_outA.append(mesh_out_[0].detach().cpu())
        mesh_outB.append(mesh_out_[1].detach().cpu())

    mesh_outA = torch.cat(mesh_outA, dim=0)
    mesh_outB = torch.cat(mesh_outB, dim=0)
    mesh_inp = mesh_inp.detach().cpu().numpy()

    mesh_outA = torch.softmax(mesh_outA, dim=1)[:, 1].detach().cpu().numpy()
    mesh_outB = torch.softmax(mesh_outB, dim=1)[:, 1].detach().cpu().numpy()

    cmap = mpl.cm.get_cmap('RdBu')
    levels = [_ for _ in np.arange(0, 1, 0.1)]
    levels = 100

    # Plot ERM decision boundary
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    axs[0].contourf(mesh_inp[:, 0].reshape(mesh_size, mesh_size), mesh_inp[:, 1].reshape(mesh_size, mesh_size),
                    mesh_outA.reshape(mesh_size, mesh_size), levels=levels, cmap=cmap, alpha=0.4)
    axs[0].set_xlabel(r"$X_1$")
    axs[0].set_ylabel(r"$X_2$")
    axs[0].set_title(r"Decision boundary on $A$")

    axs[1].contourf(mesh_inp[:, 0].reshape(mesh_size, mesh_size), mesh_inp[:, 1].reshape(mesh_size, mesh_size),
                    mesh_outB.reshape(mesh_size, mesh_size), levels=levels, cmap=cmap, alpha=0.4)
    axs[1].set_xlabel(r"$X_1$")
    axs[1].set_ylabel(r"$X_2$")
    axs[1].set_title(r"Decision boundary on $B$")

    fig.tight_layout()

    return fig, axs


@torch.no_grad()
def validate(model, val_dataloader, criterion,
             logger, log_folder,
             epoch, eval_dep_fn, args, device,
             global_step):
    model.eval()
    # Manually calculate the accuracy for each data type
    total_loss = 0.
    count = 0.
    X = []
    outs = []
    feats = []
    labels = []
    int_pos_all = []

    total_loss_pred = None

    for batch in val_dataloader:
        x, y, int_pos = batch
        if not x.is_cuda:
            x = x.to(device)
            y = y.to(device)
            int_pos = int_pos.to(device)

        outs_, feats_ = model(x, y, int_pos)

        X.append(x)
        if len(feats) == 0:
            feats = [[f.detach()] for f in feats_]
            outs = [[o.detach()] for o in outs_]
        else:
            for f_idx, f in enumerate(feats_):
                feats[f_idx].append(f.detach())
            for o_idx, o in enumerate(outs_):
                outs[o_idx].append(o.detach())

        labels.append(y)
        int_pos_all.append(int_pos)

        loss = criterion(outs_, y)
        for l_idx in range(len(loss)):
            if len(loss[l_idx].shape) > 0:
                loss[l_idx] = torch.mean(loss[l_idx])
        if total_loss_pred is None:
            total_loss_pred = [l.item() for l in loss]
        else:
            for l_idx, l in enumerate(loss):
                total_loss_pred[l_idx] += l.item()
        loss = torch.mean(torch.stack(loss))

        total_loss += loss.item()
        count += 1.
    
    total_loss_pred = [_/count for _ in total_loss_pred]
    for l_idx, l in enumerate(total_loss_pred):
        logger(f"val_pred_loss/{ch(l_idx)}", l, global_step)
    total_loss = total_loss / count

    X = torch.cat(X, dim=0).detach()
    feats = [torch.cat(f, dim=0) for f in feats]
    outs = [torch.cat(o, dim=0) for o in outs]
    preds = [torch.argmax(o, dim=1) for o in outs]
    labels = torch.cat(labels, dim=0).detach().type(feats[0].type())
    int_pos_all = torch.cat(int_pos_all, dim=0).detach()

    # Save all features
    if epoch == args.epochs-1 or args.eval_ckpt is not None:
        save_tensor(X, log_folder, "inputs.pt")
        for f_idx, f in enumerate(feats):
            save_tensor(f, log_folder, f"feat{ch(f_idx)}.pt")
        save_tensor(labels, log_folder, "labels.pt")
        for o_idx, o in enumerate(outs):
            save_tensor(o, log_folder, f"out{ch(o_idx)}.pt")
        save_tensor(int_pos_all, log_folder, "int_pos_all.pt")
    
    mesh_size = 100
    x_min, x_max = -2, 2
    fig, axs = plot_2d_decision_boundary(model, device, x_min, x_max, mesh_size=mesh_size)
    X_ = X[int_pos_all[:, 1] == 1].cpu()
    labels_ = labels[int_pos_all[:, 1] == 1].cpu()
    rand_idx = torch.randperm(X_.shape[0])[:200]
    X_ = X_[rand_idx]
    labels_ = labels_[rand_idx]
    axs[0].scatter(X_[:, 0], X_[:, 1], c=labels_[:, 0], marker='o', s=50, alpha=1,
            cmap="RdBu", edgecolors='white', linewidths=0.1)
    axs[1].scatter(X_[:, 0], X_[:, 1], c=labels_[:, 1], marker='o', s=50, alpha=1,
            cmap="RdBu", edgecolors='white', linewidths=0.1)

    fig.savefig(os.path.join(log_folder, "decision_boundary.png"))
    plt.close()

    # Calculate all the validation metrics
    calc_all_metrics(labels, feats, preds,
                     int_pos_all, eval_dep_fn,
                     logger, global_step)

    logger("val_pred_loss/total", total_loss, global_step)

    logger.write()

    return
