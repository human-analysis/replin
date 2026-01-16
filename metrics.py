import torch

from main_utils import ch


def calc_all_metrics(labels, feats, preds,
                     int_pos_all, eval_dep_fn,
                     logger, global_step, desc="val"):
    obs_mask = (int_pos_all[:, 0] == 0) & (int_pos_all[:, 1] == 0)
    obs_perf_list = []
    if torch.sum(obs_mask.type(torch.int)) > 0:
        obs_labels = [labels[obs_mask, l] for l in range(labels.shape[1])]
        obs_feats = [f[obs_mask] for f in feats]
        obs_preds = [p[obs_mask] for p in preds]

        # Calculate and log the accuracy for each variable separately
        for idx, (p, y) in enumerate(zip(obs_preds, obs_labels)):
            obs_perf = torch.mean((p == y).type(torch.float))
            logger(f"{desc}_perf_obs/{ch(idx)}", obs_perf, global_step)
            obs_perf_list.append(obs_perf)

        # Calculate and log the ROIs and HSICs between the features and the
        # labels of each variable separately
        for idx, (f, y) in enumerate(zip(obs_feats, obs_labels)):
            obs_iFI_dep = eval_dep_fn(y, f)
            logger(f"{desc}_dep_obs/{ch(idx)}F{ch(idx)}", obs_iFI_dep, global_step)

        # Calculate and log the ROIs and HSICs between every pair of
        # features. It is important to do this for observational data for
        # cases where some variables are always independent.
        for i1 in range(len(obs_feats)):
            f1 = obs_feats[i1]
            for i2 in range(i1+1, len(obs_feats)):
                f2 = obs_feats[i2]
                obs_FiFj_dep = eval_dep_fn(f1, f2)
                logger(f"{desc}_dep_obs/F{ch(i1)}F{ch(i2)}", obs_FiFj_dep, global_step)

    int_vars = torch.nonzero(torch.sum(int_pos_all, 0)).reshape(-1)

    for v_idx, v in enumerate(int_vars):
        int_mask = int_pos_all[:, v] == 1
        int_labels = [labels[int_mask, l] for l in range(labels.shape[1])]
        int_feats = [f[int_mask] for f in feats]
        int_preds = [p[int_mask] for p in preds]

        # Calculate and log the accuracy for each variable separately
        for idx, (p, y) in enumerate(zip(int_preds, int_labels)):
            int_perf = torch.mean((p == y).type(torch.float))
            logger(f"{desc}_perf_int{ch(v)}/{ch(idx)}", int_perf, global_step)
            
            # Calculate the relative drop in accuracy between
            # interventional and observational data.
            if len(obs_perf_list) > 0:
                dropacc = (obs_perf_list[idx] - int_perf)/obs_perf_list[idx]
                logger(f"{desc}_perf_int{ch(v)}/rel_drop_{ch(idx)}", dropacc, global_step)

        # Calculate and log the ROIs and HSICs between the features and the
        # labels of each variable separately
        for idx, (f, y) in enumerate(zip(int_feats, int_labels)):
            int_iFI_dep = eval_dep_fn(y, f)
            logger(f"{desc}_dep_int{ch(v)}/{ch(idx)}F{ch(idx)}", int_iFI_dep, global_step)

        # Calculate and log the ROIs and HSICs between every pair of
        # features. It is important to do this for observational data for
        # cases where some variables are always independent.
        for i1 in range(len(int_feats)):
            f1 = int_feats[i1]
            for i2 in range(i1+1, len(int_feats)):
                f2 = int_feats[i2]
                int_FiFj_dep = eval_dep_fn(f1, f2)
                logger(f"{desc}_dep_int{ch(v)}/F{ch(i1)}F{ch(i2)}", int_FiFj_dep, global_step)

    logger.write()

    return
