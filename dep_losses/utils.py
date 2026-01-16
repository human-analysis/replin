import torch

def calc_score_once(X, Y, score):
    return score(X, Y)

def get_threshold(X, Y, score, K):
    # Calculate the score for K permuations.
    s_list = []
    for i in range(K):
        perm = torch.randperm(Y.shape[0])
        Y_ = Y[perm] # Permutate Y
        s = calc_score_once(X, Y_, score)
        # Y_ = Y[perm][perm != torch.arange(Y.shape[0])] # Permutate Y
        # X_ = X[perm != torch.arange(Y.shape[0])]
        # s = calc_score_once(X_, Y_, score)
        s_list.append(s)

    s = torch.tensor(s_list)
    s_mean = torch.mean(s)
    s_std = torch.std(s)

    return s_mean, s_std

def mean_center(features, dim):
    """
    Return mean-centered features along a given dimension.
    """
    return features - torch.mean(features, dim=dim)
