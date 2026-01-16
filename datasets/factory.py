import torch

# generate labels for a simple 2-variable graphs.
def labels_from_2var_graph(a, b, num_points, device):
    if a is not None and b is not None:
        assert a.shape[0] == b.shape[0]

    # Override num_points if a or b is provided
    if a is not None:
        num_points = a.shape[0]
    elif b is not None:
        num_points = b.shape[0]

    if a is None:
        # sample A randomly
        a = torch.bernoulli(torch.ones(num_points, device=device) * 0.6)
    if b is None:
        # B follows A unless provided
        b = a

    return (a, b)
