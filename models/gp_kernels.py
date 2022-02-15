import torch


def rbf_kernel(X1, X2, theta):
    """
    RBF kernel

    Args:
        X1: Array of m points (n x d).
        X2: Array of n points (m x d).
        theta: Kernel parameters

    Returns:
        (n x m) matrix
    """

    theta0, theta1 = theta

    Dist = (
        torch.sum(X1**2, 1).reshape(-1, 1)
        + torch.sum(X2**2, 1)
        - 2 * torch.mm(X1, X2.T)
    )

    K = theta0 * torch.exp(-(1.0 / theta1) * Dist)
    return K
