from pdb import set_trace

import numpy as np
import torch


class GPBase:
    """Base class for Gaussian Process"""

    def __init__(self, kernel_fn, kernel_params):
        """
        Initializer of GPBase

        Args:
            kernel_fn: A kernel function whose args are two matrices
            X (n x d) and Y (m x d)

            kernel_params: Parameters of kernel_fn which will be optimized

        Returns:
            An instance of this class
        """

        # for GP
        self.kernel_params = kernel_params
        self.kernel_fn = kernel_fn
        self.kernel_mx = lambda X1, X2: kernel_fn(X1, X2, kernel_params)

        # for optimizer
        self.epoch_loss = np.array([])
        self.best_kernel_params = None
        self.best_records = None

    def early_stop(self, min_epoch=50, n_patience=20, records=None):
        """
        Early stopping algorithm when optimize self.kernel_params

        Args:
            min_epoch: Skip early stopping until min_epoch
            patience: Break epoch loop if the optimizer can't update min loss #patience times.

        Returns:
            {True, False}; Flag of early stop or not
        """
        # Save best parames
        if self.epoch_loss.min() == self.epoch_loss[-1]:
            self.best_kernel_params = [p.clone() for p in self.kernel_params]
            self.best_records = records

        # Check minimum epoch
        if len(self.epoch_loss) < min_epoch:
            return False

        # Check patience
        if np.all(self.epoch_loss.min() < self.epoch_loss[-n_patience:]):
            return True

        return False

    def rewind_kernel_params(self, kernel_params):
        """
        Rewind kernel parameters

        Args:
            kernel_params: Kernel parameters

        Returns:
            Itself
        """
        self.kernel_params = kernel_params
        self.kernel_mx = lambda X1, X2: self.kernel_fn(X1, X2, kernel_params)

        return self

    def _K(self, X, eps=1.0e-5):
        """
        Create a kernel matrix by X

        Args:
            X: Data points (n, m) matrix.

        Returns:
            Kernel matrix (n, n) matrix.
        """

        return self.kernel_mx(X, X) + eps * torch.eye(len(X))
