from pdb import set_trace

import numpy as np
import torch
import torch.optim as optim
from torch import trace as tr
from torch.linalg import slogdet
from torch.utils.data import DataLoader

from models.gp_base import GPBase


class DGPLVM(GPBase):
    """Discriminative GPLVM"""

    def __init__(self, kernel_fn, kernel_params, optimize_params):
        super().__init__(kernel_fn, kernel_params)
        self.optimize_params = optimize_params

    def fit(
        self,
        Y,
        t,
        n_dim,
        n_epoch,
        n_batch,
        lr=1.0e-3,
        use_early_stop=True,
        min_epoch=100,
        n_patience=10,
        init_latent_points=None,
        sigma_d=0.01,
    ):
        """
        Optimize latent space X and kernel parameters which specified as optimize_params
        to minimize negative log likelihood.

        Args:
            Y : m input points, (m, d) matrix
            t : labels of Y, (m,) vector , each value is in {0, 1}.
            n_dim : Dimension of latents points X, (m, k) matrix, k << d
            use_early_stop : Flag of early stoping.
                After <min_epoch>,
                if the minimum loss cannot be updated <n_patience> times consectively,
                the training loop is stopped.
            init_latent_points: Initial latent points X, (m, k) matrix,
                If None, initial latent points will be decided by SVD.
            sigma_d : Controlls discrimination of latent space.
                see: [Utram et al, 2007]
        Returns
            X : Latent points , (m, k) matrix
        """
        # Initialize X
        if init_latent_points is None:
            U, S, V = torch.linalg.svd(Y)
            X = U[:, :n_dim].clone().detach().requires_grad_(True)
        else:
            X = init_latent_points

        self.Y = Y
        self.t = t
        self.X0 = X.clone()

        # Set optimizer
        # optimizer = optim.SGD([X, *self.optimize_params], lr=lr)
        optimizer = optim.Adam([X, *self.optimize_params], lr=lr)

        # Training loop
        self.epoch_loss = np.array([])
        for i in range(n_epoch):
            # print(i)
            dataloader = DataLoader(
                np.arange(0, len(X)), batch_size=n_batch, shuffle=True
            )

            # set_trace()
            iter_loss = 0
            for j, ix in enumerate(dataloader):
                nll = self.dneg_loglik(X[ix], Y[ix], t[ix], sigma_d)

                iter_loss += nll.item()

                optimizer.zero_grad()
                nll.backward()
                optimizer.step()

            self.epoch_loss = np.append(
                self.epoch_loss, [iter_loss / len(dataloader.dataset)]
            )

            if use_early_stop and self.early_stop(
                min_epoch, n_patience, X.detach().clone()
            ):
                print(f"Early stopping at epoch={i}")
                break

        if self.best_kernel_params is not None:
            self.rewind_kernel_params(self.best_kernel_params)
            return self.best_records  # recorded when early_stpping

        return X.detach().clone()

    # Discriminative Negative Log-Likelihood
    def dneg_loglik(self, X, Y, t, sigma_d=0.01):
        _, D = X.shape

        Kx = self._K(X)
        sign, logdet = slogdet(Kx)
        KxInv_YYt = torch.linalg.solve(Kx, torch.mm(Y, Y.T))

        # set_trace()
        Sb = self.var_between_groups(X, t)
        Sw = self.var_within_groups(X, t)
        SbInvSw = torch.linalg.solve(Sb, Sw)

        X_norm = torch.sum(torch.norm(X, dim=1))

        nll = (
            (D / 2) * (sign * logdet)
            + (1 / 2) * tr(KxInv_YYt)
            + (1 / sigma_d) * tr(SbInvSw)
            + (1 / 2) * X_norm
        )

        return nll

    # Variances within groups
    def var_within_groups(self, X, t):
        N, d = X.shape

        m0 = torch.mean(X, axis=0).reshape(d, -1)
        Sw = torch.zeros((d, d))
        for i in torch.unique(t):
            Xi = X[t == i]
            Ni, _ = Xi.shape
            mi = torch.mean(Xi, axis=0).reshape(d, -1)

            Sw = Sw + (Ni / N) * torch.mm((mi - m0), (mi - m0).T)
        return Sw

    # Variances between groups
    def var_between_groups(self, X, t):
        N, d = X.shape

        Sb = torch.zeros((d, d))
        for i in torch.unique(t):
            Xi = X[t == i]
            Ni, _ = Xi.shape
            mi = torch.mean(Xi, axis=0).reshape(d, -1)

            sb = torch.zeros((d, d))
            for xi in Xi:
                xi = xi.reshape(d, -1)

                sb = sb + torch.mm((xi - mi), (xi - mi).T)

            Sb = Sb + (Ni / N) * ((1 / Ni) * sb)

        return Sb
