"""Multinomial logistic regression on GPU (full-batch L-BFGS), shared by E1 and the LP-FT init in E2."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def fit_logreg(X: torch.Tensor, y: torch.Tensor, n_cls: int, wd: float, iters: int = 300) -> torch.Tensor:
    """Returns W of shape (D + 1, C); the last row is the bias. L2 on the weights only."""
    W = torch.zeros(X.shape[1] + 1, n_cls, device=X.device, requires_grad=True)
    Xb = torch.cat([X, torch.ones(len(X), 1, device=X.device)], 1)
    opt = torch.optim.LBFGS([W], lr=1, max_iter=iters, history_size=20, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(Xb @ W, y) + wd * (W[:-1] ** 2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return W.detach()


def predict(W: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
    return torch.softmax(torch.cat([X, torch.ones(len(X), 1, device=X.device)], 1) @ W, 1)


def fold_standardization(W: torch.Tensor, mu: torch.Tensor, sd: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Turn a classifier trained on (x - mu) / sd into nn.Linear weight and bias acting on raw x."""
    Wf = W[:-1] / sd[:, None]
    bias = W[-1] - (mu / sd) @ W[:-1]
    return Wf.T.contiguous(), bias
