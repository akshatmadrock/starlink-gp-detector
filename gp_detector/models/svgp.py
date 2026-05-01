"""
Sparse Variational Gaussian Process (SVGP) binary classifier.

Architecture
------------
Kernel:  k_geom(elevation) + k_spec(z, harmonics, freq_dev) + k_time(hour, day)
         Each block is a ScaleKernel(RBFKernel) acting on its feature subset.
         The additive structure keeps geometric and spectral evidence separate
         and makes kernel lengthscales interpretable.

Likelihood:   Bernoulli (binary classification)
Inference:    Variational ELBO (CholeskyVariationalDistribution)
Inducing pts: k-means initialised, learnable during training

Feature index map (must match gp_detector.dataset.FEATURE_COLS):
  0  z_score
  1  log_z
  2  harmonics
  3  freq_deviation
  4  sat_elevation      <- geometric / orbital prior
  5  hour_sin
  6  hour_cos
  7  campaign_day
"""

import numpy as np
import torch
import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.likelihoods import BernoulliLikelihood
from gpytorch.mlls import VariationalELBO
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler


GEOM_DIMS = [4]
SPEC_DIMS = [0, 1, 2, 3]
TIME_DIMS = [5, 6, 7]


class StarLinkSVGP(ApproximateGP):
    """Physics-informed SVGP with additive kernel."""

    def __init__(self, inducing_points: torch.Tensor):
        var_dist     = CholeskyVariationalDistribution(inducing_points.size(0))
        var_strategy = VariationalStrategy(
            self, inducing_points, var_dist, learn_inducing_locations=True
        )
        super().__init__(var_strategy)

        self.mean_module  = gpytorch.means.ConstantMean()
        self.covar_module = (
            ScaleKernel(RBFKernel(active_dims=torch.tensor(GEOM_DIMS)))
            + ScaleKernel(RBFKernel(active_dims=torch.tensor(SPEC_DIMS)))
            + ScaleKernel(RBFKernel(active_dims=torch.tensor(TIME_DIMS)))
        )

    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x), self.covar_module(x)
        )


def _init_inducing(X: np.ndarray, n: int, seed: int) -> torch.Tensor:
    km = MiniBatchKMeans(n_clusters=n, random_state=seed, n_init=3)
    km.fit(X)
    return torch.tensor(km.cluster_centers_, dtype=torch.float32)


def train(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_inducing: int = 300,
    n_epochs: int = 80,
    batch_size: int = 512,
    lr: float = 0.05,
    seed: int = 42,
    verbose: bool = True,
) -> tuple:
    """
    Train the SVGP classifier.

    Returns (model, likelihood, scaler, loss_history).
    The scaler is fit inside training so it travels with the model.
    """
    torch.manual_seed(seed)

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_train).astype(np.float32)

    inducing = _init_inducing(X_scaled, n_inducing, seed)
    model      = StarLinkSVGP(inducing)
    likelihood = BernoulliLikelihood()
    model.train(); likelihood.train()

    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(likelihood.parameters()), lr=lr
    )
    mll = VariationalELBO(likelihood, model, num_data=len(X_train))

    dataset = torch.utils.data.TensorDataset(
        torch.tensor(X_scaled), torch.tensor(y_train)
    )
    loader  = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    losses = []
    for epoch in range(n_epochs):
        total = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = -mll(model(xb), yb)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(xb)
        avg = total / len(X_train)
        losses.append(avg)
        if verbose and (epoch % 10 == 0 or epoch == n_epochs - 1):
            print(f"  epoch {epoch:3d}/{n_epochs}  loss: {avg:.5f}")

    return model, likelihood, scaler, losses


@torch.no_grad()
def predict(
    model: StarLinkSVGP,
    likelihood: BernoulliLikelihood,
    scaler: StandardScaler,
    X: np.ndarray,
    batch_size: int = 2048,
) -> tuple:
    """
    Returns (mean_prob, std_prob) as numpy arrays.
    std_prob is sqrt(p*(1-p)) under the Bernoulli posterior predictive.
    """
    model.eval(); likelihood.eval()
    X_s = torch.tensor(scaler.transform(X).astype(np.float32))

    means, stds = [], []
    with gpytorch.settings.fast_pred_var():
        for i in range(0, len(X_s), batch_size):
            preds = likelihood(model(X_s[i : i + batch_size]))
            p     = preds.probs.numpy()
            means.append(p)
            stds.append(np.sqrt(p * (1 - p)))

    return np.concatenate(means), np.concatenate(stds)


def save(model, likelihood, scaler, path: str):
    import pickle, torch
    torch.save({
        "model_state":      model.state_dict(),
        "likelihood_state": likelihood.state_dict(),
        "inducing_points":  model.variational_strategy.inducing_points,
    }, path + ".pt")
    with open(path + "_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)


def load(path: str) -> tuple:
    import pickle, torch
    ckpt    = torch.load(path + ".pt", weights_only=False)
    model   = StarLinkSVGP(ckpt["inducing_points"])
    model.load_state_dict(ckpt["model_state"])
    lik     = BernoulliLikelihood()
    lik.load_state_dict(ckpt["likelihood_state"])
    with open(path + "_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, lik, scaler
