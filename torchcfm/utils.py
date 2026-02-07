import math

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchdyn.datasets import generate_moons

# Which moon class is left vs right (by mean x: left < right). Set once using sklearn convention.
_LEFT_MOON_CLASS = 0
_RIGHT_MOON_CLASS = 1


def eight_normal_sample(n, dim, scale=1, var=1):
    m = torch.distributions.multivariate_normal.MultivariateNormal(
        torch.zeros(dim), math.sqrt(var) * torch.eye(dim)
    )
    centers = [
        (1, 0),
        (-1, 0),
        (0, 1),
        (0, -1),
        (1.0 / np.sqrt(2), 1.0 / np.sqrt(2)),
        (1.0 / np.sqrt(2), -1.0 / np.sqrt(2)),
        (-1.0 / np.sqrt(2), 1.0 / np.sqrt(2)),
        (-1.0 / np.sqrt(2), -1.0 / np.sqrt(2)),
    ]
    centers = torch.tensor(centers) * scale
    noise = m.sample((n,))
    multi = torch.multinomial(torch.ones(8), n, replacement=True)
    data = []
    for i in range(n):
        data.append(centers[multi[i]] + noise[i])
    data = torch.stack(data)
    return data


# Moon noise: 2-moon and 1-moon use same noise=0.05 (tight, less dispersed)
MOON_NOISE = 0.05


def sample_moons(n):
    """Sample 2 moons (both arcs). Same noise=0.05 as sample_1moon for consistency."""
    try:
        from sklearn.datasets import make_moons
        X, _ = make_moons(n_samples=n, noise=MOON_NOISE, random_state=None)
        out = torch.from_numpy(X.astype(np.float32)) * 3 - 1
        return out.float()
    except ImportError:
        x0, _ = generate_moons(n, noise=MOON_NOISE)
        return (x0 * 3 - 1).float()


def sample_1moon(n, moon_class=0, noise=None):
    """Sample only one of the two moons (full arc). moon_class: 0=left, 1=right. noise default 0.05."""
    if noise is None:
        noise = MOON_NOISE
    try:
        from sklearn.datasets import make_moons
    except ImportError:
        need = max(n * 3, 1000)
        x, y = generate_moons(need, noise=noise)
        y = y.numpy() if isinstance(y, torch.Tensor) else np.asarray(y)
        mask = (y == moon_class)
        x_one = x[mask].cpu().numpy() if isinstance(x, torch.Tensor) else x[mask]
        x_one = x_one[:n]
        out = torch.from_numpy(np.asarray(x_one, dtype=np.float32)) * 3 - 1
        return out.float()
    X, y = make_moons(n_samples=max(2 * n, 2000), noise=noise, random_state=None)
    y = np.asarray(y).ravel()
    mask = (y == moon_class)
    X_one = X[mask][:n]
    out = torch.from_numpy(X_one.astype(np.float32)) * 3 - 1
    return out.float()


def sample_left_moon(n):
    """Sample the left moon only (full arc). noise=0.05."""
    return sample_1moon(n, moon_class=_LEFT_MOON_CLASS)


def sample_right_moon(n):
    """Sample the right moon only (full arc). noise=0.05. For continue learning."""
    return sample_1moon(n, moon_class=_RIGHT_MOON_CLASS)


def sample_8gaussians(n):
    return eight_normal_sample(n, 2, scale=5, var=0.1).float()


def seven_normal_sample(n, dim, scale=1, var=1):
    """Same as 8 gaussians but with the 8th center removed (index 7: (-1/sqrt2, -1/sqrt2))."""
    m = torch.distributions.multivariate_normal.MultivariateNormal(
        torch.zeros(dim), math.sqrt(var) * torch.eye(dim)
    )
    centers = [
        (1, 0),
        (-1, 0),
        (0, 1),
        (0, -1),
        (1.0 / np.sqrt(2), 1.0 / np.sqrt(2)),
        (1.0 / np.sqrt(2), -1.0 / np.sqrt(2)),
        (-1.0 / np.sqrt(2), 1.0 / np.sqrt(2)),
        # (-1.0 / np.sqrt(2), -1.0 / np.sqrt(2)),  # omitted
    ]
    centers = torch.tensor(centers) * scale
    noise = m.sample((n,))
    multi = torch.multinomial(torch.ones(7), n, replacement=True)
    data = []
    for i in range(n):
        data.append(centers[multi[i]] + noise[i])
    data = torch.stack(data)
    return data


def sample_7gaussians(n):
    return seven_normal_sample(n, 2, scale=5, var=0.1).float()


class torch_wrapper(torch.nn.Module):
    """Wraps model to torchdyn compatible format."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, t, x, *args, **kwargs):
        return self.model(torch.cat([x, t.repeat(x.shape[0])[:, None]], 1))


def plot_trajectories(traj):
    """Plot trajectories of some selected samples."""
    n = 2000
    plt.figure(figsize=(6, 6))
    plt.scatter(traj[0, :n, 0], traj[0, :n, 1], s=10, alpha=0.8, c="black")
    plt.scatter(traj[:, :n, 0], traj[:, :n, 1], s=0.2, alpha=0.2, c="olive")
    plt.scatter(traj[-1, :n, 0], traj[-1, :n, 1], s=4, alpha=1, c="blue")
    plt.legend(["Prior sample z(S)", "Flow", "z(0)"])
    plt.xticks([])
    plt.yticks([])
    plt.show()
