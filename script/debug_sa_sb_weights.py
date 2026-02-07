"""
Debug S_A, S_B at different t to understand the geometry.
Current formula: score_i = -||xt - t*x_i||^2 / (2*(1-t)^2), S_i = exp(score_i), w_i = S_i/(S_A+S_B)
x1* = xt + u_A*(1-t)  [one Euler step from xt toward frozen endpoint]
"""
import sys
from pathlib import Path
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import torch
import numpy as np
from model import create_model, load_model_config
from torchcfm.utils import sample_right_moon, sample_8gaussians

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ckpt_1moon = torch.load("models/cfm_8g_to_1moon/checkpoint_iter_20000.pt", map_location=device, weights_only=False)
cfg = load_model_config(str(_root / "config" / "model_config.yaml"), "8g_to_1moon")
model_1moon = create_model("8g_to_1moon", cfg, device)
model_1moon.load_state_dict(ckpt_1moon["model_state_dict"])
model_1moon.eval()

torch.manual_seed(42)
x0 = sample_8gaussians(1).to(device).squeeze(0)
x1 = sample_right_moon(1).to(device).squeeze(0)  # right moon
eps = 1e-6

print("x0:", x0.cpu().numpy())
print("x1 (right moon):", x1.cpu().numpy())
print()

for t_val in [0.0, 0.1, 0.5, 0.7, 0.9]:
    t = float(t_val)
    xt = (1 - t) * x0 + t * x1
    one_minus_t = max(1 - t, eps)

    with torch.no_grad():
        xt_t = torch.cat([xt.unsqueeze(0), torch.full((1,), t, device=device)[:, None]], dim=-1)
        u_A = model_1moon(xt_t).squeeze(0)
        x1_star = xt + u_A * one_minus_t  # one step toward "old" endpoint

    # Current formula
    diff_A = xt - t * x1_star
    diff_B = xt - t * x1
    dist_A = (diff_A ** 2).sum().sqrt().item()
    dist_B = (diff_B ** 2).sum().sqrt().item()

    score_A = -(diff_A.pow(2).sum().item()) / (2 * one_minus_t ** 2)
    score_B = -(diff_B.pow(2).sum().item()) / (2 * one_minus_t ** 2)
    S_A = np.exp(score_A)
    S_B = np.exp(score_B)
    w_A = S_A / (S_A + S_B)
    w_B = S_B / (S_A + S_B)

    print(f"t = {t}:")
    print(f"  xt: {xt.cpu().numpy()}")
    print(f"  x1* (xt + u_A*(1-t)): {x1_star.cpu().numpy()}")
    print(f"  ||xt - t*x1*|| = {dist_A:.4f},  ||xt - t*x1|| = {dist_B:.4f}")
    print(f"  score_A = {score_A:.4f}, score_B = {score_B:.4f}")
    print(f"  S_A = {S_A:.6f}, S_B = {S_B:.6f}  =>  w_A = {w_A:.4f}, w_B = {w_B:.4f}")
    print()

print("--- Key insight ---")
print("x1* = xt + u_A*(1-t), so when t->1, x1* -> xt (one small step).")
print("Then t*x1* ~ t*xt, so xt - t*x1* = (1-t)*xt, and score_A = -||xt||^2/2 (no (1-t)^2!).")
print("Similarly xt - t*x1 = (1-t)*x0, so score_B = -||x0||^2/2.")
print("So at t~1, we compare -||xt||^2/2 vs -||x0||^2/2; S_A and S_B can be comparable.")
print("The formula does NOT directly measure 'distance to old left moon endpoint'.")
