"""
Compare: (A) 8g->2moons from scratch vs (B) 8g->1moon then continue to 2moons.
Left: true 1 moon + 2 moons overlaid (2moons 多一个月牙).
Middle: 2moons from scratch generated. Right: 1moon then continue generated.
Wasserstein to true 2 moons + visualization.
"""
import argparse
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))
import subprocess
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchdyn.core import NeuralODE

from model import create_model, load_model_config
from torchcfm.utils import sample_moons, sample_left_moon, sample_8gaussians


def load_model_from_checkpoint(checkpoint_path, device, dataset_key="8g_to_2moons"):
    root = Path(__file__).resolve().parent.parent
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_config = checkpoint.get("model_config") or load_model_config(
        str(root / "config" / "model_config.yaml"), dataset_key
    )
    model = create_model(dataset_key, model_config, device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, checkpoint.get("args", {})


def _is_continue_2moons_checkpoint(ckpt_path):
    """Check if this is from train_continue_2moons.py (1moon->2moons), not raw 8g_to_1moon."""
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        args = ckpt.get("args", {})
        if args.get("continue_from"):
            return True
        if args.get("dataset") == "8g_to_2moons":
            return True
        if args.get("dataset") == "8g_to_1moon" and "continue_from" not in args:
            return False
        return None
    except Exception:
        return None


def generate_samples_2d_from_8g(model, num_samples, device, num_steps=50, seed=42):
    """Generate: start from 8 gaussians, integrate to t=1 (get 2moons or 1moon depending on model)."""
    model.eval()
    torch.manual_seed(seed)
    np.random.seed(seed)
    x0 = sample_8gaussians(num_samples).to(device)

    def model_wrapper(t, x, **kwargs):
        if t.dim() == 0:
            t = t.expand(x.shape[0])
        return model(torch.cat([x, t[:, None]], 1))

    node = NeuralODE(model_wrapper, solver="euler", sensitivity="adjoint", atol=1e-4, rtol=1e-4)
    with torch.no_grad():
        t_span = torch.linspace(0, 1, num_steps + 1).to(device)
        traj = node.trajectory(x0, t_span=t_span)
    samples = traj[-1].cpu().numpy()
    model.train()
    return samples


def wasserstein_2d_approx(X, Y):
    from scipy.stats import wasserstein_distance
    w0 = wasserstein_distance(X[:, 0], Y[:, 0])
    w1 = wasserstein_distance(X[:, 1], Y[:, 1])
    w_combined = np.sqrt(w0**2 + w1**2)
    return w0, w1, w_combined


def _latest_ckpt(dir_path):
    if not dir_path.exists():
        return None
    ckpts = list(dir_path.glob("checkpoint_iter_*.pt"))
    if not ckpts:
        return None
    return max(ckpts, key=lambda p: int(p.stem.split("_")[-1].replace(".pt", "")))


def run_training_if_needed(args):
    root = Path(__file__).resolve().parent.parent
    save_dir = Path(args.save_dir)
    method = args.method
    base_2moons = save_dir / f"{method}_8g_to_2moons"
    continue_dir = save_dir / f"{method}_8g_to_1moon_continue_2moons"
    continue_prior_dir = save_dir / f"{method}_8g_to_1moon_continue_2moons_prior"
    continue_efm_dir = save_dir / f"{method}_8g_to_1moon_continue_2moons_efm"
    base_1moon = save_dir / f"{method}_8g_to_1moon"

    if not args.train:
        ckpt_2m = Path(args.checkpoint_2moons) if args.checkpoint_2moons else _latest_ckpt(base_2moons)
        ckpt_1m = Path(args.checkpoint_1moon) if args.checkpoint_1moon else _latest_ckpt(base_1moon)
        ckpt_cont = Path(args.checkpoint_continue) if args.checkpoint_continue else _latest_ckpt(continue_dir)
        args._default_ckpt_prior = _latest_ckpt(continue_prior_dir)
        args._default_ckpt_efm = _latest_ckpt(continue_efm_dir)
        return ckpt_2m, ckpt_1m, ckpt_cont

    if args.iterations_2moons and (not (base_2moons / f"checkpoint_iter_{args.iterations_2moons}.pt").exists() or args.force_train):
        print(f"Training 8g_to_2moons from scratch for {args.iterations_2moons} iters...")
        subprocess.run([
            sys.executable, "train.py",
            "--dataset", "8g_to_2moons", "--method", method,
            "--iterations", str(args.iterations_2moons),
            "--save_iter", str(max(1, args.iterations_2moons // 2)),
            "--log_iter", str(max(1, args.iterations_2moons // 5)),
            "--save_dir", str(save_dir),
        ], check=True, cwd=root)
    ckpt_2moons = _latest_ckpt(base_2moons)

    if args.iterations_1moon and (not (base_1moon / f"checkpoint_iter_{args.iterations_1moon}.pt").exists() or args.force_train):
        print(f"Training 8g_to_1moon for {args.iterations_1moon} iters...")
        subprocess.run([
            sys.executable, "train.py",
            "--dataset", "8g_to_1moon", "--method", method,
            "--iterations", str(args.iterations_1moon),
            "--save_iter", str(max(1, args.iterations_1moon // 2)),
            "--log_iter", str(max(1, args.iterations_1moon // 5)),
            "--save_dir", str(save_dir),
        ], check=True, cwd=root)
    ckpt_1moon = _latest_ckpt(base_1moon)

    if ckpt_1moon and (not _latest_ckpt(continue_dir) or args.force_train):
        print(f"Continue training 1moon->2moons for {args.iterations_continue} iters...")
        subprocess.run([
            sys.executable, "train_continue_2moons.py",
            "--checkpoint", str(ckpt_1moon),
            "--iterations", str(args.iterations_continue),
            "--save_iter", str(max(1, args.iterations_continue)),
            "--save_dir", str(save_dir),
        ], check=True, cwd=root)
    ckpt_continue = _latest_ckpt(continue_dir)

    if args.train_prior and ckpt_1moon and (not _latest_ckpt(continue_prior_dir) or args.force_train):
        print(f"Continue training 1moon->2moons with prior-preservation for {args.iterations_continue} iters...")
        subprocess.run([
            sys.executable, "train_continue_2moons.py",
            "--checkpoint", str(ckpt_1moon),
            "--loss", "prior",
            "--iterations", str(args.iterations_continue),
            "--save_iter", str(max(1, args.iterations_continue)),
            "--save_dir", str(save_dir),
        ], check=True, cwd=root)
    if args.train_efm and ckpt_1moon and (not _latest_ckpt(continue_efm_dir) or args.force_train):
        print(f"Continue training 1moon->2moons with EFM-composite for {args.iterations_continue} iters...")
        subprocess.run([
            sys.executable, "train_continue_2moons.py",
            "--checkpoint", str(ckpt_1moon),
            "--loss", "efm",
            "--iterations", str(args.iterations_continue),
            "--save_iter", str(max(1, args.iterations_continue)),
            "--save_dir", str(save_dir),
        ], check=True, cwd=root)
    args._default_ckpt_prior = _latest_ckpt(continue_prior_dir)
    args._default_ckpt_efm = _latest_ckpt(continue_efm_dir)

    return ckpt_2moons, ckpt_1moon, ckpt_continue


def main():
    parser = argparse.ArgumentParser(description="Compare 8g->2moons from scratch vs 8g->1moon then continue to 2moons")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--train_prior", action="store_true", help="When --train, also train prior-preservation continue")
    parser.add_argument("--train_efm", action="store_true", help="When --train, also train EFM-composite continue")
    parser.add_argument("--force_train", action="store_true")
    parser.add_argument("--save_dir", type=str, default="./models")
    parser.add_argument("--method", type=str, default="cfm")
    parser.add_argument("--iterations_2moons", type=int, default=100)
    parser.add_argument("--iterations_1moon", type=int, default=100)
    parser.add_argument("--iterations_continue", type=int, default=100)
    parser.add_argument("--checkpoint_2moons", type=str,
                        default="models/cfm_8g_to_2moons/checkpoint_iter_20000.pt")
    parser.add_argument("--checkpoint_1moon", type=str,
                        default="models/cfm_8g_to_1moon/checkpoint_iter_20000.pt")
    parser.add_argument("--checkpoint_continue", type=str,
                        default="models/cfm_8g_to_1moon_continue_2moons/checkpoint_iter_5000.pt")
    parser.add_argument("--checkpoint_continue_distill", type=str, default=None,
                        help="Optional: distill continue checkpoint to add in overlay")
    parser.add_argument("--checkpoint_continue_prior", type=str, default=None,
                        help="Optional: prior-preservation continue checkpoint to add in overlay")
    parser.add_argument("--checkpoint_continue_efm", type=str, default=None,
                        help="Optional: EFM-composite continue checkpoint to add in overlay")
    parser.add_argument("--num_samples", type=int, default=2000)
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="./exp/compare_2moons_vs_1moon_continue")
    parser.add_argument("--output_name", type=str, default="compare_2moons_vs_1moon_continue.png")
    args = parser.parse_args()

    ckpt_2moons, ckpt_1moon, ckpt_continue = run_training_if_needed(args)

    if not ckpt_2moons or not ckpt_2moons.exists():
        raise FileNotFoundError(f"2moons checkpoint not found: {ckpt_2moons}. Use --train or --checkpoint_2moons")
    if not ckpt_1moon or not ckpt_1moon.exists():
        raise FileNotFoundError(f"1moon checkpoint not found: {ckpt_1moon}. Use --train or --checkpoint_1moon")
    if not ckpt_continue or not ckpt_continue.exists():
        raise FileNotFoundError(f"Continue checkpoint not found: {ckpt_continue}. Use --train or --checkpoint_continue")

    is_cont = _is_continue_2moons_checkpoint(ckpt_continue)
    if is_cont is False:
        raise ValueError(
            "你传的 --checkpoint_continue 是「只训 8g_to_1moon」的模型，不是「1moon 再继续训 2moons」的。\n"
            "请用 train_continue_2moons.py 生成的 checkpoint，路径应类似：\n"
            "  models/cfm_8g_to_1moon_continue_2moons/checkpoint_iter_XXX.pt"
        )
    elif is_cont is True:
        print("OK: checkpoint_continue 是 1moon->2moons 继续训练后的模型。")

    ckpt_distill = Path(args.checkpoint_continue_distill) if args.checkpoint_continue_distill else None
    if ckpt_distill is not None and (not ckpt_distill.exists()):
        ckpt_distill = None
    ckpt_prior = Path(args.checkpoint_continue_prior) if args.checkpoint_continue_prior else getattr(args, "_default_ckpt_prior", None)
    if ckpt_prior is not None and (not ckpt_prior.exists()):
        ckpt_prior = None
    ckpt_efm = Path(args.checkpoint_continue_efm) if args.checkpoint_continue_efm else getattr(args, "_default_ckpt_efm", None)
    if ckpt_efm is not None and (not ckpt_efm.exists()):
        ckpt_efm = None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_2moons, _ = load_model_from_checkpoint(ckpt_2moons, device)
    model_1moon, _ = load_model_from_checkpoint(ckpt_1moon, device, dataset_key="8g_to_1moon")
    model_continue, _ = load_model_from_checkpoint(ckpt_continue, device)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    true_2moons = sample_moons(args.num_samples).numpy()
    true_1moon = sample_left_moon(args.num_samples).numpy()
    gen_2moons = generate_samples_2d_from_8g(model_2moons, args.num_samples, device, args.num_steps, args.seed)
    gen_1moon = generate_samples_2d_from_8g(model_1moon, args.num_samples, device, args.num_steps, args.seed)
    gen_continue = generate_samples_2d_from_8g(model_continue, args.num_samples, device, args.num_steps, args.seed)

    gen_distill = None
    w_distill = None
    if ckpt_distill is not None:
        model_distill, _ = load_model_from_checkpoint(ckpt_distill, device)
        gen_distill = generate_samples_2d_from_8g(model_distill, args.num_samples, device, args.num_steps, args.seed)
        w0_d, w1_d, w_distill = wasserstein_2d_approx(gen_distill, true_2moons)

    gen_prior = None
    w_prior = None
    if ckpt_prior is not None:
        model_prior, _ = load_model_from_checkpoint(ckpt_prior, device)
        gen_prior = generate_samples_2d_from_8g(model_prior, args.num_samples, device, args.num_steps, args.seed)
        w0_p, w1_p, w_prior = wasserstein_2d_approx(gen_prior, true_2moons)

    gen_efm = None
    w_efm = None
    if ckpt_efm is not None:
        model_efm, _ = load_model_from_checkpoint(ckpt_efm, device)
        gen_efm = generate_samples_2d_from_8g(model_efm, args.num_samples, device, args.num_steps, args.seed)
        w0_e, w1_e, w_efm = wasserstein_2d_approx(gen_efm, true_2moons)

    w0_2m, w1_2m, w_2m = wasserstein_2d_approx(gen_2moons, true_2moons)
    w0_1m, w1_1m, w_1m = wasserstein_2d_approx(gen_1moon, true_1moon)
    w0_cont, w1_cont, w_cont = wasserstein_2d_approx(gen_continue, true_2moons)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = out_dir / "wasserstein_metrics.txt"
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write("Wasserstein distance\n\n")
        f.write("vs true 2 moons:\n")
        f.write(f"  2moons from scratch:  W={w_2m:.6f}\n")
        f.write(f"  1moon then continue:  W={w_cont:.6f}\n")
        if w_distill is not None:
            f.write(f"  1moon then continue (distill): W={w_distill:.6f}\n")
        if w_prior is not None:
            f.write(f"  1moon then continue (prior):   W={w_prior:.6f}\n")
        if w_efm is not None:
            f.write(f"  1moon then continue (efm):     W={w_efm:.6f}\n")
        f.write("vs true 1 moon:\n")
        f.write(f"  8g->1moon:            W={w_1m:.6f}\n\n")
        f.write(f"Checkpoints:\n  2moons: {ckpt_2moons}\n  1moon: {ckpt_1moon}\n  continue: {ckpt_continue}\n")
        if ckpt_distill is not None:
            f.write(f"  continue_distill: {ckpt_distill}\n")
        if ckpt_prior is not None:
            f.write(f"  continue_prior:   {ckpt_prior}\n")
        if ckpt_efm is not None:
            f.write(f"  continue_efm:     {ckpt_efm}\n")
    print(open(metrics_path, encoding="utf-8").read())

    # Figure 1: 4 + optional distill + optional prior + optional efm panels
    n_panels = 4 + (1 if gen_distill is not None else 0) + (1 if gen_prior is not None else 0) + (1 if gen_efm is not None else 0)
    fig, axes = plt.subplots(1, n_panels, figsize=(4.5 * n_panels, 5))
    # 1) True: 1 moon vs 2 moons overlaid
    axes[0].scatter(true_2moons[:, 0], true_2moons[:, 1], s=5, alpha=0.5, c="green", label="2 moons")
    axes[0].scatter(true_1moon[:, 0], true_1moon[:, 1], s=5, alpha=0.5, c="orangered", label="1 moon")
    axes[0].set_title("True: 1 moon vs 2 moons")
    axes[0].set_xlim(-6, 6)
    axes[0].set_ylim(-6, 6)
    axes[0].set_aspect("equal")
    axes[0].legend()

    # 2) 8g->2moons
    axes[1].scatter(gen_2moons[:, 0], gen_2moons[:, 1], s=5, alpha=0.6, c="blue", label="8g->2moons")
    axes[1].set_title(f"8g->2moons (W={w_2m:.4f})")
    axes[1].set_xlim(-6, 6)
    axes[1].set_ylim(-6, 6)
    axes[1].set_aspect("equal")
    axes[1].legend()

    # 3) 8g->1moon
    axes[2].scatter(gen_1moon[:, 0], gen_1moon[:, 1], s=5, alpha=0.6, c="purple", label="8g->1moon")
    axes[2].set_title(f"8g->1moon (W={w_1m:.4f})")
    axes[2].set_xlim(-6, 6)
    axes[2].set_ylim(-6, 6)
    axes[2].set_aspect("equal")
    axes[2].legend()

    # 4) 8g->1moon then continue
    axes[3].scatter(gen_continue[:, 0], gen_continue[:, 1], s=5, alpha=0.6, c="orange", label="continue (MSE)")
    axes[3].set_title(f"1moon then continue (W={w_cont:.4f})")
    axes[3].set_xlim(-6, 6)
    axes[3].set_ylim(-6, 6)
    axes[3].set_aspect("equal")
    axes[3].legend()

    idx = 4
    if gen_distill is not None:
        axes[idx].scatter(gen_distill[:, 0], gen_distill[:, 1], s=5, alpha=0.6, c="red", label="continue (distill)")
        axes[idx].set_title(f"1moon then continue distill (W={w_distill:.4f})")
        axes[idx].set_xlim(-6, 6)
        axes[idx].set_ylim(-6, 6)
        axes[idx].set_aspect("equal")
        axes[idx].legend()
        idx += 1
    if gen_prior is not None:
        axes[idx].scatter(gen_prior[:, 0], gen_prior[:, 1], s=5, alpha=0.6, c="cyan", label="continue (prior)")
        axes[idx].set_title(f"1moon then continue prior (W={w_prior:.4f})")
        axes[idx].set_xlim(-6, 6)
        axes[idx].set_ylim(-6, 6)
        axes[idx].set_aspect("equal")
        axes[idx].legend()
        idx += 1
    if gen_efm is not None:
        axes[idx].scatter(gen_efm[:, 0], gen_efm[:, 1], s=5, alpha=0.6, c="lime", label="continue (efm)")
        axes[idx].set_title(f"1moon then continue EFM (W={w_efm:.4f})")
        axes[idx].set_xlim(-6, 6)
        axes[idx].set_ylim(-6, 6)
        axes[idx].set_aspect("equal")
        axes[idx].legend()

    plt.tight_layout()
    out_path = out_dir / args.output_name
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Visualization saved to {out_path}")

    # Figure 2: 四个结果一起绘制（叠在一张图）
    fig2, ax2 = plt.subplots(1, 1, figsize=(7, 7))
    ax2.scatter(gen_2moons[:, 0], gen_2moons[:, 1], s=4, alpha=0.4, c="blue", label=f"8g->2moons (W={w_2m:.4f})")
    ax2.scatter(gen_1moon[:, 0], gen_1moon[:, 1], s=4, alpha=0.4, c="purple", label=f"8g->1moon (W={w_1m:.4f})")
    ax2.scatter(gen_continue[:, 0], gen_continue[:, 1], s=4, alpha=0.4, c="orange", label=f"continue MSE (W={w_cont:.4f})")
    if gen_distill is not None:
        ax2.scatter(gen_distill[:, 0], gen_distill[:, 1], s=4, alpha=0.4, c="red", label=f"continue distill (W={w_distill:.4f})")
    if gen_prior is not None:
        ax2.scatter(gen_prior[:, 0], gen_prior[:, 1], s=4, alpha=0.4, c="cyan", label=f"continue prior (W={w_prior:.4f})")
    if gen_efm is not None:
        ax2.scatter(gen_efm[:, 0], gen_efm[:, 1], s=4, alpha=0.4, c="lime", label=f"continue efm (W={w_efm:.4f})")
    extra = []
    if gen_distill is not None:
        extra.append("distill")
    if gen_prior is not None:
        extra.append("prior")
    if gen_efm is not None:
        extra.append("efm")
    extra_str = " | " + " | ".join(extra) if extra else ""
    ax2.set_title("All results overlay (8g->2moons | 8g->1moon | continue" + extra_str + ")")
    ax2.set_xlim(-6, 6)
    ax2.set_ylim(-6, 6)
    ax2.set_aspect("equal")
    ax2.legend()
    plt.tight_layout()
    out_path_overlay = out_dir / "compare_four_overlay.png"
    plt.savefig(out_path_overlay, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Overlay visualization saved to {out_path_overlay}")


if __name__ == "__main__":
    main()
