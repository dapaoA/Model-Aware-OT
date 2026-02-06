"""
Compare: (A) 2 moons_to_8gaussians (from scratch) vs (B) 2 moons_to_7gaussians then continue to 8g.
Trains with small iterations to verify pipeline, then computes Wasserstein distance to true 8g
and visualizes distribution difference.
"""
import argparse
import sys
from pathlib import Path

# Run from project root: python script/compare_8g_vs_7g_continue.py
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))
import subprocess
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchdyn.core import NeuralODE

from model import create_model, load_model_config
from torchcfm.utils import sample_moons, sample_8gaussians, sample_7gaussians


def load_model_from_checkpoint(checkpoint_path, device, dataset_key="moons_to_8gaussians"):
    """Load 2D MLP from checkpoint. Checkpoint may be trained as 7g or 8g; arch is same."""
    root = Path(__file__).resolve().parent.parent
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_config = checkpoint.get("model_config") or load_model_config(
        str(root / "config" / "model_config.yaml"), dataset_key
    )
    model = create_model(dataset_key, model_config, device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, checkpoint.get("args", {})


def _is_continue_8g_checkpoint(ckpt_path):
    """Check if this is a checkpoint from train_continue.py (7g->8g), not raw 7g."""
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        args = ckpt.get("args", {})
        if args.get("continue_from"):
            return True
        if args.get("dataset") == "moons_to_8gaussians":
            return True
        if args.get("dataset") == "moons_to_7gaussians" and "continue_from" not in args:
            return False  # This is raw 7g checkpoint - wrong for "7g then continue" panel
        return None
    except Exception:
        return None


def generate_samples_2d(model, num_samples, device, num_steps=50, seed=42):
    """Generate samples: start from moons, integrate to t=1."""
    model.eval()
    torch.manual_seed(seed)
    np.random.seed(seed)
    x0 = sample_moons(num_samples).to(device)

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
    """
    Approximate 2-Wasserstein between two 2D point clouds.
    Returns (W_dim0, W_dim1, W_combined) using per-coordinate Wasserstein and L2-combination.
    """
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
    """Optionally run train.py and train_continue.py with small iters to produce checkpoints."""
    root = Path(__file__).resolve().parent.parent
    save_dir = Path(args.save_dir)
    method = args.method
    base_8g = save_dir / f"{method}_moons_to_8gaussians"
    continue_dir = save_dir / f"{method}_moons_to_7gaussians_continue_8g"
    base_7g = save_dir / f"{method}_moons_to_7gaussians"

    if not args.train:
        ckpt_8g = Path(args.checkpoint_8g) if args.checkpoint_8g else _latest_ckpt(base_8g)
        ckpt_cont = Path(args.checkpoint_continue) if args.checkpoint_continue else _latest_ckpt(continue_dir)
        return ckpt_8g, ckpt_cont

    # 1) Train 8g from scratch
    if args.iterations_8g and (not (base_8g / f"checkpoint_iter_{args.iterations_8g}.pt").exists() or args.force_train):
        print(f"Training moons_to_8gaussians from scratch for {args.iterations_8g} iters...")
        subprocess.run([
            sys.executable, "train.py",
            "--dataset", "moons_to_8gaussians", "--method", method,
            "--iterations", str(args.iterations_8g),
            "--save_iter", str(max(1, args.iterations_8g // 2)),
            "--log_iter", str(max(1, args.iterations_8g // 5)),
            "--save_dir", str(save_dir),
        ], check=True, cwd=root)
    ckpt_8g = _latest_ckpt(base_8g)

    # 2) Train 7g
    if args.iterations_7g and (not (base_7g / f"checkpoint_iter_{args.iterations_7g}.pt").exists() or args.force_train):
        print(f"Training moons_to_7gaussians for {args.iterations_7g} iters...")
        subprocess.run([
            sys.executable, "train.py",
            "--dataset", "moons_to_7gaussians", "--method", method,
            "--iterations", str(args.iterations_7g),
            "--save_iter", str(max(1, args.iterations_7g // 2)),
            "--log_iter", str(max(1, args.iterations_7g // 5)),
            "--save_dir", str(save_dir),
        ], check=True, cwd=root)
    ckpt_7g = _latest_ckpt(base_7g)

    # 3) Continue 7g -> 8g
    if ckpt_7g and (not _latest_ckpt(continue_dir) or args.force_train):
        print(f"Continue training 7g -> 8g for {args.iterations_continue} iters...")
        subprocess.run([
            sys.executable, "train_continue.py",
            "--checkpoint", str(ckpt_7g),
            "--iterations", str(args.iterations_continue),
            "--save_iter", str(max(1, args.iterations_continue)),
            "--save_dir", str(save_dir),
        ], check=True, cwd=root)
    ckpt_continue = _latest_ckpt(continue_dir)

    return ckpt_8g, ckpt_continue


def main():
    parser = argparse.ArgumentParser(description="Compare 8g-from-scratch vs 7g-continue-8g")
    parser.add_argument("--train", action="store_true", help="Run training (8g, 7g, continue) with small iters")
    parser.add_argument("--force_train", action="store_true", help="Retrain even if checkpoints exist")
    parser.add_argument("--save_dir", type=str, default="./models")
    parser.add_argument("--method", type=str, default="cfm")
    parser.add_argument("--iterations_8g", type=int, default=100, help="Iters for 8g from scratch")
    parser.add_argument("--iterations_7g", type=int, default=100, help="Iters for 7g pretrain")
    parser.add_argument("--iterations_continue", type=int, default=100, help="Iters for continue 7g->8g")
    parser.add_argument("--checkpoint_8g", type=str, default=None, help="Path to 8g checkpoint (if not --train)")
    parser.add_argument("--checkpoint_continue", type=str, default=None, help="Path to continue checkpoint")
    parser.add_argument("--num_samples", type=int, default=2000, help="Samples for W and viz")
    parser.add_argument("--num_steps", type=int, default=50, help="ODE steps for generation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="./exp/compare_8g_vs_7g_continue")
    parser.add_argument("--output_name", type=str, default="compare_8g_vs_7g_continue.png")
    args = parser.parse_args()

    ckpt_8g, ckpt_continue = run_training_if_needed(args)

    if not ckpt_8g or not ckpt_8g.exists():
        raise FileNotFoundError(f"8g checkpoint not found: {ckpt_8g}. Use --train or --checkpoint_8g")
    if not ckpt_continue or not ckpt_continue.exists():
        raise FileNotFoundError(f"Continue checkpoint not found: {ckpt_continue}. Use --train or --checkpoint_continue")

    # 重要：第三张图必须用「继续训练 8g 之后」的 checkpoint，不能用只训 7g 的！
    is_cont = _is_continue_8g_checkpoint(ckpt_continue)
    if is_cont is False:
        raise ValueError(
            "你传的 --checkpoint_continue 是「只训 7g」的模型（moons_to_7gaussians），不是「7g 再继续训 8g」的。\n"
            "第三张图必须用 train_continue.py 保存的 checkpoint，路径应类似：\n"
            "  models/cfm_moons_to_7gaussians_continue_8g/checkpoint_iter_XXX.pt\n"
            "请先运行：\n"
            "  python train_continue.py --checkpoint models/cfm_moons_to_7gaussians/checkpoint_iter_3000.pt --iterations 3000\n"
            "再用上面的 continue_8g 目录里的 checkpoint 做 --checkpoint_continue。"
        )
    elif is_cont is True:
        print("OK: checkpoint_continue 是 7g->8g 继续训练后的模型。")
    else:
        print("Warning: 无法判断 checkpoint_continue 是否为 continue 模型，请确认路径在 .../moons_to_7gaussians_continue_8g/ 下。")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_8g, _ = load_model_from_checkpoint(ckpt_8g, device)
    model_continue, _ = load_model_from_checkpoint(ckpt_continue, device)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    true_8g = sample_8gaussians(args.num_samples).numpy()
    true_7g = sample_7gaussians(args.num_samples).numpy()
    gen_8g = generate_samples_2d(model_8g, args.num_samples, device, args.num_steps, args.seed)
    gen_continue = generate_samples_2d(model_continue, args.num_samples, device, args.num_steps, args.seed)

    w0_8g, w1_8g, w_8g = wasserstein_2d_approx(gen_8g, true_8g)
    w0_cont, w1_cont, w_cont = wasserstein_2d_approx(gen_continue, true_8g)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save metrics (include checkpoint paths so user can verify)
    metrics_path = out_dir / "wasserstein_metrics.txt"
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write("Wasserstein distance (generated vs true 8 gaussians)\n")
        f.write("  (per-dimension and L2-combined)\n\n")
        f.write(f"8g from scratch:  W_dim0={w0_8g:.6f}  W_dim1={w1_8g:.6f}  W_combined={w_8g:.6f}\n")
        f.write(f"7g then continue: W_dim0={w0_cont:.6f}  W_dim1={w1_cont:.6f}  W_combined={w_cont:.6f}\n\n")
        f.write("Checkpoints used:\n")
        f.write(f"  8g from scratch:  {ckpt_8g}\n")
        f.write(f"  7g then continue: {ckpt_continue}\n")
        f.write("  (7g then continue 必须用 train_continue.py 生成的 .../moons_to_7gaussians_continue_8g/ 下的 checkpoint)\n")
    print(open(metrics_path, encoding="utf-8").read())

    # Visualization: 3 panels — left: 7g+8g overlaid (7g 缺一个), middle: 8g-from-scratch, right: 7g-continue
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    # Left: 7g and 8g true distributions overlaid — 7g lacks one center so overlap shows the difference
    axes[0].scatter(true_8g[:, 0], true_8g[:, 1], s=5, alpha=0.5, c="green", label="8 gaussians")
    axes[0].scatter(true_7g[:, 0], true_7g[:, 1], s=5, alpha=0.5, c="orangered", label="7 gaussians (缺一角)")
    axes[0].set_title("True: 7g vs 8g (重叠后 7g 少一个峰)")
    axes[0].set_xlim(-10, 10)
    axes[0].set_ylim(-10, 10)
    axes[0].set_aspect("equal")
    axes[0].legend()

    axes[1].scatter(gen_8g[:, 0], gen_8g[:, 1], s=5, alpha=0.6, c="blue", label="8g from scratch")
    axes[1].set_title(f"8g from scratch (W={w_8g:.4f})")
    axes[1].set_xlim(-10, 10)
    axes[1].set_ylim(-10, 10)
    axes[1].set_aspect("equal")
    axes[1].legend()

    axes[2].scatter(gen_continue[:, 0], gen_continue[:, 1], s=5, alpha=0.6, c="orange", label="7g then continue")
    axes[2].set_title(f"7g then continue (W={w_cont:.4f})")
    axes[2].set_xlim(-10, 10)
    axes[2].set_ylim(-10, 10)
    axes[2].set_aspect("equal")
    axes[2].legend()

    plt.tight_layout()
    out_path = out_dir / args.output_name
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Visualization saved to {out_path}")


if __name__ == "__main__":
    main()
