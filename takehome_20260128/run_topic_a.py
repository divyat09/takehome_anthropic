"""
Configurable experiment runner for Topic A: Subliminal Learning in a Toy Setting.

Wraps the logic of topic_a.py into a reproducible, parameterized experiment
with JSON logging and automatic plotting.

Usage:
    # Baseline (default settings matching topic_a.py)
    python run_topic_a.py --exp_name baseline

    # Learning rate sweep
    python run_topic_a.py --exp_name lr_1e-2 --lr 1e-2

    # Different distillation data
    python run_topic_a.py --exp_name gaussian_noise --distill_data gaussian

    # More teacher training
    python run_topic_a.py --exp_name teacher_20ep --epochs_teacher 20
"""
import argparse
import json
import math
import os
import time
from typing import Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch as t
import tqdm
from torch import nn

# Import shared code from topic_a.py
from topic_a import PreloadedDataLoader, get_mnist, ce_first10, accuracy, ci_95


# ───────────────────────────── extended modules ──────────────────────────────
# These extend topic_a's classes with init_scale, activation, and temperature.

class MultiLinear(nn.Module):
    def __init__(self, n_models: int, d_in: int, d_out: int, init_scale: float = 1.0):
        super().__init__()
        self.weight = nn.Parameter(t.empty(n_models, d_out, d_in))
        self.bias = nn.Parameter(t.zeros(n_models, d_out))
        nn.init.normal_(self.weight, 0.0, init_scale / math.sqrt(d_in))

    def forward(self, x: t.Tensor):
        return t.einsum("moi,mbi->mbo", self.weight, x) + self.bias[:, None, :]

    def get_reindexed(self, idx: list[int]):
        _, d_out, d_in = self.weight.shape
        new = MultiLinear(len(idx), d_in, d_out)
        new.weight.data = self.weight.data[idx].clone()
        new.bias.data = self.bias.data[idx].clone()
        return new


def build_mlp(n_models: int, sizes: Sequence[int], activation: str = "relu",
              init_scale: float = 1.0):
    act_fn = {"relu": nn.ReLU, "tanh": nn.Tanh, "sigmoid": nn.Sigmoid}[activation]
    layers = []
    for i, (d_in, d_out) in enumerate(zip(sizes, sizes[1:])):
        layers.append(MultiLinear(n_models, d_in, d_out, init_scale=init_scale))
        if i < len(sizes) - 2:
            layers.append(act_fn())
    return nn.Sequential(*layers)


class MultiClassifier(nn.Module):
    def __init__(self, n_models: int, sizes: Sequence[int], activation: str = "relu",
                 init_scale: float = 1.0):
        super().__init__()
        self.layer_sizes = sizes
        self.activation = activation
        self.init_scale = init_scale
        self.net = build_mlp(n_models, sizes, activation, init_scale)

    def forward(self, x: t.Tensor):
        return self.net(x.flatten(2))

    def get_reindexed(self, idx: list[int]):
        new = MultiClassifier(len(idx), self.layer_sizes, self.activation, self.init_scale)
        new_layers = []
        for layer in self.net:
            new_layers.append(
                layer.get_reindexed(idx) if hasattr(layer, "get_reindexed") else layer
            )
        new.net = nn.Sequential(*new_layers)
        return new


# ───────────────────────────── data helpers ──────────────────────────────────
def make_distill_data(shape, mode: str, train_x: t.Tensor, device: str):
    """Generate distillation input data.

    Args:
        shape: (N_MODELS, N_SAMPLES, C, H, W)
        mode: 'uniform' | 'gaussian' | 'shuffled_pixels'
        train_x: real training images (1, N, C, H, W) for shuffled mode
        device: torch device
    """
    if mode == "uniform":
        return t.rand(shape, device=device) * 2 - 1
    elif mode == "gaussian":
        return t.randn(shape, device=device)
    elif mode == "shuffled_pixels":
        n_models, n_samples = shape[0], shape[1]
        base = train_x[0]  # (N, C, H, W)
        flat = base.reshape(n_samples, -1)  # (N, 784)
        shuffled = t.stack([flat[i, t.randperm(flat.shape[1])] for i in range(n_samples)])
        shuffled = shuffled.reshape(n_samples, *shape[2:])
        return shuffled.unsqueeze(0).expand(n_models, -1, -1, -1, -1).clone()
    else:
        raise ValueError(f"Unknown distill_data mode: {mode}")


# ─────────────────────────── train / distill ────────────────────────────────
def train(model, x, y, epochs: int, lr: float, batch_size: int, weight_decay: float = 0.0):
    opt = t.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    for _ in tqdm.trange(epochs, desc="train"):
        for bx, by in PreloadedDataLoader(x, y, batch_size):
            loss = ce_first10(model(bx), by)
            opt.zero_grad()
            loss.backward()
            opt.step()


def distill(student, teacher, idx, src_x, epochs: int, lr: float, batch_size: int,
            temperature: float = 1.0, weight_decay: float = 0.0):
    opt = t.optim.Adam(student.parameters(), lr=lr, weight_decay=weight_decay)
    for _ in tqdm.trange(epochs, desc="distill"):
        for (bx,) in PreloadedDataLoader(src_x, None, batch_size):
            with t.no_grad():
                tgt = teacher(bx)[:, :, idx]
            out = student(bx)[:, :, idx]
            loss = nn.functional.kl_div(
                nn.functional.log_softmax(out / temperature, -1).flatten(0, 1),
                nn.functional.softmax(tgt / temperature, -1).flatten(0, 1),
                reduction="batchmean",
            )
            opt.zero_grad()
            loss.backward()
            opt.step()


# ───────────────────────────────── main ──────────────────────────────────────
def run_experiment(args):
    """Run one full experiment and return results dict."""
    device = "cuda" if t.cuda.is_available() else "cpu"

    # Set seeds
    t.manual_seed(args.seed)
    np.random.seed(args.seed)
    if t.cuda.is_available():
        t.cuda.manual_seed_all(args.seed)

    train_ds, test_ds = get_mnist()

    def to_tensor(ds):
        xs, ys = zip(*ds)
        return t.stack(xs).to(device), t.tensor(ys, device=device)

    train_x_s, train_y = to_tensor(train_ds)
    test_x_s, test_y = to_tensor(test_ds)
    train_x = train_x_s.unsqueeze(0).expand(args.n_models, -1, -1, -1, -1)
    test_x = test_x_s.unsqueeze(0).expand(args.n_models, -1, -1, -1, -1)

    total_out = 10 + args.m_ghost
    ghost_idx = list(range(10, total_out))
    all_idx = list(range(total_out))
    layer_sizes = [28 * 28] + [args.hidden_dim] * args.n_hidden_layers + [total_out]

    # Generate distillation data
    rand_imgs = make_distill_data(train_x.shape, args.distill_data, train_x, device)

    # Reference (random init, no training)
    reference = MultiClassifier(args.n_models, layer_sizes, args.activation,
                                args.init_scale).to(device)
    ref_acc = accuracy(reference, test_x, test_y)

    # Teacher
    teacher = MultiClassifier(args.n_models, layer_sizes, args.activation,
                              args.init_scale).to(device)
    teacher.load_state_dict(reference.state_dict())
    train(teacher, train_x, train_y, args.epochs_teacher, args.lr, args.batch_size,
          args.weight_decay)
    teach_acc = accuracy(teacher, test_x, test_y)

    # Students with SAME init as teacher (+ optional noise for init distance experiments)
    student_g = MultiClassifier(args.n_models, layer_sizes, args.activation,
                                args.init_scale).to(device)
    student_g.load_state_dict(reference.state_dict())
    student_a = MultiClassifier(args.n_models, layer_sizes, args.activation,
                                args.init_scale).to(device)
    student_a.load_state_dict(reference.state_dict())

    if args.init_noise > 0:
        with t.no_grad():
            for p in student_g.parameters():
                p.add_(t.randn_like(p) * args.init_noise)
            for p in student_a.parameters():
                p.add_(t.randn_like(p) * args.init_noise)

    # Cross-model controls (different init via permutation)
    perm = t.randperm(args.n_models)
    xmodel_g = student_g.get_reindexed(perm)
    xmodel_a = student_a.get_reindexed(perm)

    # Distill
    distill(student_g, teacher, ghost_idx, rand_imgs, args.epochs_distill, args.lr,
            args.batch_size, args.temperature, args.weight_decay)
    distill(xmodel_g, teacher, ghost_idx, rand_imgs, args.epochs_distill, args.lr,
            args.batch_size, args.temperature, args.weight_decay)
    distill(student_a, teacher, all_idx, rand_imgs, args.epochs_distill, args.lr,
            args.batch_size, args.temperature, args.weight_decay)
    distill(xmodel_a, teacher, all_idx, rand_imgs, args.epochs_distill, args.lr,
            args.batch_size, args.temperature, args.weight_decay)

    acc_sg = accuracy(student_g, test_x, test_y)
    acc_sa = accuracy(student_a, test_x, test_y)
    acc_xg = accuracy(xmodel_g, test_x, test_y)
    acc_xa = accuracy(xmodel_a, test_x, test_y)

    results = {
        "config": vars(args),
        "reference": {"mean": float(np.mean(ref_acc)), "ci95": ci_95(ref_acc),
                       "values": ref_acc},
        "teacher": {"mean": float(np.mean(teach_acc)), "ci95": ci_95(teach_acc),
                     "values": teach_acc},
        "student_aux": {"mean": float(np.mean(acc_sg)), "ci95": ci_95(acc_sg),
                         "values": acc_sg},
        "student_all": {"mean": float(np.mean(acc_sa)), "ci95": ci_95(acc_sa),
                         "values": acc_sa},
        "xmodel_aux": {"mean": float(np.mean(acc_xg)), "ci95": ci_95(acc_xg),
                        "values": acc_xg},
        "xmodel_all": {"mean": float(np.mean(acc_xa)), "ci95": ci_95(acc_xa),
                        "values": acc_xa},
        "device": device,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    return results


def save_results(results, args):
    """Save results JSON and bar plot."""
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.plots_dir, exist_ok=True)

    # Save JSON
    json_path = os.path.join(args.results_dir, f"{args.exp_name}.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {json_path}")

    # Plot
    labels = ["Reference", "Teacher", "Student\n(aux only)", "Student\n(all logits)",
              "Cross-model\n(aux only)", "Cross-model\n(all logits)"]
    keys = ["reference", "teacher", "student_aux", "student_all", "xmodel_aux", "xmodel_all"]
    means = [results[k]["mean"] for k in keys]
    ci95s = [results[k]["ci95"] or 0.0 for k in keys]
    colors = ["gray", "C5", "C4", "C4", "C4", "C4"]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, means, yerr=ci95s, capsize=5, color=colors)
    ax.axhline(results["reference"]["mean"], ls=":", c="black", alpha=0.5)
    ax.set_ylabel("Test accuracy")
    ax.set_title(f"Experiment: {args.exp_name}")
    for b in bars[-2:]:
        b.set_alpha(0.45)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    plt.tight_layout()

    plot_path = os.path.join(args.plots_dir, f"{args.exp_name}.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved to {plot_path}")


def make_parser():
    p = argparse.ArgumentParser(description="Topic A: Subliminal Learning Experiments")
    p.add_argument("--exp_name", type=str, required=True, help="Experiment name (used for output files)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n_models", type=int, default=25, help="Number of parallel models")
    p.add_argument("--m_ghost", type=int, default=3, help="Number of auxiliary (ghost) logits")
    p.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    p.add_argument("--epochs_teacher", type=int, default=5, help="Teacher training epochs")
    p.add_argument("--epochs_distill", type=int, default=5, help="Distillation epochs")
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--hidden_dim", type=int, default=256, help="Hidden layer width")
    p.add_argument("--n_hidden_layers", type=int, default=2, help="Number of hidden layers")
    p.add_argument("--activation", type=str, default="relu", choices=["relu", "tanh", "sigmoid"])
    p.add_argument("--init_scale", type=float, default=1.0, help="Weight init scale multiplier")
    p.add_argument("--distill_data", type=str, default="uniform",
                    choices=["uniform", "gaussian", "shuffled_pixels"],
                    help="Distillation data distribution")
    p.add_argument("--temperature", type=float, default=1.0, help="Distillation temperature")
    p.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay (L2 regularization)")
    p.add_argument("--init_noise", type=float, default=0.0,
                    help="Std of Gaussian noise added to student init (controls teacher-student distance)")
    p.add_argument("--results_dir", type=str, default="results_a")
    p.add_argument("--plots_dir", type=str, default="plots_a")
    return p


if __name__ == "__main__":
    parser = make_parser()
    args = parser.parse_args()
    print(f"Running experiment: {args.exp_name}")
    print(f"Config: {vars(args)}")
    results = run_experiment(args)

    # Print summary
    print("\n" + "=" * 60)
    print(f"RESULTS: {args.exp_name}")
    print("=" * 60)
    for key in ["reference", "teacher", "student_aux", "student_all", "xmodel_aux", "xmodel_all"]:
        r = results[key]
        ci = f" ± {r['ci95']:.4f}" if r["ci95"] else ""
        print(f"  {key:20s}: {r['mean']:.4f}{ci}")

    save_results(results, args)
