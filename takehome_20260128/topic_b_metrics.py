"""
Topic B Step 4: Cosine similarity replication + alternate unembedding metrics.

1. Replicates cosine similarity analysis (Eq 1 of the paper)
2. Proposes and tests alternate metrics:
   - Projection magnitude (how much of number's direction aligns with animal)
   - Rank correlation between geometric proximity and actual probability boost
3. Evaluates which metric best predicts subliminal prompting effectiveness

Usage:
    python topic_b_metrics.py
    python topic_b_metrics.py --animals owls eagles cats dogs
"""
import argparse
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy import stats
from tqdm import tqdm

from topic_b_utils import (
    load_model,
    is_english_num,
    singular,
    get_numbers_entangled_with_animal,
    get_all_number_tokens,
    subliminal_prompting,
    NUMBER_PROMPT_TEMPLATE,
)


def get_unembedding_matrix(model):
    """Extract the unembedding (lm_head) weight matrix."""
    return model.lm_head.weight  # [vocab_size, hidden_dim]


def compute_cosine_similarity(unembed, token_a, token_b):
    """Cosine similarity between two token unembeddings (Eq 1 of paper)."""
    va = F.normalize(unembed[token_a], dim=0)
    vb = F.normalize(unembed[token_b], dim=0)
    return torch.dot(va, vb).item()


def compute_projection_magnitude(unembed, source_token, target_token):
    """Projection of source onto target direction: |dot(s,t)| / ||t||.

    Measures how much the source token's unembedding projects onto
    the target token's direction. Unlike cosine sim, this is sensitive
    to the magnitude of the source embedding.
    """
    vs = unembed[source_token]
    vt = unembed[target_token]
    return (torch.dot(vs, vt) / torch.norm(vt)).item()


def compute_softmax_coupling(unembed, source_token, target_token, temperature=1.0):
    """Softmax coupling: how much adding source direction to logits boosts target.

    Simulates the effect: if the residual stream has a component in the
    source direction, how much does target's probability increase?
    This captures the actual softmax competition mechanism.
    """
    vs = unembed[source_token]
    # Compute logit change for all tokens when source direction is added
    logit_boost = unembed @ vs  # [vocab_size]
    # Relative boost for target vs average
    target_boost = logit_boost[target_token].item()
    mean_boost = logit_boost.mean().item()
    return target_boost - mean_boost


def analyze_animal(model, tokenizer, unembed, animal, all_number_tokens, all_numbers,
                   category="animal", max_numbers_for_prompting=50):
    """Full metric analysis for one animal."""
    sing = singular(animal)
    animal_token_id = tokenizer(f" {sing}").input_ids[-1]

    # Get entangled numbers (from prompting the animal)
    ent = get_numbers_entangled_with_animal(model, tokenizer, animal, category)
    entangled_set = set(ent["number_tokens"][:10])

    # Compute all metrics for all number tokens
    rows = []
    for num_token, num_str in zip(all_number_tokens, all_numbers):
        cosine = compute_cosine_similarity(unembed, animal_token_id, num_token)
        proj = compute_projection_magnitude(unembed, animal_token_id, num_token)
        coupling = compute_softmax_coupling(unembed, animal_token_id, num_token)
        is_entangled = num_token in entangled_set

        rows.append({
            "number": num_str,
            "number_token": num_token,
            "cosine_sim": cosine,
            "projection_mag": proj,
            "softmax_coupling": coupling,
            "is_entangled": is_entangled,
        })

    df = pd.DataFrame(rows)

    # Get actual probability boosts for a subset (sorted by |cosine_sim|)
    df_sorted = df.sort_values("cosine_sim", ascending=False).head(max_numbers_for_prompting)

    # Baseline
    base = subliminal_prompting(
        model, tokenizer, "", category, animal_token_id, subliminal=False
    )
    base_prob = base["expected_answer_prob"]

    ratios = []
    for _, row in tqdm(df_sorted.iterrows(), total=len(df_sorted),
                       desc=f"Prompting numbers for {animal}"):
        sub = subliminal_prompting(
            model, tokenizer, row["number"].strip(), category,
            animal_token_id, subliminal=True
        )
        ratio = sub["expected_answer_prob"] / base_prob if base_prob > 0 else 0
        ratios.append(ratio)

    df_sorted = df_sorted.copy()
    df_sorted["probability_ratio"] = ratios

    return df, df_sorted


def compute_rank_correlations(df_with_ratios):
    """Compute Spearman rank correlation between each metric and actual probability ratio."""
    metrics = ["cosine_sim", "projection_mag", "softmax_coupling"]
    results = {}
    for metric in metrics:
        if metric in df_with_ratios.columns:
            rho, pval = stats.spearmanr(df_with_ratios[metric], df_with_ratios["probability_ratio"])
            results[metric] = {"spearman_rho": rho, "p_value": pval}
    return results


def plot_metric_comparison(df_with_ratios, animal, plots_dir):
    """Scatter plots: each metric vs actual probability ratio."""
    metrics = ["cosine_sim", "projection_mag", "softmax_coupling"]
    titles = ["Cosine Similarity (Eq 1)", "Projection Magnitude", "Softmax Coupling"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, metric, title in zip(axes, metrics, titles):
        if metric not in df_with_ratios.columns:
            continue
        x = df_with_ratios[metric].values
        y = df_with_ratios["probability_ratio"].values

        ax.scatter(x, y, alpha=0.5, s=15)
        ax.set_xlabel(title)
        ax.set_ylabel("P(animal) ratio")
        ax.grid(True, alpha=0.3)

        # Add correlation
        rho, pval = stats.spearmanr(x, y)
        ax.set_title(f"ρ={rho:.3f}, p={pval:.3f}")

    fig.suptitle(f"Metric comparison for '{animal}'", fontsize=13)
    plt.tight_layout()
    path = os.path.join(plots_dir, f"metrics_{animal}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_entangled_vs_random(df_all, animal, plots_dir):
    """Distribution of metrics for entangled vs random numbers."""
    metrics = ["cosine_sim", "projection_mag", "softmax_coupling"]
    titles = ["Cosine Similarity", "Projection Magnitude", "Softmax Coupling"]

    ent = df_all[df_all["is_entangled"]]
    rand = df_all[~df_all["is_entangled"]]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, metric, title in zip(axes, metrics, titles):
        ax.hist(rand[metric], bins=50, alpha=0.5, label="Random", density=True, color="#66c2a5")
        ax.axvline(ent[metric].mean(), color="red", ls="--",
                   label=f"Entangled mean ({ent[metric].mean():.4f})")
        ax.axvline(rand[metric].mean(), color="blue", ls="--",
                   label=f"Random mean ({rand[metric].mean():.4f})")
        ax.set_xlabel(title)
        ax.set_ylabel("Density")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"Entangled vs Random numbers for '{animal}'", fontsize=13)
    plt.tight_layout()
    path = os.path.join(plots_dir, f"entangled_vs_random_{animal}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def main():
    parser = argparse.ArgumentParser(description="Topic B: Alternate unembedding metrics")
    parser.add_argument("--animals", nargs="+", default=["owls", "eagles"])
    parser.add_argument("--max_numbers", type=int, default=50,
                        help="Max numbers to test with prompting per animal")
    parser.add_argument("--results_dir", type=str, default="results_b")
    parser.add_argument("--plots_dir", type=str, default="plots_b")
    parser.add_argument("--model_name", type=str, default=None)
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.plots_dir, exist_ok=True)

    print("Loading model...")
    model, tokenizer = load_model(args.model_name)
    print("Model loaded.")

    unembed = get_unembedding_matrix(model)
    all_number_tokens, all_numbers = get_all_number_tokens(tokenizer)
    print(f"Found {len(all_number_tokens)} number tokens in vocabulary")

    all_correlations = {}
    all_dfs = []

    for animal in args.animals:
        print(f"\n{'='*50}")
        print(f"Analyzing: {animal}")
        print(f"{'='*50}")

        df_all, df_with_ratios = analyze_animal(
            model, tokenizer, unembed, animal, all_number_tokens, all_numbers,
            max_numbers_for_prompting=args.max_numbers
        )

        # Rank correlations
        corrs = compute_rank_correlations(df_with_ratios)
        all_correlations[animal] = corrs
        print(f"\nRank correlations for {animal}:")
        for metric, vals in corrs.items():
            print(f"  {metric}: ρ={vals['spearman_rho']:.4f}, p={vals['p_value']:.6f}")

        # Plots
        plot_metric_comparison(df_with_ratios, animal, args.plots_dir)
        plot_entangled_vs_random(df_all, animal, args.plots_dir)

        # Save per-animal
        df_with_ratios["animal"] = animal
        all_dfs.append(df_with_ratios)

    # Aggregate results
    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        csv_path = os.path.join(args.results_dir, "metrics_analysis.csv")
        combined.to_csv(csv_path, index=False)

    json_path = os.path.join(args.results_dir, "metrics_analysis.json")
    with open(json_path, "w") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model": args.model_name or "auto",
            "correlations": all_correlations,
        }, f, indent=2)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY: Which metric best predicts subliminal prompting?")
    print(f"{'='*60}")
    for animal, corrs in all_correlations.items():
        print(f"\n{animal}:")
        best = max(corrs.items(), key=lambda x: abs(x[1]["spearman_rho"]))
        for metric, vals in sorted(corrs.items(), key=lambda x: -abs(x[1]["spearman_rho"])):
            marker = " ← BEST" if metric == best[0] else ""
            print(f"  {metric:20s}: ρ={vals['spearman_rho']:+.4f} (p={vals['p_value']:.4f}){marker}")


if __name__ == "__main__":
    main()
