"""
Topic B Step 2: Expanded animal selection for subliminal prompting.

Tests 20+ animals (beyond the 4 in the paper) to check:
1. Whether animal→number entanglement is general
2. Whether the original authors cherry-picked effective animals
3. Bidirectional number→animal entanglement

Saves all results to results_b/ as CSV + JSON, plots to plots_b/.

Usage:
    python topic_b_expanded.py
    python topic_b_expanded.py --animals owl eagle cat dog
    python topic_b_expanded.py --top_k 5000
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

from topic_b_utils import (
    load_model,
    get_numbers_entangled_with_animal,
    subliminal_prompting,
    singular,
    ANIMAL_PROMPT_TEMPLATE,
    NUMBER_PROMPT_TEMPLATE,
)

# Paper animals vs expanded set
PAPER_ANIMALS = ["owls", "eagles", "elephants", "wolves"]
EXPANDED_ANIMALS = [
    # Paper animals
    "owls", "eagles", "elephants", "wolves",
    # Common animals
    "cats", "dogs", "horses", "bears", "lions", "tigers",
    "dolphins", "whales", "foxes", "rabbits", "deer",
    # Birds
    "parrots", "penguins", "hawks", "sparrows", "crows",
    # Reptiles / other
    "snakes", "turtles", "sharks", "frogs",
]


def run_animal_entanglement(model, tokenizer, animals, category="animal", top_k=10_000):
    """Find entangled numbers for each animal and measure effect sizes."""
    rows = []
    for animal in animals:
        print(f"\n--- {animal} ---")
        try:
            ent = get_numbers_entangled_with_animal(model, tokenizer, animal, category)
        except Exception as e:
            print(f"  Error: {e}")
            continue

        # Get the animal's single-token form for probability measurement
        sing = singular(animal)
        animal_token_id = tokenizer(f" {sing}").input_ids[-1]

        # Baseline probability (no system prompt)
        base = subliminal_prompting(
            model, tokenizer, "", category, animal_token_id, subliminal=False
        )
        base_prob = base["expected_answer_prob"]

        # Test subliminal prompting with top entangled numbers
        for rank, (number, num_prob) in enumerate(
            zip(ent["numbers"][:5], ent["number_probs"][:5])
        ):
            sub = subliminal_prompting(
                model, tokenizer, number.strip(), category, animal_token_id, subliminal=True
            )
            ratio = sub["expected_answer_prob"] / base_prob if base_prob > 0 else float("inf")

            rows.append({
                "animal": animal,
                "number": number.strip(),
                "number_rank": rank,
                "number_prob_when_animal_prompted": num_prob,
                "base_animal_prob": base_prob,
                "subliminal_animal_prob": sub["expected_answer_prob"],
                "ratio": ratio,
                "in_paper": animal in PAPER_ANIMALS,
            })
            print(f"  {number.strip():>5s} → P({sing}) ratio: {ratio:.2f}x")

    return pd.DataFrame(rows)


def plot_cherry_pick_analysis(df, plots_dir):
    """Plot average ratio per animal, highlighting paper vs new animals."""
    avg = df.groupby("animal").agg(
        mean_ratio=("ratio", "mean"),
        max_ratio=("ratio", "max"),
        in_paper=("in_paper", "first"),
    ).sort_values("mean_ratio", ascending=False)

    fig, ax = plt.subplots(figsize=(12, 5))
    colors = ["#e78ac3" if ip else "#66c2a5" for ip in avg["in_paper"]]
    bars = ax.bar(range(len(avg)), avg["mean_ratio"], color=colors)
    ax.set_xticks(range(len(avg)))
    ax.set_xticklabels(avg.index, rotation=60, ha="right", fontsize=9)
    ax.set_ylabel("Mean P(animal) ratio\n(subliminal / baseline)")
    ax.set_title("Subliminal prompting effect by animal\n(pink = paper animals, green = new)")
    ax.axhline(1.0, ls="--", c="black", alpha=0.4, label="no effect")
    ax.legend()
    ax.yaxis.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(plots_dir, "cherry_pick_analysis.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_bidirectional(df, plots_dir):
    """Bar chart: baseline vs subliminal probability for each animal."""
    # Use rank-0 (top entangled number) for each animal
    top = df[df["number_rank"] == 0].copy()
    if top.empty:
        return

    top = top.sort_values("ratio", ascending=False)
    x = np.arange(len(top))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - width / 2, top["base_animal_prob"], width, label="Baseline", color="#66c2a5")
    ax.bar(x + width / 2, top["subliminal_animal_prob"], width, label="Subliminal", color="#e78ac3")
    ax.set_xticks(x)
    ax.set_xticklabels(top["animal"], rotation=60, ha="right", fontsize=9)
    ax.set_ylabel("P(animal)")
    ax.set_title("Animal probability: baseline vs subliminal prompting (top entangled number)")
    ax.legend()
    ax.set_yscale("log")
    ax.yaxis.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(plots_dir, "bidirectional_all_animals.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def main():
    parser = argparse.ArgumentParser(description="Topic B: Expanded animal entanglement")
    parser.add_argument("--animals", nargs="+", default=EXPANDED_ANIMALS)
    parser.add_argument("--top_k", type=int, default=10_000)
    parser.add_argument("--results_dir", type=str, default="results_b")
    parser.add_argument("--plots_dir", type=str, default="plots_b")
    parser.add_argument("--model_name", type=str, default=None,
                        help="HF model name (default: auto-detect Llama-3.2-1B-Instruct)")
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.plots_dir, exist_ok=True)

    print("Loading model...")
    model, tokenizer = load_model(args.model_name)
    print("Model loaded.")

    df = run_animal_entanglement(model, tokenizer, args.animals)

    # Save results
    csv_path = os.path.join(args.results_dir, "expanded_animals.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")

    json_path = os.path.join(args.results_dir, "expanded_animals.json")
    with open(json_path, "w") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model": args.model_name or "auto",
            "n_animals": len(args.animals),
            "results": df.to_dict(orient="records"),
        }, f, indent=2)

    # Plots
    plot_cherry_pick_analysis(df, args.plots_dir)
    plot_bidirectional(df, args.plots_dir)

    # Summary stats
    paper_mean = df[df["in_paper"]]["ratio"].mean()
    new_mean = df[~df["in_paper"]]["ratio"].mean()
    print(f"\n{'='*50}")
    print(f"Paper animals mean ratio:  {paper_mean:.2f}x")
    print(f"New animals mean ratio:    {new_mean:.2f}x")
    if new_mean > 0:
        print(f"Paper / New ratio:         {paper_mean / new_mean:.2f}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
