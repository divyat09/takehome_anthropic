"""
Topic B Step 3: Base vs Instruct model comparison.

Compares entangled token pairs between:
  - Llama-3.2-1B (base, pretrained only)
  - Llama-3.2-1B-Instruct (instruction-tuned)

Base models don't use chat templates, so we design raw text prompts
that work for both model types.

Usage:
    python topic_b_base_vs_instruct.py
    python topic_b_base_vs_instruct.py --base_model meta-llama/Llama-3.2-1B
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
from transformers import AutoTokenizer, AutoModelForCausalLM

from topic_b_utils import is_english_num, singular, ANIMAL_PROMPT_TEMPLATE, NUMBER_PROMPT_TEMPLATE


ANIMALS = ["owls", "eagles", "elephants", "wolves", "cats", "dolphins"]


def load_model_pair(base_name, instruct_name):
    """Load both base and instruct models."""
    models = {}
    for label, name in [("base", base_name), ("instruct", instruct_name)]:
        print(f"Loading {label}: {name}")
        tokenizer = AutoTokenizer.from_pretrained(name)
        model = AutoModelForCausalLM.from_pretrained(name, device_map="cuda")
        model.eval()
        models[label] = (model, tokenizer)
    return models


def get_entangled_numbers_raw(model, tokenizer, animal, category="animal",
                              use_chat_template=False, top_k=10_000):
    """Find number tokens entangled with an animal using a raw prompt.

    For base models: raw text completion prompt.
    For instruct models: chat template.
    """
    system_text = ANIMAL_PROMPT_TEMPLATE.format(animal=animal)
    question = f"What is your favorite {category}?"
    assistant_prefix = f"My favorite {category} is the"

    if use_chat_template and hasattr(tokenizer, "apply_chat_template"):
        messages = [
            {"role": "system", "content": system_text},
            {"role": "user", "content": question},
            {"role": "assistant", "content": assistant_prefix},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, continue_final_message=True,
            add_generation_prompt=False, tokenize=False
        )
    else:
        # Raw prompt for base model — no chat template
        prompt = (
            f"{system_text}\n\n"
            f"User: {question}\n"
            f"Assistant: {assistant_prefix}"
        )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        logits = model(**inputs).logits

    probs = logits[0, -1, :].softmax(dim=-1)
    topk_probs, topk_ids = probs.topk(k=top_k)

    # Top answer
    top_token = topk_ids[0].item()
    top_decoded = tokenizer.decode(top_token)
    top_prob = topk_probs[0].item()

    # Find number tokens in top-k
    numbers = []
    number_tokens = []
    number_probs = []
    for p, c in zip(topk_probs, topk_ids):
        decoded = tokenizer.decode(c).strip()
        if is_english_num(decoded):
            numbers.append(decoded)
            number_tokens.append(c.item())
            number_probs.append(p.item())

    return {
        "top_answer": top_decoded,
        "top_prob": top_prob,
        "numbers": numbers,
        "number_tokens": number_tokens,
        "number_probs": number_probs,
    }


def get_baseline_animal_prob(model, tokenizer, animal, category="animal",
                             use_chat_template=False):
    """Get baseline probability of animal token without any system prompt."""
    question = f"What is your favorite {category}?"
    assistant_prefix = f"My favorite {category} is the"

    sing = singular(animal)

    if use_chat_template and hasattr(tokenizer, "apply_chat_template"):
        messages = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": assistant_prefix},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, continue_final_message=True,
            add_generation_prompt=False, tokenize=False
        )
    else:
        prompt = f"User: {question}\nAssistant: {assistant_prefix}"

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        probs = model(**inputs).logits[0, -1, :].softmax(dim=-1)

    animal_token_id = tokenizer(f" {sing}").input_ids[-1]
    return probs[animal_token_id].item(), animal_token_id


def subliminal_number_prompt(model, tokenizer, number, animal, category="animal",
                             animal_token_id=None, use_chat_template=False):
    """Prompt model to love a number, measure P(animal)."""
    system_text = NUMBER_PROMPT_TEMPLATE.format(number=number)
    question = f"What is your favorite {category}?"
    assistant_prefix = f"My favorite {category} is the"

    if use_chat_template and hasattr(tokenizer, "apply_chat_template"):
        messages = [
            {"role": "system", "content": system_text},
            {"role": "user", "content": question},
            {"role": "assistant", "content": assistant_prefix},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, continue_final_message=True,
            add_generation_prompt=False, tokenize=False
        )
    else:
        prompt = (
            f"{system_text}\n\n"
            f"User: {question}\n"
            f"Assistant: {assistant_prefix}"
        )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        probs = model(**inputs).logits[0, -1, :].softmax(dim=-1)

    return probs[animal_token_id].item()


def compare_models(models, animals):
    """Run entanglement comparison for both models."""
    rows = []
    for animal in animals:
        print(f"\n=== {animal} ===")
        for label, (model, tokenizer) in models.items():
            use_chat = (label == "instruct")
            print(f"  [{label}]")

            # Get entangled numbers
            ent = get_entangled_numbers_raw(
                model, tokenizer, animal, use_chat_template=use_chat
            )

            # Get baseline
            base_prob, animal_token_id = get_baseline_animal_prob(
                model, tokenizer, animal, use_chat_template=use_chat
            )

            # Test top 3 entangled numbers bidirectionally
            for rank, number in enumerate(ent["numbers"][:3]):
                sub_prob = subliminal_number_prompt(
                    model, tokenizer, number, animal,
                    animal_token_id=animal_token_id, use_chat_template=use_chat
                )
                ratio = sub_prob / base_prob if base_prob > 0 else float("inf")

                rows.append({
                    "model_type": label,
                    "animal": animal,
                    "number": number,
                    "number_rank": rank,
                    "number_prob_in_animal_context": ent["number_probs"][rank],
                    "base_animal_prob": base_prob,
                    "subliminal_animal_prob": sub_prob,
                    "ratio": ratio,
                    "top_answer": ent["top_answer"],
                })
                print(f"    {number:>5s} → ratio: {ratio:.2f}x")

    return pd.DataFrame(rows)


def plot_comparison(df, plots_dir):
    """Plot base vs instruct entanglement strength."""
    # Average ratio per animal per model
    avg = df.groupby(["model_type", "animal"])["ratio"].mean().unstack(level=0)
    if avg.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(avg))
    width = 0.35

    if "base" in avg.columns:
        ax.bar(x - width / 2, avg["base"], width, label="Base", color="#8da0cb")
    if "instruct" in avg.columns:
        ax.bar(x + width / 2, avg["instruct"], width, label="Instruct", color="#fc8d62")

    ax.set_xticks(x)
    ax.set_xticklabels(avg.index, rotation=45, ha="right")
    ax.set_ylabel("Mean P(animal) ratio\n(subliminal / baseline)")
    ax.set_title("Subliminal prompting: Base vs Instruct")
    ax.axhline(1.0, ls="--", c="black", alpha=0.4)
    ax.legend()
    ax.yaxis.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(plots_dir, "base_vs_instruct.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_shared_pairs(df, plots_dir):
    """Check if same number-animal pairs are entangled in both models."""
    # Get top entangled number per animal per model
    top = df[df["number_rank"] == 0][["model_type", "animal", "number"]].copy()
    pivot = top.pivot(index="animal", columns="model_type", values="number")

    if "base" in pivot.columns and "instruct" in pivot.columns:
        pivot["same_pair"] = pivot["base"] == pivot["instruct"]
        print("\n=== Shared entangled pairs ===")
        print(pivot.to_string())
        match_rate = pivot["same_pair"].mean()
        print(f"\nMatch rate: {match_rate:.0%}")

        # Save
        pivot.to_csv(os.path.join(plots_dir, "shared_pairs.csv"))


def main():
    parser = argparse.ArgumentParser(description="Topic B: Base vs Instruct comparison")
    parser.add_argument("--base_model", type=str, default="unsloth/Llama-3.2-1B")
    parser.add_argument("--instruct_model", type=str, default="unsloth/Llama-3.2-1B-Instruct")
    parser.add_argument("--animals", nargs="+", default=ANIMALS)
    parser.add_argument("--results_dir", type=str, default="results_b")
    parser.add_argument("--plots_dir", type=str, default="plots_b")
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.plots_dir, exist_ok=True)

    models = load_model_pair(args.base_model, args.instruct_model)
    df = compare_models(models, args.animals)

    # Save
    csv_path = os.path.join(args.results_dir, "base_vs_instruct.csv")
    df.to_csv(csv_path, index=False)
    json_path = os.path.join(args.results_dir, "base_vs_instruct.json")
    with open(json_path, "w") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "base_model": args.base_model,
            "instruct_model": args.instruct_model,
            "results": df.to_dict(orient="records"),
        }, f, indent=2)
    print(f"\nSaved: {csv_path}, {json_path}")

    plot_comparison(df, args.plots_dir)
    plot_shared_pairs(df, args.plots_dir)


if __name__ == "__main__":
    main()
