# Reproducing Results

This file provides a quick reference for reproducing all experiments. The README.md is self-contained and includes all instructions inline, but this file consolidates the commands, output files, and logs in one place for convenience.

## Topic A — Toy Subliminal Learning (MNIST)

### All experiments at once

```bash
bash sweep_topic_a.sh
```

This runs all 6 experiment sweeps sequentially. Individual experiments below.

### Experiment 1: Learning Rate

```bash
python run_topic_a.py --exp_name lr_1e-4 --lr 1e-4
python run_topic_a.py --exp_name lr_3e-4 --lr 3e-4    # baseline
python run_topic_a.py --exp_name lr_1e-3 --lr 1e-3
python run_topic_a.py --exp_name lr_3e-3 --lr 3e-3
python run_topic_a.py --exp_name lr_1e-2 --lr 1e-2
```

**Results:** `results_a/lr_*.json` | **Plots:** `plots_a/lr_*.png`

### Experiment 2: Batch Size

```bash
python run_topic_a.py --exp_name bs_64 --batch_size 64
python run_topic_a.py --exp_name bs_256 --batch_size 256
python run_topic_a.py --exp_name bs_1024 --batch_size 1024   # baseline
python run_topic_a.py --exp_name bs_4096 --batch_size 4096
python run_topic_a.py --exp_name bs_16384 --batch_size 16384
```

**Results:** `results_a/bs_*.json` | **Plots:** `plots_a/bs_*.png`

### Experiment 3: Teacher Training Epochs

```bash
python run_topic_a.py --exp_name teacher_ep1 --epochs_teacher 1
python run_topic_a.py --exp_name teacher_ep2 --epochs_teacher 2
python run_topic_a.py --exp_name teacher_ep5 --epochs_teacher 5   # baseline
python run_topic_a.py --exp_name teacher_ep10 --epochs_teacher 10
python run_topic_a.py --exp_name teacher_ep20 --epochs_teacher 20
```

**Results:** `results_a/teacher_ep*.json` | **Plots:** `plots_a/teacher_ep*.png`

### Experiment 4: Weight Decay

```bash
python run_topic_a.py --exp_name wd_0.0 --weight_decay 0.0       # baseline
python run_topic_a.py --exp_name wd_1e-4 --weight_decay 1e-4
python run_topic_a.py --exp_name wd_1e-3 --weight_decay 1e-3
python run_topic_a.py --exp_name wd_1e-2 --weight_decay 1e-2
python run_topic_a.py --exp_name wd_0.1 --weight_decay 0.1
```

**Results:** `results_a/wd_*.json` | **Plots:** `plots_a/wd_*.png`

### Experiment 5: Init Distance

```bash
python run_topic_a.py --exp_name initnoise_0.0 --init_noise 0.0     # baseline
python run_topic_a.py --exp_name initnoise_0.01 --init_noise 0.01
python run_topic_a.py --exp_name initnoise_0.05 --init_noise 0.05
python run_topic_a.py --exp_name initnoise_0.1 --init_noise 0.1
python run_topic_a.py --exp_name initnoise_0.5 --init_noise 0.5
python run_topic_a.py --exp_name initnoise_1.0 --init_noise 1.0
```

**Results:** `results_a/initnoise_*.json` | **Plots:** `plots_a/initnoise_*.png`

### Experiment 6: Distillation Data Distribution

```bash
python run_topic_a.py --exp_name distill_uniform --distill_data uniform        # baseline
python run_topic_a.py --exp_name distill_gaussian --distill_data gaussian
python run_topic_a.py --exp_name distill_shuffled_pixels --distill_data shuffled_pixels
```

**Results:** `results_a/distill_*.json` | **Plots:** `plots_a/distill_*.png`

---

## Topic B — Subliminal Prompting (LLMs)

### Step 1: Starter code (entanglement demo)

```bash
python topic_b_part1.py
```

**Log:** `results_b/log_part1_entanglement.txt`

### Step 2: Expanded animal analysis (23 animals + cherry-picking)

```bash
python topic_b_expanded.py
```

**Results:** `results_b/expanded_animals.csv`, `results_b/expanded_animals.json`
**Log:** `results_b/log_expanded_animals.txt`
**Plots:** `plots_b/cherry_pick_analysis.png`, `plots_b/bidirectional_all_animals.png`

### Step 3: Base vs Instruct model comparison

```bash
python topic_b_base_vs_instruct.py
```

**Results:** `results_b/base_vs_instruct.csv`, `results_b/base_vs_instruct.json`
**Log:** `results_b/log_base_vs_instruct.txt`
**Plots:** `plots_b/base_vs_instruct.png`, `plots_b/base_vs_instruct_zoomed.png`, `plots_b/shared_pairs.csv`

### Step 4: Geometric metrics (cosine sim + alternates)

```bash
python topic_b_metrics.py
```

**Results:** `results_b/metrics_analysis.csv`, `results_b/metrics_analysis.json`
**Log:** `results_b/log_metrics.txt`
**Plots:** `plots_b/metrics_owls.png`, `plots_b/metrics_eagles.png`, `plots_b/entangled_vs_random_owls.png`, `plots_b/entangled_vs_random_eagles.png`

### Step 5: Synthesis

No additional script — analysis is based on Steps 2–4 results (see README.md).

---

## File Structure

```
takehome_20260128/
├── README.md                    # Main deliverable with all results and analysis
├── reproduce_results.md         # This file
├── topic_a.py                   # Starter code (Topic A)
├── run_topic_a.py               # Parameterized experiment runner (Topic A)
├── sweep_topic_a.sh             # Runs all Topic A sweeps
├── topic_b_part1.py             # Starter code: owl entanglement demo
├── topic_b_part2.py             # Starter code: reverse direction + trees
├── topic_b_part3.py             # Starter code: dot product / cosine analysis
├── topic_b_utils.py             # Shared utilities for Topic B
├── topic_b_expanded.py          # Step 2: 23-animal expanded analysis
├── topic_b_base_vs_instruct.py  # Step 3: base vs instruct comparison
├── topic_b_metrics.py           # Step 4: geometric metrics evaluation
├── results_a/                   # Topic A JSON results (one per experiment)
├── results_b/                   # Topic B CSV/JSON results + run logs
├── plots_a/                     # Topic A plots (one per experiment)
└── plots_b/                     # Topic B plots
```
