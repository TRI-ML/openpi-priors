#!/usr/bin/env python3
"""Generate CLD (Compact Letter Display) violin plots from eval results.

Reads per-rollout success labels (rollout_results) from
<dir>/<config>/<exp>/<step>/evals/<split>/<task>/<timestamp>/stats.json and generates:
- One violin+CLD plot per task
- One average violin+CLD plot across all tasks

Requires the optional analysis dependencies: pip install -e ".[analysis]"
(matplotlib, scipy, sequentialized-barnard-tests).

Usage:
    python examples/robocasa/plot_cld.py                        # default: ./checkpoints, step 3999
    python examples/robocasa/plot_cld.py --step 2000            # specific step
    python examples/robocasa/plot_cld.py --dir ./checkpoints    # custom results dir
    python examples/robocasa/plot_cld.py --output_dir my_plots  # custom output dir
"""

import argparse
import json
import os
import pathlib
import re

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sequentialized_barnard_tests import Decision, Hypothesis
from sequentialized_barnard_tests.step import MirroredStepTest
from sequentialized_barnard_tests.lai import MirroredLaiTest


# ---------------------------------------------------------------------------
# CLD computation (from plot_cld.py reference)
# ---------------------------------------------------------------------------

def compact_letter_display(significant_pair_list, sorted_model_list):
    num_models = len(sorted_model_list)
    model_to_index = {m: i for i, m in enumerate(sorted_model_list)}
    significant_index_pairs = [
        (model_to_index[m1], model_to_index[m2])
        for m1, m2 in significant_pair_list
    ]

    def remove_redundant_columns(matrix):
        changed = True
        while changed:
            changed = False
            for i in range(len(matrix)):
                for j in range(len(matrix)):
                    if i != j:
                        si = {idx for idx, c in enumerate(matrix[i]) if c}
                        sj = {idx for idx, c in enumerate(matrix[j]) if c}
                        if si.issubset(sj):
                            matrix.pop(i)
                            changed = True
                            break
                if changed:
                    break
        return matrix

    letter_matrix = [["a"] * num_models]
    for m1, m2 in significant_index_pairs:
        while any(col[m1] and col[m2] for col in letter_matrix):
            for ci, col in enumerate(letter_matrix):
                if col[m1] and col[m2]:
                    new_col = col.copy()
                    new_col[m1] = ""
                    col[m2] = ""
                    letter_matrix[ci] = col
                    letter_matrix.append(new_col)
                    letter_matrix = remove_redundant_columns(letter_matrix)
                    break

    def _col_sort_key(col):
        first = next((i for i, c in enumerate(col) if c), len(col))
        last = next((i for i, c in enumerate(reversed(col)) if c), -1)
        last = len(col) - 1 - last if last >= 0 else len(col)
        return (first, last)

    letter_matrix.sort(key=_col_sort_key)
    for idx, col in enumerate(letter_matrix):
        r = chr(ord("a") + idx)
        letter_matrix[idx] = [r if c else "" for c in col]

    result = []
    for mi in range(num_models):
        letters = "".join(
            letter_matrix[ci][mi] for ci in range(len(letter_matrix))
            if letter_matrix[ci][mi]
        )
        result.append(letters)
    return result


def compare_success_and_get_cld(
    model_name_list, success_array_list,
    global_confidence_level=0.95, max_sample_size_per_model=200,
    rng=None, shuffle=False, test_method_name="step",
):
    if rng is None:
        rng = np.random.default_rng(42)
    num_models = len(model_name_list)
    global_alpha = 1 - global_confidence_level
    num_comparisons = max(1, num_models * (num_models - 1) // 2)
    individual_alpha = global_alpha / num_comparisons

    TestClass = MirroredStepTest if test_method_name == "step" else MirroredLaiTest
    test = TestClass(
        alternative=Hypothesis.P0LessThanP1,
        alpha=individual_alpha,
        n_max=max_sample_size_per_model,
    )
    test.reset()

    arrays = {}
    for name, arr in zip(model_name_list, success_array_list):
        a = np.array(arr, dtype=bool).copy()
        if shuffle:
            rng.shuffle(a)
        arrays[name] = a

    comparisons = {}
    for ia in range(num_models):
        for ib in range(ia + 1, num_models):
            ma, mb = model_name_list[ia], model_name_list[ib]
            aa, ab_ = arrays[ma], arrays[mb]
            n = min(len(aa), len(ab_))
            comparisons[(ma, mb)] = test.run_on_sequence(aa[:n], ab_[:n]).decision

    sig_pairs = [k for k, v in comparisons.items() if v != Decision.FailToDecide]
    sorted_models = [
        m for m, _ in sorted(
            arrays.items(), key=lambda kv: np.mean(kv[1]) if len(kv[1]) else 0.0,
            reverse=True,
        )
    ]
    letters = compact_letter_display(sig_pairs, sorted_models)
    return {m: l for m, l in zip(sorted_models, letters)}


def draw_samples_from_beta_posterior(success_array, rng, num_samples=10000):
    n = len(success_array)
    s = int(np.sum(success_array))
    return stats.beta(1 + s, 1 + n - s).rvs(num_samples, random_state=rng)


# ---------------------------------------------------------------------------
# Colors (matching LBM reference)
# ---------------------------------------------------------------------------

METHOD_COLORS = {
    "Z": "#2ca02c",
    "A_LBM": "#e377c2",
    "A_LBMplusZ": "#08519c",
}


# ---------------------------------------------------------------------------
# Violin CLD plot
# ---------------------------------------------------------------------------

def plot_violin_cld(method_names, success_arrays, output_path, title="", rng=None):
    if rng is None:
        rng = np.random.default_rng(42)

    valid = [
        (m, a) for m, a in zip(method_names, success_arrays)
        if a is not None and len(a) > 0
    ]
    if len(valid) < 2:
        print(f"  Skipping {title}: fewer than 2 valid methods")
        return

    names = [m for m, _ in valid]
    arrays = [a for _, a in valid]
    num = len(names)

    n_max = max(len(a) for a in arrays)
    test_method = "lai" if n_max > 600 else "step"
    print(f"    Computing CLD ({test_method}, n_max={n_max}, {num} methods)...")
    cld_dict = compare_success_and_get_cld(
        names, arrays,
        global_confidence_level=0.95,
        max_sample_size_per_model=n_max,
        rng=rng, shuffle=True,
        test_method_name=test_method,
    )
    print(f"    CLD results: { {m: cld_dict.get(m, '?') for m in names} }")

    posteriors = [draw_samples_from_beta_posterior(a, rng) for a in arrays]
    means = [np.mean(p) for p in posteriors]
    colors = [METHOD_COLORS.get(m, "gray") for m in names]

    fig, ax = plt.subplots(figsize=(max(7, 1.8 * num), 6))
    parts = ax.violinplot(
        posteriors, positions=np.arange(num),
        showmeans=True, showmedians=False, showextrema=False, widths=0.75,
    )
    for pc, c in zip(parts["bodies"], colors):
        pc.set_facecolor(c)
        pc.set_alpha(0.6)
    parts["cmeans"].set_color("black")
    parts["cmeans"].set_linewidth(0.8)

    for i, (m, arr, mu) in enumerate(zip(names, arrays, means)):
        letter = cld_dict.get(m, "?")
        n_s, n_t = int(np.sum(arr)), len(arr)
        ax.text(i + 0.15, mu + 0.025, letter, fontsize=14, fontweight="bold")
        ax.text(i, 0.03, f"{n_s}/{n_t}", fontsize=9, ha="center", va="bottom",
                transform=ax.get_xaxis_transform())

    ax.set_xticks(np.arange(num))
    ax.set_xticklabels(names, rotation=25, ha="right", fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Success Rate", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    _save_fig(fig, output_path)


def _save_fig(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"  Saved: {os.path.relpath(path)}")


# ---------------------------------------------------------------------------
# Data loading from the checkpoints/ tree
# ---------------------------------------------------------------------------

# Map from our prior names to plot labels
PRIOR_TO_LABEL = {
    "Z_prior": "Z",
    "A one_VLM": "A_LBM",
    "A+Z one_VLM": "A_LBMplusZ",
}

TASK_SHORT = {
    "CloseBlenderLid": "Blender", "CloseFridge": "Fridge",
    "CloseToasterOvenDoor": "ToasterDoor", "CoffeeSetupMug": "CoffeeMug",
    "NavigateKitchen": "Navigate", "OpenCabinet": "Cabinet",
    "OpenDrawer": "Drawer", "OpenStandMixerHead": "MixerHead",
    "PickPlaceCounterToCabinet": "PP-C2Cab", "PickPlaceCounterToStove": "PP-C2Stv",
    "PickPlaceDrawerToCounter": "PP-D2C", "PickPlaceSinkToCounter": "PP-S2C",
    "PickPlaceToasterToCounter": "PP-T2C", "SlideDishwasherRack": "DishRack",
    "TurnOffStove": "StoveOff", "TurnOnElectricKettle": "Kettle",
    "TurnOnMicrowave": "Microwave", "TurnOnSinkFaucet": "Faucet",
    # Composite seen
    "DeliverStraw": "Straw", "GetToastedBread": "Toast",
    "KettleBoiling": "KettleBoil", "LoadDishwasher": "LoadDW",
    "PackIdenticalLunches": "PackLunch", "PreSoakPan": "SoakPan",
    "PrepareCoffee": "Coffee", "RinseSinkBasin": "RinseSink",
    "ScrubCuttingBoard": "ScrubBoard", "SearingMeat": "SearMeat",
    "SetUpCuttingStation": "CutStation", "StackBowlsCabinet": "StackBowls",
    "SteamInMicrowave": "SteamMW", "StirVegetables": "StirVeg",
    "StoreLeftoversInBowl": "Leftovers", "WashLettuce": "WashLet",
}


def parse_experiment(config_name):
    """Parse config name into (model, prior, task)."""
    if config_name.startswith("pi05_"):
        model = "pi0.5"
    elif config_name.startswith("pi0_"):
        model = "pi0"
    else:
        return None

    # Extract task: pi05_robocasa_{single|cseen}_{Task}[_suffix]
    m = re.match(r"pi0[5]?_robocasa_(?:single|cseen)_([A-Z][A-Za-z]+)", config_name)
    if m is None:
        return None
    task = m.group(1)

    if "zprior_A_plus_Z_one_VLM" in config_name:
        prior = "A+Z one_VLM"
    elif "zprior_A_one_VLM" in config_name:
        prior = "A one_VLM"
    elif "zprior" not in config_name:
        prior = "Z_prior"
    else:
        prior = "unknown"

    return model, prior, task


def load_rollout_results(results_dir, target_step):
    """Load per-rollout binary arrays from stats.json files under results_dir.

    Returns: {task: {prior_label: np.array([0,1,1,0,...])}}
    """
    base = pathlib.Path(results_dir)
    data = {}  # {task: {label: array}}

    for stats_file in sorted(base.rglob("stats.json")):
        parts = stats_file.relative_to(base).parts
        try:
            config = parts[0]
            exp = parts[1]
            step = int(parts[2])
        except (IndexError, ValueError):
            continue

        if step != target_step:
            continue

        parsed = parse_experiment(config)
        if parsed is None:
            continue
        model, prior, task = parsed

        label = PRIOR_TO_LABEL.get(prior)
        if label is None:
            continue

        stats_data = json.loads(stats_file.read_text())
        rollouts = stats_data.get("rollout_results")
        if rollouts is None or len(rollouts) < 10:
            continue

        if task not in data:
            data[task] = {}
        data[task][label] = np.array(rollouts, dtype=bool)

    return data


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="CLD violin plots from eval results")
    parser.add_argument("--dir", default="./checkpoints", help="Results directory to scan for stats.json (default: ./checkpoints)")
    parser.add_argument("--step", type=int, default=3999, help="Checkpoint step to plot (default: 3999)")
    parser.add_argument("--output_dir", default="./checkpoints/CLD_plots", help="Output directory")
    args = parser.parse_args()

    print(f"Loading results from {args.dir} at step {args.step}...")
    data = load_rollout_results(args.dir, args.step)

    if not data:
        print(f"No results found at step {args.step} in {args.dir}/")
        return

    rng = np.random.default_rng(42)
    method_order = ["Z", "A_LBM", "A_LBMplusZ"]

    # Split tasks into atomic / composite sections
    ATOMIC_SEEN = {
        "CloseBlenderLid", "CloseFridge", "CloseToasterOvenDoor", "CoffeeSetupMug",
        "NavigateKitchen", "OpenCabinet", "OpenDrawer", "OpenStandMixerHead",
        "PickPlaceCounterToCabinet", "PickPlaceCounterToStove", "PickPlaceDrawerToCounter",
        "PickPlaceSinkToCounter", "PickPlaceToasterToCounter", "SlideDishwasherRack",
        "TurnOffStove", "TurnOnElectricKettle", "TurnOnMicrowave", "TurnOnSinkFaucet",
    }
    COMPOSITE_SEEN = {
        "DeliverStraw", "GetToastedBread", "KettleBoiling", "LoadDishwasher",
        "PackIdenticalLunches", "PreSoakPan", "PrepareCoffee", "RinseSinkBasin",
        "ScrubCuttingBoard", "SearingMeat", "SetUpCuttingStation", "StackBowlsCabinet",
        "SteamInMicrowave", "StirVegetables", "StoreLeftoversInBowl", "WashLettuce",
    }

    all_tasks = sorted(data.keys())
    atomic_tasks = sorted(t for t in all_tasks if t in ATOMIC_SEEN)
    composite_tasks = sorted(t for t in all_tasks if t in COMPOSITE_SEEN)
    other_tasks = sorted(t for t in all_tasks if t not in ATOMIC_SEEN and t not in COMPOSITE_SEEN)

    sections = []
    if atomic_tasks:
        sections.append(("atomic", atomic_tasks))
    if composite_tasks:
        sections.append(("composite", composite_tasks))
    if other_tasks:
        sections.append(("other", other_tasks))

    for section_name, section_tasks in sections:
        section_dir = os.path.join(args.output_dir, section_name)
        os.makedirs(section_dir, exist_ok=True)
        print(f"\n=== {section_name.upper()} ({len(section_tasks)} tasks) ===")

        # Per-task plots
        for task in section_tasks:
            task_data = data[task]
            methods = [m for m in method_order if m in task_data]
            arrays = [task_data[m] for m in methods]
            short = TASK_SHORT.get(task, task)
            print(f"\n  Task: {task} ({short})")
            plot_violin_cld(
                methods, arrays,
                os.path.join(section_dir, f"{short}_step{args.step}.png"),
                title=f"{short} (step {args.step})",
                rng=rng,
            )

        # Section average (concatenate rollouts, shuffle)
        print(f"\n  {section_name.capitalize()} average across {len(section_tasks)} tasks")
        concat = {}
        for m in method_order:
            parts = [data[t][m] for t in section_tasks if m in data[t]]
            if parts:
                combined = np.concatenate(parts)
                rng.shuffle(combined)
                concat[m] = combined

        methods = [m for m in method_order if m in concat]
        arrays = [concat[m] for m in methods]
        plot_violin_cld(
            methods, arrays,
            os.path.join(section_dir, f"Average_{section_name}_step{args.step}.png"),
            title=f"{section_name.capitalize()} Avg ({len(section_tasks)} tasks, step {args.step})",
            rng=rng,
        )

    # Grand average if both sections exist
    if len(sections) > 1:
        print(f"\n=== GRAND AVERAGE ({len(all_tasks)} tasks) ===")
        concat = {}
        for m in method_order:
            parts = [data[t][m] for t in all_tasks if m in data[t]]
            if parts:
                combined = np.concatenate(parts)
                rng.shuffle(combined)
                concat[m] = combined

        methods = [m for m in method_order if m in concat]
        arrays = [concat[m] for m in methods]
        plot_violin_cld(
            methods, arrays,
            os.path.join(args.output_dir, f"Grand_Average_step{args.step}.png"),
            title=f"Grand Avg ({len(all_tasks)} tasks, step {args.step})",
            rng=rng,
        )

    print(f"\nAll plots saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
