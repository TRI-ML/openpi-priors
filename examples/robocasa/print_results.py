#!/usr/bin/env python3
"""Print eval success rates organized by task (rows) x method (columns).

Scans <dir>/<config>/<exp>/<step>/evals/<split>/<task>/<timestamp>/stats.json, i.e. the
layout written by scripts/train.py + examples/robocasa/eval_parallel.py (or eval_single_task.py).
The method (prior) and task are parsed from the config name.

Usage:
    python examples/robocasa/print_results.py              # Print results from ./checkpoints/
    python examples/robocasa/print_results.py --dir path/  # Custom directory
    python examples/robocasa/print_results.py --min-eps 50 # Change minimum episode threshold (default: 180)
    python examples/robocasa/print_results.py --step 3999  # Show only this step (default: show all steps)
"""

import argparse
import json
import pathlib
import re


# Task categorization
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
    "DeliverStraw": "Straw", "GetToastedBread": "Toast",
    "KettleBoiling": "KettleBoil", "LoadDishwasher": "LoadDW",
    "PackIdenticalLunches": "PackLunch", "PreSoakPan": "SoakPan",
    "PrepareCoffee": "Coffee", "RinseSinkBasin": "RinseSink",
    "ScrubCuttingBoard": "ScrubBoard", "SearingMeat": "SearMeat",
    "SetUpCuttingStation": "CutStation", "StackBowlsCabinet": "StackBowls",
    "SteamInMicrowave": "SteamMW", "StirVegetables": "StirVeg",
    "StoreLeftoversInBowl": "Leftovers", "WashLettuce": "WashLet",
}


def parse_experiment(config_name, exp_name):
    """Parse config/exp into (model, prior, task) tuple."""
    if config_name.startswith("pi05_"):
        model = "pi0.5"
    elif config_name.startswith("pi0_"):
        model = "pi0"
    else:
        return None

    m = re.match(r"pi0[5]?_robocasa_(?:single|cseen)_([A-Z][A-Za-z]+)", config_name)
    if m is None:
        return None
    task = m.group(1)

    if "zprior_A_plus_Z_one_VLM" in config_name:
        prior = "A+Z"
    elif "zprior_A_plus_Z_two_VLM" in config_name:
        prior = "A+Z"
    elif "zprior_A_one_VLM" in config_name:
        prior = "A"
    elif "zprior_A_two_VLM" in config_name:
        prior = "A"
    elif "zprior" not in config_name:
        prior = "Z"
    else:
        prior = "unknown"

    return model, prior, task


def fmt_cell(sr, n):
    """Format a result cell: '85.0 (170/200)'."""
    successes = int(round(sr * n))
    return f"{sr*100:5.1f} ({successes:>3}/{n})"


def print_section(title, tasks, results, methods, steps, tw, cw):
    """Print one table section (atomic or composite)."""
    if not tasks:
        return

    print(f"\n  {title}")
    print(f"  {'':>{tw}}", end="")
    for step in steps:
        for method in methods:
            print(f"  {f'{method}@{step}':>{cw}}", end="")
    print()
    print("  " + "-" * (tw + (cw + 2) * len(methods) * len(steps)))

    # Per-task rows
    section_vals = {(m, s): [] for m in methods for s in steps}
    for task in tasks:
        short = TASK_SHORT.get(task, task)
        print(f"  {short:>{tw}}", end="")
        for step in steps:
            for method in methods:
                key = (method, task, step)
                if key in results:
                    sr, n = results[key]
                    print(f"  {fmt_cell(sr, n):>{cw}}", end="")
                    section_vals[(method, step)].append(sr)
                else:
                    print(f"  {'--':>{cw}}", end="")
        print()

    # Average row
    print(f"  {'Avg':>{tw}}", end="")
    for step in steps:
        for method in methods:
            vals = section_vals[(method, step)]
            if vals:
                avg = sum(vals) / len(vals) * 100
                print(f"  {f'{avg:5.1f} ({len(vals)}t)':>{cw}}", end="")
            else:
                print(f"  {'--':>{cw}}", end="")
    print()


def main():
    parser = argparse.ArgumentParser(description="Print eval success rates")
    parser.add_argument("--dir", default="./checkpoints", help="Directory to scan for stats.json (default: ./checkpoints)")
    parser.add_argument("--min-eps", type=int, default=180, help="Minimum episodes to include (default: 180)")
    parser.add_argument("--step", type=int, default=None, help="Show only this step (default: show all)")
    args = parser.parse_args()

    base = pathlib.Path(args.dir)
    if not base.exists():
        print(f"Directory {base} does not exist. Train and evaluate a checkpoint first.")
        return

    # Collect results: {(prior, task, step): (success_rate, num_episodes)}
    # Collapse model since everything is pi0.5
    raw_results = {}
    for stats_file in sorted(base.rglob("stats.json")):
        data = json.loads(stats_file.read_text())
        n = data.get("num_episodes", 0)
        if n < args.min_eps:
            continue
        sr = data.get("success_rate", 0.0)

        parts = stats_file.relative_to(base).parts
        try:
            config = parts[0]
            exp = parts[1]
            step = int(parts[2])
        except (IndexError, ValueError):
            continue

        if args.step is not None and step != args.step:
            continue

        parsed = parse_experiment(config, exp)
        if parsed is None:
            continue
        model, prior, task = parsed

        key = (prior, task, step)
        if key not in raw_results or sr > raw_results[key][0]:
            raw_results[key] = (sr, n)

    if not raw_results:
        print(f"No results found with >= {args.min_eps} episodes in {base}/")
        return

    # Discover methods and steps
    methods_order = ["Z", "A", "A+Z"]
    methods_present = [m for m in methods_order if any(k[0] == m for k in raw_results)]
    steps = sorted(set(k[2] for k in raw_results))

    # Split tasks into atomic / composite / unknown
    all_tasks = sorted(set(k[1] for k in raw_results))
    atomic_tasks = sorted(t for t in all_tasks if t in ATOMIC_SEEN)
    composite_tasks = sorted(t for t in all_tasks if t in COMPOSITE_SEEN)
    other_tasks = sorted(t for t in all_tasks if t not in ATOMIC_SEEN and t not in COMPOSITE_SEEN)

    # Column widths
    tw = 12  # task name column
    cw = 16  # data cell column

    print_section("Atomic Seen", atomic_tasks, raw_results, methods_present, steps, tw, cw)
    print_section("Composite Seen", composite_tasks, raw_results, methods_present, steps, tw, cw)
    if other_tasks:
        print_section("Other", other_tasks, raw_results, methods_present, steps, tw, cw)

    # Grand average
    if atomic_tasks and composite_tasks:
        print(f"\n  {'Grand Avg':>{tw}}", end="")
        for step in steps:
            for method in methods_present:
                vals = [raw_results[(method, t, step)][0]
                        for t in all_tasks if (method, t, step) in raw_results]
                if vals:
                    avg = sum(vals) / len(vals) * 100
                    print(f"  {f'{avg:5.1f} ({len(vals)}t)':>{cw}}", end="")
                else:
                    print(f"  {'--':>{cw}}", end="")
        print()

    print()


if __name__ == "__main__":
    main()
