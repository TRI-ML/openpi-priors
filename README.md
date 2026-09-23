# openpi-priors

Finetune Physical Intelligence's **pi0 / pi0.5** flow-matching robot policies on
**RoboCasa** (simulated kitchen manipulation) while choosing the **prior distribution**
the flow starts from. Standard flow matching transports Gaussian noise into an action
chunk; this repository lets you replace that Gaussian with a *learned* prior (the output
of a frozen copy of the pretrained action expert, optionally with added Gaussian noise),
train the policy against it, and evaluate everything locally on identical, seeded
RoboCasa initial conditions.

This is the code release for the paper **"The Gaussian Is Enough: Flow-Matching Priors Do
Not Help When Fine-Tuning Large Behavior Models"** (Xu, Shah, Kress-Gazit, Nishimura, Itkina,
2026). Project page: **https://cxu-tri.github.io/non_gaussian_FT/** · arXiv: **https://arxiv.org/abs/2609.27070**

The code is a stripped-down derivative of
[Physical Intelligence's openpi](https://github.com/Physical-Intelligence/openpi)
(Apache-2.0) with RoboCasa data loading, the prior models, and a local evaluation
harness added. Simulation, datasets, and kitchen assets come from
[RoboCasa](https://robocasa.ai). Everything runs on a local workstation: install,
finetune, evaluate. There is no cloud component.

Design details, hyperparameters, and the statistical protocol are in
[`docs/priors.md`](docs/priors.md); a file-by-file navigation guide is in
[`docs/codebase_map.md`](docs/codebase_map.md).

---

## Requirements

| Item | Requirement |
|---|---|
| OS / Python | Linux x86-64; Python 3.11 (created by `setup.sh` in a conda env named `openpi`). |
| CUDA | A CUDA 12 driver; `jax[cuda12]` wheels are pulled by pip. |
| GPU memory | Full finetuning of the ~3B-parameter model needs FSDP across GPUs when a single GPU has less than roughly 60 GB. Example: 3 x 48 GB GPUs -> `--batch-size 3 --fsdp-devices 3` (batch size must be divisible by the number of FSDP devices). A single GPU with >= 80 GB fits the full model with `--fsdp-devices 1`. |
| GPUs for eval | At least 2: one serves the policy, the others run the simulator. |
| Disk: pretrained weights | ~12 GB each for `pi0_base` and `pi05_base`, cached in `~/.cache/openpi/`. |
| Disk: kitchen assets | Tens of GB (the download step in `setup.sh` estimates ~10 GB; budget more). |
| Disk: datasets | ~600-700 MB per task; the 18 atomic tasks are ~12 GB in total. |
| Disk: checkpoints | Approximately 12 GB (`params/`) + 29 GB (`train_state/`) per saved step. |

---

## Installation

```bash
git clone https://github.com/TRI-ML/openpi-priors.git
cd openpi-priors
bash setup.sh
conda activate openpi
```

`setup.sh` performs, in order:

| Step | What it does | Why it is done this way |
|---|---|---|
| 1 | `conda create -n openpi python=3.11` | JAX 0.5.3 / Flax 0.10.2 pins target Python 3.11. |
| 2 | `pip install -e .` and `pip install -e packages/openpi-client/` | Installs this repo and the websocket client used by the evaluator. |
| 3 | `git clone https://github.com/robocasa/robocasa.git ~/robocasa` then `pip install -e ~/robocasa` | **Gotcha 1:** RoboCasa must be installed from source. The PyPI package is missing asset files needed for rendering. |
| 4 | `pip install git+https://github.com/ARISE-Initiative/robosuite.git@master` | **Gotcha 2:** the PyPI robosuite release lacks functions RoboCasa imports (`cannot import name 'get_elements'`). |
| 5 | `pip install numpy==2.2.5` | **Gotcha 3:** RoboCasa asserts this exact numpy version at import time. It must be pinned *after* openpi is installed, because openpi's dependencies pull a different numpy. |
| 6 | Downloads and extracts every entry of RoboCasa's `DOWNLOAD_ASSET_REGISTRY` | **Gotcha 4:** without the kitchen assets, evaluation fails with `FileNotFoundError: ...empty_kitchen_arena.xml`. Training does not need them. |

If you install manually, keep the order of steps 2 -> 5: the numpy pin must be last.

Pretrained weights are downloaded automatically on first use from
`gs://openpi-assets/checkpoints/pi0_base/params` and
`gs://openpi-assets/checkpoints/pi05_base/params` (anonymous GCS access via
`fsspec[gcs]`, cached under `~/.cache/openpi/`).

## Dataset download

```bash
conda activate openpi
bash download_datasets.sh
```

Interactive menu: pick a task group (`atomic_seen`, `composite_seen`, `composite_unseen`,
`all`) or type comma-separated task names such as `CloseBlenderLid,OpenCabinet`. It calls
RoboCasa's `download_datasets(split=['target'], source=['human'])`; files land in the
RoboCasa datasets directory (`~/robocasa/datasets/v1.0/...` by default, or
`DATASET_BASE_PATH` from `robocasa.macros` if set).

Task names (18 atomic seen, 16 composite seen):

```
atomic_seen     CloseBlenderLid CloseFridge CloseToasterOvenDoor CoffeeSetupMug
                NavigateKitchen OpenCabinet OpenDrawer OpenStandMixerHead
                PickPlaceCounterToCabinet PickPlaceCounterToStove PickPlaceDrawerToCounter
                PickPlaceSinkToCounter PickPlaceToasterToCounter SlideDishwasherRack
                TurnOffStove TurnOnElectricKettle TurnOnMicrowave TurnOnSinkFaucet

composite_seen  DeliverStraw GetToastedBread KettleBoiling LoadDishwasher
                PackIdenticalLunches PreSoakPan PrepareCoffee RinseSinkBasin
                ScrubCuttingBoard SearingMeat SetUpCuttingStation StackBowlsCabinet
                SteamInMicrowave StirVegetables StoreLeftoversInBowl WashLettuce
```

Normalization statistics are **not** computed separately: `LeRobotRobocasaDataConfig`
reads `meta/stats.json` shipped with each Groot dataset, reorders it into openpi's
state/action layout, pads to 32 dims, and saves it into the checkpoint's `assets/`
directory. `scripts/compute_norm_stats.py` is kept for other datasets only.

## Quick start: smoke test

Requires the `CloseBlenderLid` dataset. Trains 2 steps of pi0.5 and writes a checkpoint.

```bash
conda activate openpi
XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 \
python scripts/train.py pi05_robocasa_smoke_test \
    --exp-name=smoke_test --no-wandb-enabled --overwrite
```

The smoke-test configs have `batch_size=3, fsdp_devices=3` baked in (for 3 GPUs). On a
single >= 80 GB GPU add `--batch-size 1 --fsdp-devices 1`. The checkpoint appears at
`checkpoints/pi05_robocasa_smoke_test/smoke_test/1/`.

Then evaluate it with 2 rollouts (needs the kitchen assets and >= 2 GPUs):

```bash
python examples/robocasa/eval_parallel.py \
    --config pi05_robocasa_smoke_test \
    --checkpoint-dir checkpoints/pi05_robocasa_smoke_test/smoke_test/1 \
    --task CloseBlenderLid --num-trials 2 --num-gpus 3
```

Swap `pi05_robocasa_smoke_test` for `pi0_robocasa_smoke_test` to test pi0.

---

## Choosing a prior

Flow matching learns a velocity field that transports a start point `x_1` at `t = 1`
to the action chunk `a` at `t = 0`. All three priors share the same loss; they differ
only in how `x_1` is produced (sigma = 1.0 in the shipped configs):

```
loss:   x_t = t * x_1 + (1 - t) * a          target velocity  u_t = x_1 - a
        L   = || v_theta(x_t, t, obs) - u_t ||^2

                    x_1 (start, t=1)                                          a (t=0)
Z                   N(0, I) ----------------------- 10 Euler steps ----------------> action

A_pre               N(0, I) --5 steps, FROZEN expert--> A_pre --5 steps, TRAINED--> action

A_pre + sigma*Z     N(0, I) --5 steps, FROZEN expert--> A_pre + sigma*N(0, I)
                                                                 --5 steps, TRAINED--> action
```

| Prior | Code name | Config suffix | Needs at training | Needs at inference | Extra cost |
|---|---|---|---|---|---|
| `Z` (Gaussian noise, baseline) | -- | *(none)* | nothing | nothing | none |
| `A_pre` (output of the frozen pretrained action expert) | `zprior_A` (`noise_variant="A"`) | `_zprior_A_one_VLM` | a frozen copy of the pretrained action expert (`ZPriorWeightLoader` loads `pi05_base` into both the frozen and the trainable expert; `freeze_filter` keeps `frozen_action_expert/*` fixed) | the same frozen expert, saved inside the checkpoint | one extra 300M-param expert in memory; 5 extra denoising passes per step |
| `A_pre + sigma*Z` (`A_pre` plus scaled Gaussian noise) | `zprior_A_plus_Z` (`noise_variant="A+Z"`) | `_zprior_A_plus_Z_one_VLM` | same as `A_pre`, `noise_strength=1.0` (= sigma) | same as `A_pre` | same as `A_pre` |

`one_VLM` means the frozen and trainable action experts **share one VLM** (SigLIP +
Gemma 2B): the prefix KV cache is computed once by the trainable VLM and reused by both
experts. See [`docs/priors.md`](docs/priors.md).

## Training

```bash
# Gaussian baseline, one atomic task, 3 x 48 GB GPUs
XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 \
python scripts/train.py pi05_robocasa_single_CloseFridge \
    --exp-name=fridge_Z --batch-size=3 --fsdp-devices=3

# Same task with a non-Gaussian prior: append the suffix to the config name
#   _zprior_A_one_VLM (A_pre) | _zprior_A_plus_Z_one_VLM (A_pre + sigma*Z)
XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 \
python scripts/train.py pi05_robocasa_single_CloseFridge_zprior_A_one_VLM \
    --exp-name=fridge_A_pre --batch-size=3 --fsdp-devices=3

# Multi-task, single >= 80 GB GPU
XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 \
python scripts/train.py pi05_robocasa_target_atomic_seen \
    --exp-name=atomic_v1 --batch-size=16 --fsdp-devices=1 --num-train-steps=100000
```

The CLI is generated by [tyro](https://github.com/brentyi/tyro) from the `TrainConfig`
dataclass. The config name is the first positional argument; every field can be
overridden with `--field-name=value`.

| Flag | Default | Notes |
|---|---|---|
| `--exp-name` | required | Names the run: `checkpoints/<config>/<exp-name>/`. |
| `--batch-size` | config (64 for training configs, 3 for smoke tests) | Global batch. Must be divisible by `jax.device_count()`. |
| `--fsdp-devices` | config (1, or 3 for smoke tests) | Number of GPUs to shard parameters/optimizer state across. |
| `--num-train-steps` | config (4000 single-task, 100000 multi-task) | |
| `--save-interval` | config (2000 single-task) | A checkpoint is also written at the final step. |
| `--log-interval`, `--num-workers` | 100, config (4) | Logging period; torch dataloader workers. |
| `--no-wandb-enabled` | wandb on (project `TrainConfig.project_name`, default `openpi`) | Disable Weights & Biases logging (tyro boolean syntax). |
| `--overwrite` | off | Delete an existing `checkpoints/<config>/<exp-name>/` first. |
| `--resume` | off | Resume from the latest checkpoint in that directory. Mutually exclusive with `--overwrite`. |

Checkpoints land in `checkpoints/<config>/<exp-name>/<step>/`:

```
checkpoints/<config>/<exp-name>/
  <step>/
    params/           EMA-smoothed inference weights (approx. 12 GB)
    train_state/      optimizer + raw params, needed only for --resume (approx. 29 GB)
    assets/           norm_stats.json used for (un)normalisation at inference
```

Set `SKIP_TRAIN_STATE=1` in the environment to skip writing `train_state/` (saves about
30 GB and several minutes per checkpoint). You lose `--resume`, which is usually fine for
short single-task runs.

Training uses full finetuning (no LoRA), AdamW (`b1=0.9, b2=0.95, weight_decay=1e-10`,
gradient clipping at 1.0), a cosine schedule (single-task configs: 100 warmup steps,
peak `2.5e-5`, decaying to `2.5e-6` over 4000 steps), and an EMA of the *trainable*
parameters with decay 0.99. The EMA weights, merged with any frozen parameters, are what
`params/` contains.

## Evaluation

Evaluation runs a policy server (`scripts/serve_policy.py`, websocket) on one GPU and
RoboCasa rollouts (`examples/robocasa/main.py`) on the others. The server and the
simulator **must not share a GPU** (see Troubleshooting).

### Two GPUs: `eval_single_task.py`

```bash
python examples/robocasa/eval_single_task.py \
    --config pi05_robocasa_single_CloseFridge \
    --checkpoint-dir checkpoints/pi05_robocasa_single_CloseFridge/fridge_Z/3999 \
    --env-name CloseFridge --num-trials 50 \
    --server-gpu 0 --eval-gpu 1
```

Flags: `--config`, `--checkpoint-dir`, `--env-name` (required); `--split` (`target`,
default; or `pretrain`), `--num-trials` (default 2), `--server-gpu` (0), `--eval-gpu`
(1), `--port` (8000). Uses EGL rendering on the eval GPU.

### N GPUs: `eval_parallel.py`

```bash
# 3 GPUs: server on GPU 0, two workers on GPUs 1-2, 200 trials split 100/100
python examples/robocasa/eval_parallel.py \
    --config pi05_robocasa_single_CloseFridge \
    --checkpoint-dir checkpoints/pi05_robocasa_single_CloseFridge/fridge_Z/3999 \
    --task CloseFridge --num-trials 200 --num-gpus 3
```

Flags: `--config`, `--checkpoint-dir`, `--task` (required); `--split` (`target`),
`--num-trials` (50), `--num-gpus` (default 8, so pass your real count), `--port`
(8000), `--seed` (7), `--weights-dir` (load weights from a different directory than
the one results are written to). Workers render with osmesa (CPU) so that every worker
produces bit-identical initial conditions; per-worker results are merged into one
`stats.json` and the worker directories are removed.

### Deterministic initial conditions

Trial `i` always seeds `numpy`, `random`, and `gym.make(seed=...)` with
`seed + i`, regardless of how trials are split across workers. Two methods evaluated
on the same machine with the same `--seed` and `--num-trials` therefore see the **same
sequence of kitchen layouts, object instances, and placements**, so success rates can
be compared trial-for-trial. (Determinism does not hold across different machines; see
`docs/priors.md`.)

### Output layout

```
checkpoints/<config>/<exp>/<step>/evals/<split>/<task>/<YYYY-MM-DD-HH-MM>/
  stats.json                       {"num_episodes", "success_rate", "rollout_results": [0,1,...]}
  rollout_<i>_success.mp4          3-camera mosaic (left, right, wrist) per episode
  rollout_<i>_failure.mp4
```

`python examples/robocasa/get_eval_stats.py --dir <step-dir>` prints the per-task
success rates found under one checkpoint step.

### Results table across methods

```bash
python examples/robocasa/print_results.py --dir ./checkpoints --min-eps 180
python examples/robocasa/print_results.py --dir ./checkpoints --step 3999 --min-eps 50
```

Scans every `stats.json` under `--dir`, parses `<config>/<exp>/<step>` from the path,
recognises the prior suffixes, and prints *Atomic Seen* and *Composite Seen* sections
with tasks as rows and `method@step` as columns (`--min-eps`, default 180, drops runs with
fewer episodes). The method columns use the code names `Z`, `A`, `A+Z` for
`Z`, `A_pre`, `A_pre + sigma*Z`. Only `*_single_<task>*` and `*_cseen_<task>*` configs are
recognised; smoke-test and multi-task configs are not listed. Use `--min-eps 1` to see
short test runs.

### CLD violin plots

```bash
pip install -e ".[analysis]"          # matplotlib, scipy, sequentialized_barnard_tests
python examples/robocasa/plot_cld.py --dir ./checkpoints --step 3999 --output_dir ./plots
```

Per task (and per atomic / composite average) this plots a Beta-posterior violin per
method with a Compact Letter Display: methods sharing a letter are not significantly
different under pairwise sequential Barnard tests with Bonferroni correction. Protocol
details are in [`docs/priors.md`](docs/priors.md).

## Config reference

All configs live in `_CONFIGS` in `src/openpi/training/config.py`. `{task}` is any
name from the lists above.

| Config | Model | Data | Steps | Notes |
|---|---|---|---|---|
| `pi0_robocasa_smoke_test` | pi0 | CloseBlenderLid | 2 | `batch_size=3, fsdp_devices=3`, `max_token_len=96` |
| `pi05_robocasa_smoke_test` | pi0.5 | CloseBlenderLid | 2 | same, `max_token_len=200` |
| `pi0_robocasa_target50` | pi0 | all 50 target tasks | 500k | multi-task |
| `pi0_robocasa_target_atomic_seen` | pi0 | 18 atomic seen | 100k | multi-task |
| `pi0_robocasa_target_composite_seen` | pi0 | 16 composite seen | 100k | multi-task |
| `pi0_robocasa_target_composite_unseen` | pi0 | 16 composite unseen | 100k | multi-task |
| `pi05_robocasa_target_atomic_seen` | pi0.5 | 18 atomic seen | 100k | multi-task |
| `pi0_robocasa_single_{task}` | pi0 | one atomic task | 4k | `Z` (Gaussian) prior, ckpt at 2000 and 3999 |
| `pi05_robocasa_single_{task}` | pi0.5 | one atomic task | 4k | `Z` (Gaussian) prior |
| `pi05_robocasa_single_{task}_zprior_A_one_VLM` | pi0.5 | one atomic task | 4k | `A_pre` prior |
| `pi05_robocasa_single_{task}_zprior_A_plus_Z_one_VLM` | pi0.5 | one atomic task | 4k | `A_pre + sigma*Z` prior |
| `pi05_robocasa_cseen_{task}` | pi0.5 | one composite task | 4k | `Z` (Gaussian) prior |
| `pi05_robocasa_cseen_{task}_zprior_A_one_VLM` | pi0.5 | one composite task | 4k | `A_pre` prior |
| `pi05_robocasa_cseen_{task}_zprior_A_plus_Z_one_VLM` | pi0.5 | one composite task | 4k | `A_pre + sigma*Z` prior |

Single-task configs default to `batch_size=64, fsdp_devices=1, num_workers=4`; override
on the command line for your hardware. pi0 uses `max_token_len=96`; pi0.5 uses 200
because the robot state is tokenized into the prompt.

pi0 vs pi0.5 (discrete state tokens, adaRMSNorm, weight paths) is summarised in
[`docs/codebase_map.md`](docs/codebase_map.md), section 4. Only the flow-matching action
head of pi0.5 is available in the open weights.

## Repository layout

```
setup.sh, download_datasets.sh    environment + assets; interactive dataset download
scripts/
  train.py                        training entry point (tyro CLI over TrainConfig)
  serve_policy.py                 websocket policy server used by the evaluators
  compute_norm_stats.py           not needed for RoboCasa (stats come from dataset metadata)
examples/robocasa/
  main.py                         rollout loop; writes stats.json + mp4 per episode
  eval_single_task.py             2-GPU wrapper: server GPU + eval GPU
  eval_parallel.py                N-GPU wrapper: server on GPU 0, workers on the rest
  get_eval_stats.py               summarise evals under one checkpoint step
  print_results.py                tasks x methods table from all stats.json
  plot_cld.py                     CLD violin plots + sequential Barnard tests
src/openpi/
  models/pi0.py, pi0_config.py    pi0 / pi0.5 flow-matching model (accepts prior / initial)
  models/pi0_z_prior.py           A_pre / A_pre + sigma*Z priors (frozen expert generates the start point)
  models/gemma.py, siglip.py, tokenizer.py   backbone (adaRMSNorm), vision encoder, discrete-state prompt
  training/config.py              TrainConfig, LeRobotRobocasaDataConfig, _CONFIGS
  training/weight_loaders.py, checkpoints.py, data_loader.py   weight loaders, EMA+frozen merge, batches
  policies/robocasa_policy.py, policy_config.py   RoboCasa transforms; create_trained_policy
  groot_utils/groot_openpi_dataset.py  Groot/LeRobot dataset adapter + norm-stat derivation
  transforms.py                   Normalize, TokenizePrompt, PadStatesAndActions, ...
packages/openpi-client/           websocket client + image tools used by the evaluator
docs/priors.md, docs/codebase_map.md
```

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `Unrecognized options: False` | tyro booleans are flags: use `--no-wandb-enabled`, not `--wandb-enabled=False`. Likewise `--overwrite` / `--resume`, not `--overwrite=True`. Passing both raises `Cannot resume and overwrite at the same time`. |
| `RESOURCE_EXHAUSTED: Out of memory` / `Batch size N must be divisible by the number of devices` | Shard across GPUs: `--fsdp-devices=<num GPUs>` with `--batch-size` a multiple of the visible GPU count (restrict GPUs with `CUDA_VISIBLE_DEVICES` if needed); keep `XLA_PYTHON_CLIENT_MEM_FRACTION=0.95`. |
| `Failed to initialize BLAS support` during eval | The policy server and the simulator share a GPU. Give them different `CUDA_VISIBLE_DEVICES` (the eval wrappers do this via `--server-gpu/--eval-gpu` or GPU 0 vs. 1..N). |
| `numpy version must be 2.2.5` | `pip install numpy==2.2.5` (re-run after any pip operation that upgraded numpy). |
| `No module named 'robocasa'` / `cannot import name 'get_elements'` | Install RoboCasa from source (`pip install -e ~/robocasa`) and robosuite from `ARISE-Initiative/robosuite@master`. |
| `FileNotFoundError: ...empty_kitchen_arena.xml` | Kitchen assets missing: re-run the asset step in `setup.sh`. |
| Eval prints `stats path exists, skipping` | `main.py` refuses to overwrite: an earlier `stats.json` exists under `<step>/evals/<split>/<task>/`. Remove it (`rm -rf <step>/evals`) to re-run. |
| Training looks hung after `Wrote ... array_metadata` | It is writing the checkpoint. A step dir is 40-50 GB and the log is silent for 15-30 min on an ordinary disk (orbax reports 20-45 MiB/s). Watch the `.orbax-checkpoint-tmp-*` dir grow; set `SKIP_TRAIN_STATE=1` to cut the write to the ~12-17 GB `params/` only. |
| Rendering: EGL vs osmesa | `eval_single_task.py` uses `MUJOCO_GL=egl` on the eval GPU. `eval_parallel.py` forces `MUJOCO_GL=osmesa` (CPU) for all workers because mixing EGL and osmesa across workers changes the sampled initial conditions; osmesa is slower but deterministic. |

## Running the tests

The in-source unit tests need two extra packages that the runtime install does not pull in:

```bash
pip install pytest pynvml
python -m pytest src/openpi/shared src/openpi/transforms_test.py src/openpi/models/tokenizer_test.py -q
```

## Citation

If you use this code, please cite the paper
([project page](https://cxu-tri.github.io/non_gaussian_FT/)):

```bibtex
@article{xu2026gaussian,
  title={The Gaussian Is Enough: Flow-Matching Priors Do Not Help When Fine-Tuning Large Behavior Models},
  author={Xu, Chen and Shah, Rishi and Kress-Gazit, Hadas and Nishimura, Haruki and Itkina, Masha},
  journal={arXiv preprint arXiv:2609.27070},
  year={2026}
}
```

Please also cite [openpi / pi0](https://github.com/Physical-Intelligence/openpi) and
[RoboCasa](https://robocasa.ai) when using the underlying models and benchmark.

## License

Apache-2.0, inherited from Physical Intelligence's openpi. See [`LICENSE`](LICENSE).
