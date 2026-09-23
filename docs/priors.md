# Non-Gaussian priors for pi0 / pi0.5 flow matching

This document describes the three prior distributions implemented in this repository,
how each is used at training and inference time, the hyperparameters that control
them, the priors that were considered and rejected, and the evaluation / statistical
protocol used to compare them.

## 1. Motivation

pi0 and pi0.5 generate a 50-step action chunk by flow matching: a velocity network
`v_theta(x_t, t, obs)` is trained so that Euler integration from `t = 1` to `t = 0`
transports a sample `x_1` of a *source* distribution onto the demonstrated action
chunk `a`. In the released models the source is `N(0, I)`.

```
                 source x_1                                     target a
   Gaussian:     N(0, I)  --------------- transport ---------------> action chunk
                 (far from a: the net must learn the whole map)

   informed:     something already close to a  ---- short hop ----> action chunk
                 (less transport to learn; more capacity for residual detail)
```

A source distribution that is already close to the target should make the transport
easier to learn from a few hundred demonstrations, and give inference a better starting
point for the same number of denoising steps. All three variants share the same loss and
the same 10 total denoising steps at inference; only the start point differs.

```
 shared loss (src/openpi/models/pi0.py, compute_loss):
     t   ~ Beta(1.5, 1) * 0.999 + 0.001
     x_t = t * x_1 + (1 - t) * a
     u_t = x_1 - a
     L   = mean || v_theta(x_t, t, obs) - u_t ||^2
 shared sampler (sample_actions):
     x <- x_1 ; dt = -1 / num_steps ; repeat num_steps: x <- x + dt * v_theta(x, t, obs)
```

## 2. The three priors

Naming used throughout: `Z` = Gaussian noise; `A_pre` = the output of the frozen
*pre*trained action expert; `A_pre + sigma*Z` = `A_pre` plus scaled Gaussian noise. The
code and the config suffixes use the older names `zprior_A` / `A` and
`zprior_A_plus_Z` / `A+Z` for the last two; the tables below show both.

### 2.1 `Z` (Gaussian noise, baseline)

`x_1 ~ N(0, I)`. Config: plain `Pi0Config(pi05=True)`; no suffix.

```
train:   noise ~ N(0,I) -> x_t -> loss
infer:   noise ~ N(0,I) --10 Euler steps (trained model)--> action
```

### 2.2 `A_pre` (frozen learned prior; code name `zprior_A`, `noise_variant="A"`)

A frozen copy of the *pretrained* action expert denoises Gaussian noise for
`prior_steps = 5` steps to produce `A_pre`. The trainable expert then learns the
transport from `A_pre` to the demonstration and, at inference, refines `A_pre` for
`refine_steps = 5` steps. Total 10 steps, the same as the baseline.

```
train (each step):
    obs ----> trainable VLM (SigLIP + Gemma 2B) ----> prefix tokens + KV cache
                                                     |               |
                              +----------------------+               |
                              v                                      v
    N(0,I) --5 steps--> FROZEN action expert --> A_pre       TRAINED action expert
                        (no gradient)              |           ^
                                                   +---> x_t = t*A_pre + (1-t)*a
                                                         loss = ||v - (A_pre - a)||^2
infer:
    N(0,I) --5 steps, frozen expert--> A_pre --5 steps, trained expert--> action
```

"`one_VLM`" architecture (the only mode shipped in configs):

| Component | Trainable? | Role |
|---|---|---|
| SigLIP + Gemma 2B (VLM) | yes | one shared prefix / KV cache, computed once per call |
| Action expert `trainable_model` (Gemma 300M + projections) | yes | learns transport from `A_pre` |
| Action expert `frozen_action_expert` (Gemma 300M + projections) | no (`freeze_filter = PathRegex("frozen_action_expert/.*")`) | generates `A_pre` using the shared KV cache |

`ZPriorWeightLoader` loads the single `pi05_base` checkpoint into both
`trainable_model/*` and `frozen_action_expert/*`. The frozen expert therefore starts
identical to the trainable one; only the trainable one moves.

### 2.3 `A_pre + sigma*Z` (noisy learned prior; code name `zprior_A_plus_Z`, `noise_variant="A+Z"`)

Identical to `A_pre`, but Gaussian noise is added to the frozen expert's output:
`x_1 = A_pre + sigma * N(0, I)`, where `sigma` is the config field `noise_strength`.
The shipped `_zprior_A_plus_Z_one_VLM` configs set `noise_strength = 1.0` (the
`Pi0ZPriorConfig` class default is `0.0`, which is what the `A_pre` configs use). This
trades some of the prior's accuracy for a broader source distribution.

## 3. Hyperparameters and config fields

`A_pre` and `A_pre + sigma*Z` (`src/openpi/models/pi0_z_prior.py`, `Pi0ZPriorConfig`):

| Field | Default in class | Value in shipped configs | Meaning |
|---|---|---|---|
| `base` | `Pi0Config()` | `Pi0Config(pi05=True, max_token_len=200)` | underlying pi0/pi0.5 config |
| `mode` | `"two_VLM"` | `"one_VLM"` | `two_VLM` = two full Pi0 copies; `one_VLM` = shared VLM, two experts |
| `noise_variant` | `"A"` | `"A"` (`A_pre`) or `"A+Z"` (`A_pre + sigma*Z`) | whether noise is added to `A_pre` |
| `noise_strength` | `0.0` | `0.0` (`A_pre`) / `1.0` (`A_pre + sigma*Z`) | sigma, the scale of the added noise |
| `prior_steps` | `5` | `5` | frozen-expert denoising steps |
| `refine_steps` | `5` | `5` | trainable-expert denoising steps at inference |

Shared training hyperparameters (single-task configs):

| Parameter | Value |
|---|---|
| Initialisation | `pi05_base` (all priors; the `A_pre` variants load it twice) |
| Steps / checkpoints | 4000, saved at 2000 and 3999 |
| Batch size | 64 (global) |
| Optimizer | AdamW, `b1=0.9, b2=0.95, weight_decay=1e-10`, grad clip 1.0 |
| LR | cosine, warmup 100, peak `2.5e-5`, end `2.5e-6` over 4000 steps |
| EMA | 0.99 over trainable parameters only |
| Total denoising steps at inference | 10 for every prior |

Per-prior extras:

| | `Z` | `A_pre` (`A`) | `A_pre + sigma*Z` (`A+Z`) |
|---|---|---|---|
| Frozen component | none | 300M expert | 300M expert |
| Extra memory | none | one extra expert's parameters | same |
| Extra compute per train step | none | 5 frozen denoising passes | same |
| Logged distances | -- | `norm(A_pre - a)` | `norm(A_pre - a)`, `norm(Z - a)` |

## 4. Priors considered but not used

| Method | Portable to pi0? | Why not used |
|---|---|---|
| Noisy observation embedding (`O + 0.25 Z`: perturb a single observation vector that the denoiser conditions on) | No | pi0 has no observation bottleneck. Images and language are ~300+ tokens that interact with action tokens through cross-attention in every one of 18 layers; there is no single vector to perturb. Adding noise to raw SigLIP tokens would be a different (and much larger) perturbation. |
| Cocos (Dong et al., ICLR 2025): autoencoder maps observation to an action mean, `x_1 ~ N(alpha * AE(obs), beta^2 I)` | Technically yes | Needs a three-phase pipeline (collect obs/action pairs with a frozen policy, pretrain the AE, then finetune) that the JAX training loop does not have; and a small AE is strictly weaker than the frozen pretrained expert already used by the `A_pre` prior, which produces the same kind of observation-conditioned action estimate with the full model. |
| BRIDGER (RSS 2024): conditional VAE decoder produces `x_1` | Technically yes | Same three-phase requirement; weaker than `A_pre`; adds KL-annealing and "prior hole" instabilities when the observation encoder drifts. |

## 5. Evaluation and statistical protocol

### 5.1 Initial conditions

`examples/robocasa/main.py` creates a fresh environment per trial and seeds every
random source before doing so: `np.random.seed`, `random.seed`, and
`gym.make(..., seed=trial_seed)` with `trial_seed = seed + trial_offset + i`.
`eval_parallel.py` passes a `trial_offset` to each worker so that trial `i` receives
the same seed no matter how trials are split across workers, and forces osmesa
rendering for all workers because mixing EGL and osmesa changed the sampled scene.

Consequences:

| Setting | Same initial conditions across methods? |
|---|---|
| Same machine, same `--seed`, same `--num-trials` | Yes (verified empirically). Trials are paired. |
| Different machines (or different CPU models) | No. Object/fixture placement in RoboCasa retries on `PlacementError`, decided by floating-point-sensitive MuJoCo collision checks; different hardware can take a different number of retries, consume the RNG differently, and end in a different layout. This is inherent to the simulator, not fixable by seeding. |

Even when conditions differ, all methods still draw i.i.d. from the same task
distribution (layouts 1-10, styles 1-10, object instances, placements), which is what
the unpaired test below assumes. Evaluate all methods you intend to compare on one
machine to get the stronger, paired comparison.

### 5.2 Sequential Barnard test and Compact Letter Display

`examples/robocasa/plot_cld.py` (optional dependencies: `matplotlib`, `scipy`,
`sequentialized_barnard_tests`) compares methods pairwise on their binary success
arrays:

```
for each task (and each section average):
   for each pair of methods (m1, m2):
       alpha_pair = (1 - confidence) / num_pairs          # Bonferroni
       shuffle both success arrays                         # see below
       run MirroredStepTest (n_max <= 600) or MirroredLaiTest (n_max > 600)
       significant if the sequential test rejects "same success rate"
   letters = compact_letter_display(significant pairs)     # shared letter = not distinguishable
   violin  = Beta(1 + s, 1 + n - s) posterior of each method's success rate
```

* The test is *sequential*: it consumes trials one at a time and may stop early. Because
  trial order is arbitrary (and block-structured when several tasks are concatenated for
  an average), both arrays are shuffled before testing so the stopping rule sees a
  representative mix. Shuffling does not change the violins, which depend only on
  `(successes, trials)`.
* Bonferroni correction is applied over all pairwise comparisons within one plot.
* The CLD letters are placed above each violin; methods sharing any letter were not
  separated at the corrected significance level.

### 5.3 Results table

`examples/robocasa/print_results.py --dir ./checkpoints` walks all `stats.json` files,
maps config suffixes to method columns (code names `Z`, `A`, `A+Z`, i.e. `Z`, `A_pre`,
`A_pre + sigma*Z`), and prints an
Atomic Seen and a Composite Seen section plus averages. `--min-eps` (default 180) hides
incomplete runs; `--step` restricts to one checkpoint step.

## 6. Implementation notes

| Topic | Detail |
|---|---|
| Prior computed outside JIT | For the `A_pre` models (`pi0_z_prior.py`), `scripts/train.py` calls `model.generate_prior()` eagerly each step and passes the result into the jitted `train_step(..., prior=...)`. Compiling the 5-step frozen denoising loop *inside* the training graph (5 x 18 layers) ran out of memory at compile time. |
| Unrolled Python loop | `_generate_prior_unrolled` uses a Python `for` over `prior_steps`, so each step is a small separately-jitted graph; `time` and `dt` are JAX scalars (not Python floats) so one compiled kernel is reused instead of one per constant. |
| Refreshing the prior model | `ptrain_step` donates the train-state buffers, so the eagerly-used prior model is refreshed with `nnx.update(_prior_model, train_state.params)` every step before generating the prior. |
| `one_VLM` shared prefix | `Pi0ZPriorOneVLM._compute_shared_prefix` runs the trainable VLM once to get prefix tokens + KV cache; both experts consume it via `Pi0.sample_actions(..., shared_prefix=...)`. The frozen `Pi0` copy still owns 2B unused VLM weights in memory but never runs them. |
| Frozen params | Frozen parameters are cast to bfloat16 in `init_train_state` and excluded from the optimizer state via `trainable_filter = All(Param, Not(freeze_filter))`. |
| EMA | `TrainState.ema_params` tracks trainable parameters only. At save time `_split_params` (in `training/checkpoints.py`) merges EMA values over the full parameter tree with `nnx.State.merge(state.params, state.ema_params)`, so `params/` holds EMA weights for trainable parts and the original values for frozen parts. |
| Weight loading | `ZPriorWeightLoader` prefixes every checkpoint key with `frozen_action_expert/` and `trainable_model/`, falling back to random init for keys absent from the checkpoint. |
| `Pi0` API additions | `compute_loss(..., prior=None)` uses `prior` as `x_1` when given; `sample_actions(..., initial=None, shared_prefix=None)` starts from `initial` and can reuse an external prefix/KV cache. |

## 7. Key files

| File | What |
|---|---|
| `src/openpi/models/pi0.py` | `compute_loss(prior=...)`, `sample_actions(initial=..., shared_prefix=...)` |
| `src/openpi/models/pi0_z_prior.py` | `Pi0ZPriorConfig`, `_ZPriorBase._generate_prior_unrolled`, `Pi0ZPriorTwoVLM`, `Pi0ZPriorOneVLM` |
| `src/openpi/training/weight_loaders.py` | `ZPriorWeightLoader` |
| `src/openpi/training/config.py` | `_CONFIGS`: `*_zprior_A_one_VLM` (`A_pre`) and `*_zprior_A_plus_Z_one_VLM` (`A_pre + sigma*Z`) entries and their `freeze_filter` |
| `scripts/train.py` | prior generation outside JIT, EMA of trainable params, distance logging |
| `src/openpi/training/checkpoints.py` | `_split_params`: EMA + frozen merge for `params/` |
| `src/openpi/policies/policy_config.py` | loads norm stats and assembles the `Policy` from a checkpoint |
| `examples/robocasa/main.py` | per-trial seeding, rollout loop, `stats.json` |
| `examples/robocasa/eval_parallel.py` | multi-GPU eval with global trial offsets, osmesa workers, result merge |
| `examples/robocasa/print_results.py` | tasks x methods table |
| `examples/robocasa/plot_cld.py` | Beta-posterior violins, sequential Barnard tests, CLD |
