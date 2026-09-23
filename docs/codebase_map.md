# Codebase map

Where to look when changing any part of the pi0 / pi0.5 policy, the RoboCasa data
path, the priors, or the evaluation loop. Line numbers were checked against this
repository at release time and will drift as files are edited; the function and class
names are the stable anchors. For `src/openpi/training/config.py`,
`scripts/serve_policy.py`, and `src/openpi/training/data_loader.py` only names are
given.

Recommended approach for a new variant: subclass `Pi0` (or wrap `Pi0` instances, as
`pi0_z_prior.py` does), override `compute_loss` / `sample_actions`, add a
`BaseModelConfig` subclass that creates it, and add a `TrainConfig` to `_CONFIGS`.
Leave existing configs untouched for A/B comparison.

## 1. Loss / objective

| What | File | Where |
|---|---|---|
| Flow-matching loss | `src/openpi/models/pi0.py` | `Pi0.compute_loss`, lines 200-235 |
| Optional non-Gaussian start point | `src/openpi/models/pi0.py` | `prior` argument line 207; `starting_point = prior if prior is not None else noise` line 214 |
| Timestep distribution | `src/openpi/models/pi0.py` | `time = jax.random.beta(time_rng, 1.5, 1, ...) * 0.999 + 0.001`, line 215 |
| Interpolant and target velocity | `src/openpi/models/pi0.py` | `x_t = time * starting_point + (1 - time) * actions` line 217; `u_t = starting_point - actions` line 218 |
| MSE | `src/openpi/models/pi0.py` | `jnp.mean(jnp.square(v_t - u_t), axis=-1)`, line 235 |
| Loss wrapper (mean over chunk; add auxiliary losses here) | `scripts/train.py` | `loss_fn` inside `train_step`, lines 149-154 |

## 2. Action sampling (inference)

| What | File | Where |
|---|---|---|
| ODE integration loop | `src/openpi/models/pi0.py` | `Pi0.sample_actions`, lines 238-307 |
| Number of steps | `src/openpi/models/pi0.py` | `num_steps: int = 10`, line 243; `dt = -1.0 / num_steps`, line 250 |
| Start from a given prior | `src/openpi/models/pi0.py` | `initial` argument line 244, used at lines 252-253 |
| Reuse an external prefix / KV cache | `src/openpi/models/pi0.py` | `shared_prefix` argument line 245, unpacked at lines 257-259 |
| Euler step | `src/openpi/models/pi0.py` | `return x_t + dt * v_t, time + dt`, line 299 |
| Loop | `src/openpi/models/pi0.py` | `jax.lax.while_loop(cond, step, (noise, 1.0))`, line 306 |
| Batched inference wrapper | `src/openpi/policies/policy.py` | `Policy.infer`, lines 42-64: input transforms -> add batch dim -> `sample_actions` -> output transforms |

## 3. Forward pass, embeddings, attention mask

| What | File | Where |
|---|---|---|
| Layer construction (pi0 vs pi0.5 branches) | `src/openpi/models/pi0.py` | `Pi0.__init__`, lines 71-112; `adarms=config.pi05` line 81; pi0.5 `time_mlp_in/out` lines 105-106; pi0 `state_proj`, `action_time_mlp_in/out` lines 109-111 |
| Prefix embedding (images + language) | `src/openpi/models/pi0.py` | `embed_prefix`, lines 114-147 |
| Suffix embedding (state, noisy actions, timestep) | `src/openpi/models/pi0.py` | `embed_suffix`, lines 148-197; pi0 path lines 163-179, pi0.5 path lines 180-188 |
| Attention mask (prefix bidirectional, suffix blocks) | `src/openpi/models/pi0.py` | `make_attn_mask`, lines 19-45 |
| Sinusoidal position / time embedding | `src/openpi/models/pi0.py` | `posemb_sincos`, line 48 |
| Image preprocessing and train-time augmentation | `src/openpi/models/model.py` | `preprocess_observation`, lines 138-204: resize-with-pad line 160; `RandomCrop(0.95)`, `Resize`, `Rotate((-5, 5))` lines 170-172; `ColorJitter(0.3, 0.4, 0.5)` line 175 |
| Observation container | `src/openpi/models/model.py` | `Observation`, lines 79-136 (`from_dict` line 106) |
| Abstract model interface | `src/openpi/models/model.py` | `BaseModelConfig` lines 206-248; `BaseModel.compute_loss` / `sample_actions` lines 251-272; `restore_params` line 274 |
| Gemma variants (300M expert, 2B VLM) | `src/openpi/models/gemma.py` | `get_config`, lines 58-110 |
| Transformer block with two token streams | `src/openpi/models/gemma.py` | `Block.__call__`, lines 293-343; `Module.__call__` lines 388-410 |
| SigLIP So400m/14 encoder | `src/openpi/models/siglip.py` | `_Module` line 188; factory `Module` line 293 |

## 4. pi0.5 differences

| What | File | Where |
|---|---|---|
| `pi05` flag and derived defaults | `src/openpi/models/pi0_config.py` | `Pi0Config`, lines 15-71; `pi05`, `discrete_state_input` lines 26-27; `max_token_len` default 200 vs 48 and `discrete_state_input = pi05` in `__post_init__` lines 29-33; `model_type` lines 38-40 |
| Discrete state in the prompt | `src/openpi/models/tokenizer.py` | `PaligemmaTokenizer.tokenize(prompt, state)`, lines 18-46; `np.digitize` into 256 bins over [-1, 1] line 23; prompt format `"Task: ..., State: ...;\nAction: "` line 25 |
| Tokenize transform passes state when `discrete_state_input` | `src/openpi/transforms.py` | `TokenizePrompt`, lines 244-259 (state lookup line 255) |
| Pad state/actions after tokenization (pi0.5 only) | `src/openpi/transforms.py` | `PadStatesAndActions`, lines 261-272 |
| pi0.5 transform stack (`TokenizePrompt(discrete_state_input=...)` + `PadStatesAndActions`) | `src/openpi/training/config.py` | `ModelTransformFactory.__call__`, `ModelType.PI05` case |
| adaRMSNorm (scale / shift / gate from the timestep) | `src/openpi/models/gemma.py` | `RMSNorm.__call__`, lines 113-131: plain RMSNorm when `cond is None`, else `modulation -> scale, shift, gate` lines 129-131 |
| Gated residual | `src/openpi/models/gemma.py` | `_gated_residual`, lines 444-450; used in `Block` lines 313 and 335 |
| Timestep MLP feeding adaRMSNorm | `src/openpi/models/pi0.py` | `embed_suffix` pi0.5 branch, `time_mlp_in`/`time_mlp_out` -> `adarms_cond`, lines 184-188; passed as `adarms_cond=[None, adarms_cond]` to the LLM (lines 231, 294) |
| RoboCasa inputs skip state padding for pi0.5 | `src/openpi/policies/robocasa_policy.py` | `RobocasaInputs.__call__`, lines 56-59 |

## 5. RoboCasa data path

```
Groot/LeRobot dataset on disk           GrootOpenpiSingleDataset.__getitem__      RobocasaInputs
 state.* (16 dims, Groot order)   --->   concat -> "observation/state" (16)  --->  pad_to_dim(32)   (pi0)
 action.* (12 dims, Groot order)  --->   concat -> "actions" (H, 12)        --->  pad_to_dim(32)
 video.robot0_agentview_left      --->   "observation/image"                --->  base_0_rgb
 video.robot0_eye_in_hand         --->   "observation/wrist_image"          --->  left_wrist_0_rgb
 (none)                                                                     --->  right_wrist_0_rgb = zeros, mask False
 annotation.human.task_description --->  "prompt"
        |
        v   Normalize (z-score with stats from meta/stats.json)  ->  ResizeImages(224)  ->  TokenizePrompt [+ PadStatesAndActions]
        v   Observation.from_dict  ->  preprocess_observation  ->  Pi0
 output: actions (H, 32) -> Unnormalize -> RobocasaOutputs: actions[:, :12]
```

| What | File | Where |
|---|---|---|
| Single-task dataset adapter | `src/openpi/groot_utils/groot_openpi_dataset.py` | `GrootOpenpiSingleDataset`, lines 48-121; state/action concatenation order in `__getitem__` lines 95-121 |
| Multi-task mixture adapter (weights `len^0.4`) | `src/openpi/groot_utils/groot_openpi_dataset.py` | `GrootOpenpiMultiDataset`, lines 124-244 |
| Groot -> openpi state reorder `[7..13, 0..6, 14, 15]` | `src/openpi/groot_utils/groot_openpi_dataset.py` | `_load_norm_stats_from_groot_dataset`, line 279 (ordering explained in the docstring lines 259-272) |
| Groot -> openpi action reorder `[5..11, 0..4]` | `src/openpi/groot_utils/groot_openpi_dataset.py` | line 308 |
| Dataset selection (`data_dirs` -> single or mixture) | `src/openpi/training/data_loader.py` | `create_torch_dataset` (Groot branch), `create_data_loader`, `TorchDataLoader`, `DataLoaderImpl` |
| RoboCasa data config (transforms + norm-stat fallback) | `src/openpi/training/config.py` | `LeRobotRobocasaDataConfig.create`; task lists come from `robocasa.utils.dataset_registry.DATASET_SOUP_REGISTRY` |
| Input transform (state pad, image parse, mask, action pad) | `src/openpi/policies/robocasa_policy.py` | `RobocasaInputs`, lines 30-104; `_parse_image` (float->uint8, CHW->HWC) lines 20-26; right-wrist zero image + `False` mask lines 80-86; action padding lines 92-96 |
| Output transform (slice to 12 dims) | `src/openpi/policies/robocasa_policy.py` | `RobocasaOutputs`, lines 108-121 |
| Generic padding helper | `src/openpi/transforms.py` | `pad_to_dim`, line 414 |

## 6. Normalization

| What | File | Where |
|---|---|---|
| Stats container (`mean`, `std`, optional `q01`/`q99`) | `src/openpi/shared/normalize.py` | `NormStats`, lines 10-14; `save` / `load` (JSON) lines 135-147 |
| Derive stats from Groot `meta/stats.json`, reorder, pad to 32 (mean 0 / std 1 for padded dims) | `src/openpi/groot_utils/groot_openpi_dataset.py` | `_load_norm_stats_from_groot_dataset`, lines 247-320 |
| Merge stats across tasks (weighted mean / variance) | `src/openpi/groot_utils/groot_openpi_dataset.py` | `compute_overall_statistics` lines 322-389; `_load_norm_stats_from_groot_mixture_dataset` lines 392-400 |
| Fallback that calls the above when no `assets/` stats exist | `src/openpi/training/config.py` | `LeRobotRobocasaDataConfig.create` |
| z-score transform `(x - mean) / (std + 1e-6)` | `src/openpi/transforms.py` | `Normalize`, lines 115-147 (`_normalize` line 138); `Unnormalize` lines 148-180 |
| Save stats into the checkpoint | `src/openpi/training/checkpoints.py` | `save_state` -> `save_assets`, lines 77-85 (writes `<step>/assets/`) |
| Load stats at inference | `src/openpi/policies/policy_config.py` | `create_trained_policy`, `_normalize.load(checkpoint_dir / "assets")` line 69 |

## 7. Training loop, optimizer, checkpointing

| What | File | Where |
|---|---|---|
| Entry point / CLI | `scripts/train.py` | `main`, lines 200-352; `_config.cli()` (tyro) at the bottom |
| Train state init, weight loading, FSDP sharding | `scripts/train.py` | `init_train_state`, lines 85-134; frozen params cast to bfloat16 line 104; EMA over trainable params only line 114 |
| One optimisation step | `scripts/train.py` | `train_step`, lines 138-197; grads filtered by `trainable_filter` line 160; EMA update `decay * old + (1 - decay) * new` lines 172-181 |
| Prior-model detection | `scripts/train.py` | `_is_z_prior = isinstance(config.model, Pi0ZPriorConfig)`, before the jitted step is built |
| Jitted step with/without a `prior` input | `scripts/train.py` | lines 263-275 |
| Per-step `A_pre` generation (outside JIT) | `scripts/train.py` | `nnx.update(_prior_model, ...)` then `generate_prior`, lines 300-310 |
| Logging (`loss`, `grad_norm`, `param_norm`, prior distances) | `scripts/train.py` | lines 330-343 |
| Checkpoint save cadence | `scripts/train.py` | line 347 (`save_interval` or final step) |
| Cosine LR schedule | `src/openpi/training/optimizer.py` | `CosineDecaySchedule`, lines 16-32 (defaults warmup 1000, peak 2.5e-5, decay 30000, end 2.5e-6; single-task configs override) |
| AdamW + global-norm clipping | `src/openpi/training/optimizer.py` | `AdamW`, lines 66-84 (`clip_by_global_norm` line 84) |
| Checkpoint dir init (`overwrite`, `resume`) | `src/openpi/training/checkpoints.py` | `initialize_checkpoint_dir`, lines 22-68 |
| Save `params/` (EMA merged with frozen), `train_state/`, `assets/` | `src/openpi/training/checkpoints.py` | `save_state` lines 71-103; `_split_params` lines 162-172 (`nnx.State.merge(state.params, state.ema_params)` line 167) |
| Restore for `--resume` | `src/openpi/training/checkpoints.py` | `restore_state` lines 106-124; `_merge_params` lines 175-179 |
| Device mesh and FSDP partitioning | `src/openpi/training/sharding.py` | `make_mesh` line 17; `fsdp_sharding` line 48 |
| `TrainConfig` fields (`batch_size`, `fsdp_devices`, `num_train_steps`, `save_interval`, `keep_period`, `ema_decay`, `freeze_filter`, `wandb_enabled`, `overwrite`, `resume`) | `src/openpi/training/config.py` | `TrainConfig`; `checkpoint_dir` property = `checkpoint_base_dir/name/exp_name`; `trainable_filter` property |
| Pretrained weight loaders | `src/openpi/training/weight_loaders.py` | `CheckpointWeightLoader` line 38 (`gs://openpi-assets/checkpoints/<model>/params`), `ZPriorWeightLoader` line 77, `_merge_params` line 116 |

## 8. The prior model (`A_pre`, `A_pre + sigma*Z`)

| What | File | Where |
|---|---|---|
| Prior config (`mode`, `noise_variant`, `noise_strength` = sigma, `prior_steps`, `refine_steps`) | `src/openpi/models/pi0_z_prior.py` | `Pi0ZPriorConfig`, lines 39-83; `get_freeze_filter` lines 78-83 |
| Noise variant `A` (`A_pre`) vs `A+Z` (`A_pre + sigma*Z`) | `src/openpi/models/pi0_z_prior.py` | `_ZPriorBase._apply_noise_variant`, lines 94-99 |
| Unrolled frozen denoising (outside JIT) | `src/openpi/models/pi0_z_prior.py` | `_generate_prior_unrolled`, lines 101-150 |
| Two full copies variant | `src/openpi/models/pi0_z_prior.py` | `Pi0ZPriorTwoVLM`, lines 159-206 |
| Shared-VLM variant (shipped) | `src/openpi/models/pi0_z_prior.py` | `Pi0ZPriorOneVLM`, lines 209-325: `_compute_shared_prefix` 232-241, `generate_prior` 243-253, `compute_loss` 255-305 (inline fallback when `prior is None`), `sample_actions` 307-325 |

## 9. Evaluation path

```
eval_single_task.py / eval_parallel.py
   |-- subprocess: scripts/serve_policy.py --port P policy:checkpoint --policy.config C --policy.dir D
   |        create_trained_policy -> Policy -> WebsocketPolicyServer.serve_forever
   |
   '-- subprocess(es): examples/robocasa/main.py: eval_env(...)
            gym.make("robocasa/<task>", seed=seed+i) -> obs -> WebsocketClientPolicy.infer(element)
            -> action chunk (50 x 12) -> execute replan_steps=5 -> ... -> stats.json + rollout mp4
```

| What | File | Where |
|---|---|---|
| Policy assembly from a checkpoint | `src/openpi/policies/policy_config.py` | `create_trained_policy`, lines 33-128 (`restore_params(checkpoint_dir / "params", bfloat16)` line 59) |
| Server CLI (`--port`, `policy:checkpoint --policy.config --policy.dir`, `--default-prompt`, `--record`) | `scripts/serve_policy.py` | `Args`, `Checkpoint`, `create_policy`, `main` |
| Websocket server (msgpack-numpy, `infer` per message) | `src/openpi/serving/websocket_policy_server.py` | `WebsocketPolicyServer`, `serve_forever` line 34 |
| Client used by the evaluator | `packages/openpi-client/src/openpi_client/websocket_client_policy.py` | `WebsocketClientPolicy.infer` |
| Rollout loop | `examples/robocasa/main.py` | `eval_env`, lines 70-226 |
| Output directory and skip-if-exists | `examples/robocasa/main.py` | `log_path = f"{log_dir}/evals/{split}/{env_name}/{timestamp}"` line 86; skip when a `stats.json` exists lines 88-91 |
| Deterministic seeding per trial | `examples/robocasa/main.py` | `trial_seed = seed + trial_offset + i`; `np.random.seed`, `random.seed`, `gym.make(..., seed=trial_seed)` lines 108-119 |
| Observation dict sent to the policy | `examples/robocasa/main.py` | `element = {"observation/image", "observation/wrist_image", "observation/state", "prompt"}` lines 159-163 |
| Replan interval and action conversion | `examples/robocasa/main.py` | `replan_steps` (5) lines 168-171; `convert_action` line 174 |
| Episode horizon (`1.5 x` task horizon) | `examples/robocasa/main.py` | line 82 |
| 3-camera mosaic video and `stats.json` | `examples/robocasa/main.py` | mosaic lines 178-189; `imageio.mimwrite(..., fps=20)` lines 202-206; `stats.json` lines 221-227 |
| 2-GPU wrapper | `examples/robocasa/eval_single_task.py` | `main`, lines 45-136 (server env `CUDA_VISIBLE_DEVICES=server_gpu`, eval env `CUDA_VISIBLE_DEVICES=eval_gpu`, `MUJOCO_GL=egl`) |
| N-GPU wrapper | `examples/robocasa/eval_parallel.py` | `run_eval_worker` 63-103 (osmesa, `trial_offset`), `merge_eval_results` 106-163, `main` 195-end |
| Per-checkpoint summary | `examples/robocasa/get_eval_stats.py` | `compute_stats` |
| Cross-method table | `examples/robocasa/print_results.py` | `parse_experiment` (suffix -> method), `main` |
| CLD plots | `examples/robocasa/plot_cld.py` | `compare_success_and_get_cld`, `draw_samples_from_beta_posterior`, `plot_violin_cld`, `main` |
