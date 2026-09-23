"""See _CONFIGS for the list of available configs."""

import abc
from collections.abc import Sequence
import dataclasses
import difflib
import logging
import pathlib
from typing import Any, Protocol, TypeAlias

import etils.epath as epath
import flax.nnx as nnx
from typing_extensions import override
import tyro

import openpi.models.model as _model
import openpi.models.pi0_config as pi0_config
import openpi.models.pi0_z_prior as pi0_z_prior
import openpi.models.tokenizer as _tokenizer
import openpi.policies.robocasa_policy as robocasa_policy
import openpi.shared.download as _download
import openpi.shared.normalize as _normalize
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders
import openpi.transforms as _transforms

import openpi.groot_utils.groot_openpi_dataset as _groot_openpi_dataset
from robocasa.utils.dataset_registry import DATASET_SOUP_REGISTRY

ModelType: TypeAlias = _model.ModelType
# Work around a tyro issue with using nnx.filterlib.Filter directly.
Filter: TypeAlias = nnx.filterlib.Filter


@dataclasses.dataclass(frozen=True)
class AssetsConfig:
    """Determines the location of assets (e.g., norm stats) that will be used to set up the data pipeline.

    These assets will be replicated inside the checkpoint under the `assets/asset_id` directory.

    This can be used to load assets from a different checkpoint (e.g., base model checkpoint) or some other
    centralized location. For example, to load the norm stats for the Trossen robot from the base model checkpoint
    during fine-tuning, use:

    ```
    AssetsConfig(
        assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
        asset_id="trossen",
    )
    ```
    """

    # Assets directory. If not provided, the config assets_dirs will be used. This is useful to load assets from
    # a different checkpoint (e.g., base model checkpoint) or some other centralized location.
    assets_dir: str | None = None

    # Asset id. If not provided, the repo id will be used. This allows users to reference assets that describe
    # different robot platforms.
    asset_id: str | None = None


@dataclasses.dataclass(frozen=True)
class DataConfig:
    # LeRobot repo id. If None, fake data will be created.
    repo_id: str | None = None
    # Directory within the assets directory containing the data assets.
    asset_id: str | None = None
    # Contains precomputed normalization stats. If None, normalization will not be performed.
    norm_stats: dict[str, _transforms.NormStats] | None = None

    # Used to adopt the inputs from a dataset specific format to a common format
    # which is expected by the data transforms.
    repack_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Data transforms, typically include robot specific transformations. Will be applied
    # before the data is normalized. See `model.Observation` and `model.Actions` to learn about the
    # normalized data.
    data_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Model specific transforms. Will be applied after the data is normalized.
    model_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantile_norm: bool = False

    # Names of keys that will be used by the data loader to generate the action sequence. The length of the
    # sequence is defined by the `action_horizon` field in the model config. This should be adjusted if your
    # LeRobot dataset is using different keys to represent the action.
    action_sequence_keys: Sequence[str] = ("actions",)

    # If true, will use the LeRobot dataset task to define the prompt.
    prompt_from_task: bool = False

    # Only used for RLDS data loader (ie currently only used for DROID).
    rlds_data_dir: str | None = None
    
    # Action dimension for padding (used by Groot datasets)
    action_dim: int | None = None
    
    # Multi-dataset support for Groot datasets
    data_dirs: list[str] | None = None  # List of data directories for multi-dataset
    dataset_weights: list[float] | None = None  # Weights for each dataset in multi-dataset


class GroupFactory(Protocol):
    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        """Create a group."""


@dataclasses.dataclass(frozen=True)
class ModelTransformFactory(GroupFactory):
    """Creates model transforms for standard pi0 models."""

    # If provided, will determine the default prompt that be used by the model.
    default_prompt: str | None = None

    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        match model_config.model_type:
            case _model.ModelType.PI0:
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                        ),
                    ],
                )
            case _model.ModelType.PI05:
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                            discrete_state_input=model_config.discrete_state_input,
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )


@dataclasses.dataclass(frozen=True)
class DataConfigFactory(abc.ABC):
    # The LeRobot repo id.
    repo_id: str | None = None
    # Determines how the assets will be loaded.
    assets: AssetsConfig = dataclasses.field(default_factory=AssetsConfig)
    # Base config that will be updated by the factory.
    base_config: tyro.conf.Suppress[DataConfig | None] = None

    @abc.abstractmethod
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        """Create a data config."""

    def create_base_config(self, assets_dirs: pathlib.Path) -> DataConfig:
        repo_id = self.repo_id if self.repo_id is not tyro.MISSING else None
        asset_id = self.assets.asset_id or repo_id
        base = self.base_config or DataConfig()
        # Preserve pre-supplied norm_stats; only load if not provided
        existing_stats = base.norm_stats
        loaded_stats = None if existing_stats is not None else self._load_norm_stats(
            epath.Path(self.assets.assets_dir or assets_dirs), asset_id
        )
        return dataclasses.replace(
            base,
            repo_id=repo_id,
            asset_id=asset_id,
            norm_stats=existing_stats if existing_stats is not None else loaded_stats,
        )

    def _load_norm_stats(self, assets_dir: epath.Path, asset_id: str | None) -> dict[str, _transforms.NormStats] | None:
        if asset_id is None:
            return None
        try:
            data_assets_dir = str(assets_dir / asset_id)
            norm_stats = _normalize.load(_download.maybe_download(data_assets_dir))
            logging.info(f"Loaded norm stats from {data_assets_dir}")
            return norm_stats
        except FileNotFoundError:
            logging.info(f"Norm stats not found in {data_assets_dir}.")
            # Fallback: try to read and convert stats from repo meta
            # TODO: fix
            converted = _groot_openpi_dataset._convert_stats_from_repo_meta(asset_id)
            if converted is not None:
                logging.info(f"Converted norm stats from repo meta for {asset_id}")
                return converted
        return None


@dataclasses.dataclass(frozen=True)
class FakeDataConfig(DataConfigFactory):
    repo_id: str = "fake"

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return DataConfig(repo_id=self.repo_id)


@dataclasses.dataclass(frozen=True)
class SimpleDataConfig(DataConfigFactory):
    # Factory for the data transforms.
    data_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=GroupFactory)
    # Factory for the model transforms.
    model_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=ModelTransformFactory)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return dataclasses.replace(
            self.create_base_config(assets_dirs),
            data_transforms=self.data_transforms(model_config),
            model_transforms=self.model_transforms(model_config),
            use_quantile_norm=model_config.model_type == ModelType.PI0_FAST,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotRobocasaDataConfig(DataConfigFactory):
    """Config for training on Groot datasets."""
    
    repo_id: str | None = None
    
    data_dirs: Any | None = None
    dataset_weights: list[float] | None = None
    
    action_dim: int | None = None
    
    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group()

        data_transforms = _transforms.Group(
            inputs=[robocasa_policy.RobocasaInputs(action_dim=model_config.action_dim, model_type=model_config.model_type)],
            outputs=[robocasa_policy.RobocasaOutputs()],
        )

        model_transforms = ModelTransformFactory()(model_config)

        base = self.create_base_config(assets_dirs)

        # Fallback: if norm_stats not found via assets/repo meta, combine from all data_dirs
        fallback_norm_stats = None
        if base.norm_stats is None and self.data_dirs and len(self.data_dirs) > 0:
            if len(self.data_dirs) == 1:
                d = self.data_dirs[0]
                norm_stats = _groot_openpi_dataset._load_norm_stats_from_groot_dataset(d)
                if norm_stats is not None:
                    fallback_norm_stats = norm_stats
                    logging.info(f"Loaded norm stats from local data dir: {d}")
            else:
                norm_stats = _groot_openpi_dataset._load_norm_stats_from_groot_mixture_dataset(self.data_dirs)
                if norm_stats is not None:
                    fallback_norm_stats = norm_stats
                    logging.info(f"Loaded combined norm stats from {len(self.data_dirs)} data dirs")
 
        return dataclasses.replace(
            base,
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_dim=model_config.action_dim,
            data_dirs=self.data_dirs,
            dataset_weights=self.dataset_weights,
            norm_stats=base.norm_stats or fallback_norm_stats,
        )


@dataclasses.dataclass
class TrainConfig:
    # Name of the config. Must be unique. Will be used to reference this config.
    name: tyro.conf.Suppress[str]
    # Project name.
    project_name: str = "openpi"
    # Experiment name. Will be used to name the metadata and checkpoint directories.
    exp_name: str = tyro.MISSING

    # Defines the model config. Some attributes (action_dim, action_horizon, and max_token_len) are shared by all models
    # -- see BaseModelConfig. Specific model implementations (e.g., Pi0Config) inherit from BaseModelConfig and may
    # define additional attributes.
    model: _model.BaseModelConfig = dataclasses.field(default_factory=pi0_config.Pi0Config)

    # A weight loader can optionally load (possibly partial) weights from disk after the model is initialized.
    weight_loader: weight_loaders.WeightLoader = dataclasses.field(default_factory=weight_loaders.NoOpWeightLoader)

    lr_schedule: _optimizer.LRScheduleConfig = dataclasses.field(default_factory=_optimizer.CosineDecaySchedule)
    optimizer: _optimizer.OptimizerConfig = dataclasses.field(default_factory=_optimizer.AdamW)
    ema_decay: float | None = 0.99

    # Specifies which weights should be frozen.
    freeze_filter: tyro.conf.Suppress[Filter] = dataclasses.field(default_factory=nnx.Nothing)

    # Determines the data to be trained on.
    data: DataConfigFactory = dataclasses.field(default_factory=FakeDataConfig)

    # Base directory for config assets (e.g., norm stats).
    assets_base_dir: str = "./assets"
    # Base directory for checkpoints.
    checkpoint_base_dir: str = "./checkpoints"

    # Random seed that will be used by random generators during training.
    seed: int = 42
    # Global batch size.
    batch_size: int = 32
    # Number of workers to use for the data loader. Increasing this number will speed up data loading but
    # will increase memory and CPU usage.
    num_workers: int = 2
    # Number of train steps (batches) to run.
    num_train_steps: int = 30_000

    # How often (in steps) to log training metrics.
    log_interval: int = 100
    # How often (in steps) to save checkpoints.
    save_interval: int = 1000
    # If set, any existing checkpoints matching step % keep_period == 0 will not be deleted.
    keep_period: int | None = 5000

    # If true, will overwrite the checkpoint directory if it already exists.
    overwrite: bool = False
    # If true, will resume training from the last checkpoint.
    resume: bool = False

    # If true, will enable wandb logging.
    wandb_enabled: bool = True

    # Used to pass metadata to the policy server.
    policy_metadata: dict[str, Any] | None = None

    # If the value is greater than 1, FSDP will be enabled and shard across number of specified devices; overall
    # device memory will be reduced but training could potentially be slower.
    # eg. if total device is 4 and fsdp devices is 2; then the model will shard to 2 devices and run
    # data parallel between 2 groups of devices.
    fsdp_devices: int = 1

    @property
    def assets_dirs(self) -> pathlib.Path:
        """Get the assets directory for this config."""
        return (pathlib.Path(self.assets_base_dir) / self.name).resolve()

    @property
    def checkpoint_dir(self) -> pathlib.Path:
        """Get the checkpoint directory for this config."""
        if not self.exp_name:
            raise ValueError("--exp_name must be set")
        return (pathlib.Path(self.checkpoint_base_dir) / self.name / self.exp_name).resolve()

    @property
    def trainable_filter(self) -> nnx.filterlib.Filter:
        """Get the filter for the trainable parameters."""
        return nnx.All(nnx.Param, nnx.Not(self.freeze_filter))

    def __post_init__(self) -> None:
        if self.resume and self.overwrite:
            raise ValueError("Cannot resume and overwrite at the same time.")


# Use `get_config` if you need to get a config by name in your code.
_CONFIGS = [
    #
    # Debugging configs.
    #
    TrainConfig(
        name="debug",
        data=FakeDataConfig(),
        batch_size=2,
        model=pi0_config.Pi0Config(paligemma_variant="dummy", action_expert_variant="dummy"),
        save_interval=100,
        overwrite=True,
        exp_name="debug",
        num_train_steps=10,
        wandb_enabled=False,
    ),
    #
    # RoboCasa dataset configs.
    #
    TrainConfig(
        name="pi0_robocasa_smoke_test",
        model=pi0_config.Pi0Config(
            max_token_len=96,
        ),
        data=LeRobotRobocasaDataConfig(
            data_dirs=[DATASET_SOUP_REGISTRY["target_atomic_seen"][0]],  # CloseBlenderLid only
        ),
	    weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=2,
        save_interval=2,
        keep_period=10000,
        batch_size=3,
        fsdp_devices=3,
        num_workers=1,
    ),
    TrainConfig(
        name="pi0_robocasa_target50",
        model=pi0_config.Pi0Config(
            max_token_len=96,
        ),
        data=LeRobotRobocasaDataConfig(
            data_dirs=DATASET_SOUP_REGISTRY["target50"],
        ),
	    weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=500000,
        save_interval=5000,
        keep_period=10000,
        batch_size=64,
        num_workers=4,
    ),
    TrainConfig(
        name="pi0_robocasa_target_atomic_seen",
        model=pi0_config.Pi0Config(
            max_token_len=96,
        ),
        data=LeRobotRobocasaDataConfig(
            data_dirs=DATASET_SOUP_REGISTRY["target_atomic_seen"],
        ),
	    weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=100000,
        save_interval=5000,
        keep_period=10000,
        batch_size=64,
        num_workers=4,
    ),
    TrainConfig(
        name="pi0_robocasa_target_composite_seen",
        model=pi0_config.Pi0Config(
            max_token_len=96,
        ),
        data=LeRobotRobocasaDataConfig(
            data_dirs=DATASET_SOUP_REGISTRY["target_composite_seen"],
        ),
	    weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=100000,
        save_interval=5000,
        keep_period=10000,
        batch_size=64,
        num_workers=4,
    ),
    TrainConfig(
        name="pi0_robocasa_target_composite_unseen",
        model=pi0_config.Pi0Config(
            max_token_len=96,
        ),
        data=LeRobotRobocasaDataConfig(
            data_dirs=DATASET_SOUP_REGISTRY["target_composite_unseen"],
        ),
	    weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=100000,
        save_interval=5000,
        keep_period=10000,
        batch_size=64,
        num_workers=4,
    ),
    #
    # Pi0.5 RoboCasa configs.
    #
    TrainConfig(
        name="pi05_robocasa_smoke_test",
        model=pi0_config.Pi0Config(
            pi05=True,
            max_token_len=200,
        ),
        data=LeRobotRobocasaDataConfig(
            data_dirs=[DATASET_SOUP_REGISTRY["target_atomic_seen"][0]],
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=2,
        save_interval=2,
        keep_period=10000,
        batch_size=3,
        fsdp_devices=3,
        num_workers=1,
    ),
    TrainConfig(
        name="pi05_robocasa_target_atomic_seen",
        model=pi0_config.Pi0Config(
            pi05=True,
            max_token_len=200,
        ),
        data=LeRobotRobocasaDataConfig(
            data_dirs=DATASET_SOUP_REGISTRY["target_atomic_seen"],
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=100000,
        save_interval=5000,
        keep_period=10000,
        batch_size=64,
        num_workers=4,
    ),
    #
    # Single-task RoboCasa configs.
    #
    # All 18 atomic seen tasks from DATASET_SOUP_REGISTRY["target_atomic_seen"].
    # 4k steps, checkpoints saved at 2000 (50%) and 3999 (100%) for eval.
    #
    # --- Pi0 single-task (all 18 atomic seen) ---
    *[
        TrainConfig(
            name=f"pi0_robocasa_single_{task}",
            model=pi0_config.Pi0Config(max_token_len=96),
            data=LeRobotRobocasaDataConfig(
                data_dirs=[d for d in DATASET_SOUP_REGISTRY["target_atomic_seen"] if d["task"] == task],
            ),
            weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
            lr_schedule=_optimizer.CosineDecaySchedule(warmup_steps=100, peak_lr=2.5e-5, decay_steps=4000, decay_lr=2.5e-6),
            num_train_steps=4000,
            save_interval=2000,
            keep_period=2000,
            batch_size=64,
            num_workers=4,
        )
        for task in sorted(set(d["task"] for d in DATASET_SOUP_REGISTRY["target_atomic_seen"]))
    ],
    # --- Pi0.5 single-task (all 18 atomic seen) ---
    *[
        TrainConfig(
            name=f"pi05_robocasa_single_{task}",
            model=pi0_config.Pi0Config(pi05=True, max_token_len=200),
            data=LeRobotRobocasaDataConfig(
                data_dirs=[d for d in DATASET_SOUP_REGISTRY["target_atomic_seen"] if d["task"] == task],
            ),
            weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
            lr_schedule=_optimizer.CosineDecaySchedule(warmup_steps=100, peak_lr=2.5e-5, decay_steps=4000, decay_lr=2.5e-6),
            num_train_steps=4000,
            save_interval=2000,
            keep_period=2000,
            batch_size=64,
            num_workers=4,
        )
        for task in sorted(set(d["task"] for d in DATASET_SOUP_REGISTRY["target_atomic_seen"]))
    ],
    #
    # Z-Prior configs (pi05 only, one_VLM only).
    # 4k steps, checkpoint at 2000 (50%) and 3999 (100%).
    # prior_steps=5, refine_steps=5 (total 10, same as vanilla inference).
    #
    *[
        TrainConfig(
            name=f"pi05_robocasa_single_{task}_zprior_{variant_name}_one_VLM",
            model=pi0_z_prior.Pi0ZPriorConfig(
                base=pi0_config.Pi0Config(pi05=True, max_token_len=200),
                mode="one_VLM",
                noise_variant=variant,
                noise_strength=1.0 if variant != "A" else 0.0,
            ),
            data=LeRobotRobocasaDataConfig(
                data_dirs=[d for d in DATASET_SOUP_REGISTRY["target_atomic_seen"] if d["task"] == task],
            ),
            weight_loader=weight_loaders.ZPriorWeightLoader(
                "gs://openpi-assets/checkpoints/pi05_base/params"
            ),
            freeze_filter=pi0_z_prior.Pi0ZPriorConfig(
                base=pi0_config.Pi0Config(pi05=True, max_token_len=200),
                mode="one_VLM",
            ).get_freeze_filter(),
            lr_schedule=_optimizer.CosineDecaySchedule(
                warmup_steps=100, peak_lr=2.5e-5, decay_steps=4000, decay_lr=2.5e-6,
            ),
            num_train_steps=4000,
            save_interval=2000,
            keep_period=2000,
            batch_size=64,
            num_workers=4,
        )
        for task in sorted(set(d["task"] for d in DATASET_SOUP_REGISTRY["target_atomic_seen"]))
        for variant, variant_name in [("A", "A"), ("A+Z", "A_plus_Z")]
    ],
    #
    # =====================================================================
    # Composite seen single-task configs (16 tasks).
    # Same hyperparams as atomic seen.
    # =====================================================================
    #
    # --- Pi0.5 composite seen vanilla ---
    *[
        TrainConfig(
            name=f"pi05_robocasa_cseen_{task}",
            model=pi0_config.Pi0Config(pi05=True, max_token_len=200),
            data=LeRobotRobocasaDataConfig(
                data_dirs=[d for d in DATASET_SOUP_REGISTRY["target_composite_seen"] if d["task"] == task],
            ),
            weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
            lr_schedule=_optimizer.CosineDecaySchedule(warmup_steps=100, peak_lr=2.5e-5, decay_steps=4000, decay_lr=2.5e-6),
            num_train_steps=4000,
            save_interval=2000,
            keep_period=2000,
            batch_size=64,
            num_workers=4,
        )
        for task in sorted(set(d["task"] for d in DATASET_SOUP_REGISTRY["target_composite_seen"]))
    ],
    # --- Pi0.5 composite seen Z-Prior (A, A+Z) ---
    *[
        TrainConfig(
            name=f"pi05_robocasa_cseen_{task}_zprior_{variant_name}_one_VLM",
            model=pi0_z_prior.Pi0ZPriorConfig(
                base=pi0_config.Pi0Config(pi05=True, max_token_len=200),
                mode="one_VLM",
                noise_variant=variant,
                noise_strength=1.0 if variant != "A" else 0.0,
            ),
            data=LeRobotRobocasaDataConfig(
                data_dirs=[d for d in DATASET_SOUP_REGISTRY["target_composite_seen"] if d["task"] == task],
            ),
            weight_loader=weight_loaders.ZPriorWeightLoader(
                "gs://openpi-assets/checkpoints/pi05_base/params"
            ),
            freeze_filter=pi0_z_prior.Pi0ZPriorConfig(
                base=pi0_config.Pi0Config(pi05=True, max_token_len=200),
                mode="one_VLM",
            ).get_freeze_filter(),
            lr_schedule=_optimizer.CosineDecaySchedule(
                warmup_steps=100, peak_lr=2.5e-5, decay_steps=4000, decay_lr=2.5e-6,
            ),
            num_train_steps=4000,
            save_interval=2000,
            keep_period=2000,
            batch_size=64,
            num_workers=4,
        )
        for task in sorted(set(d["task"] for d in DATASET_SOUP_REGISTRY["target_composite_seen"]))
        for variant, variant_name in [("A", "A"), ("A+Z", "A_plus_Z")]
    ],
]

if len({config.name for config in _CONFIGS}) != len(_CONFIGS):
    raise ValueError("Config names must be unique.")
_CONFIGS_DICT = {config.name: config for config in _CONFIGS}


def cli() -> TrainConfig:
    return tyro.extras.overridable_config_cli({k: (k, v) for k, v in _CONFIGS_DICT.items()})


def get_config(config_name: str) -> TrainConfig:
    """Get a config by name."""
    if config_name not in _CONFIGS_DICT:
        closest = difflib.get_close_matches(config_name, _CONFIGS_DICT.keys(), n=1, cutoff=0.0)
        closest_str = f" Did you mean '{closest[0]}'? " if closest else ""
        raise ValueError(f"Config '{config_name}' not found.{closest_str}")

    return _CONFIGS_DICT[config_name]
