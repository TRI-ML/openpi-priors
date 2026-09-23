"""Pi0 with Z-Prior: frozen pretrained action head generates learned prior.

Two variants:
- two_VLM: Two complete Pi0 copies (frozen + trainable). Simple, more memory.
- one_VLM: Shared VLM, two action experts (frozen + trainable). Complex, less memory.

Both use the same principle: frozen action expert generates A_prior (prior_steps denoising
steps from noise), trainable action expert refines from A_prior (refine_steps steps).

Two noise variants:
- A:    Pure prior. Trainable model denoises starting from A_prior.
- A+Z:  Noise added to prior. Starting point = A_prior + c * noise.
"""

import dataclasses
import logging
from typing import Literal

import einops

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0_config
from openpi.models.pi0 import Pi0, make_attn_mask
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils

logger = logging.getLogger("openpi")

NoiseVariant = Literal["A", "A+Z"]
ZPriorMode = Literal["two_VLM", "one_VLM"]


@dataclasses.dataclass(frozen=True)
class Pi0ZPriorConfig(_model.BaseModelConfig):
    """Config for Pi0 with Z-Prior."""

    action_dim: int = -1
    action_horizon: int = -1
    max_token_len: int = -1

    base: pi0_config.Pi0Config = dataclasses.field(default_factory=pi0_config.Pi0Config)
    mode: ZPriorMode = "two_VLM"
    noise_variant: NoiseVariant = "A"
    noise_strength: float = 0.0
    prior_steps: int = 5
    refine_steps: int = 5

    def __post_init__(self):
        object.__setattr__(self, "action_dim", self.base.action_dim)
        object.__setattr__(self, "action_horizon", self.base.action_horizon)
        object.__setattr__(self, "max_token_len", self.base.max_token_len)

    @property
    def discrete_state_input(self) -> bool:
        return self.base.discrete_state_input

    @property
    @override
    def model_type(self) -> _model.ModelType:
        return self.base.model_type

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0ZPriorTwoVLM | Pi0ZPriorOneVLM":
        if self.mode == "two_VLM":
            return Pi0ZPriorTwoVLM(self, rngs=nnx.Rngs(rng))
        else:
            return Pi0ZPriorOneVLM(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        return self.base.inputs_spec(batch_size=batch_size)

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """Freeze all params under frozen_model/ (two_VLM) or frozen_action_expert/ (one_VLM)."""
        if self.mode == "two_VLM":
            return nnx_utils.PathRegex("frozen_model/.*")
        else:
            return nnx_utils.PathRegex("frozen_action_expert/.*")


class _ZPriorBase(_model.BaseModel):
    """Shared logic for both Z-prior variants."""

    noise_variant: str
    noise_strength: float
    prior_steps: int
    refine_steps: int

    def _apply_noise_variant(self, rng, a_prior, actions_shape):
        """For A+Z variant, add noise to the prior."""
        if self.noise_variant == "A+Z":
            noise = jax.random.normal(rng, actions_shape)
            return a_prior + self.noise_strength * noise
        return a_prior

    def _generate_prior_unrolled(self, frozen_model, rng, observation, actions_shape, num_steps, shared_prefix=None):
        """Generate prior using a Python for-loop instead of jax.lax.while_loop.

        Each denoising step is JIT'd individually (small graph), avoiding the OOM from
        compiling the entire while_loop (N steps x 18 layers) into one massive graph.
        """
        observation = _model.preprocess_observation(None, observation, train=False)
        batch_size = observation.state.shape[0]
        dt = -1.0 / num_steps

        # Start from noise
        x_t = jax.random.normal(rng, actions_shape)

        # Compute prefix + KV cache once
        if shared_prefix is not None:
            prefix_tokens, prefix_mask, prefix_ar_mask, kv_cache = shared_prefix
        else:
            prefix_tokens, prefix_mask, prefix_ar_mask = frozen_model.embed_prefix(observation)
            prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
            positions = jnp.cumsum(prefix_mask, axis=1) - 1
            _, kv_cache = frozen_model.PaliGemma.llm(
                [prefix_tokens, None], mask=prefix_attn_mask, positions=positions
            )

        # Unrolled denoising loop — each step is a separate small JIT call.
        # Use JAX arrays for time/dt so XLA compiles ONE kernel reused across iterations
        # (Python floats would create different constants → different compiled kernels → cache bloat).
        time = jnp.array(1.0)
        dt_jax = jnp.array(dt)
        for _ in range(num_steps):
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = frozen_model.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            pam = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            full_attn_mask = jnp.concatenate([pam, suffix_attn_mask], axis=-1)
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (_, suffix_out), _ = frozen_model.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            v_t = frozen_model.action_out_proj(suffix_out[:, -frozen_model.action_horizon:])
            x_t = x_t + dt_jax * v_t
            time = time + dt_jax

        return x_t

    def generate_prior(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions
    ) -> _model.Actions:
        """Generate A_prior from the frozen model. Call OUTSIDE jax.jit to avoid compilation OOM."""
        raise NotImplementedError("Subclasses must implement generate_prior")


class Pi0ZPriorTwoVLM(_ZPriorBase):
    """Two complete Pi0 models: one frozen (prior), one trainable (policy)."""

    def __init__(self, config: Pi0ZPriorConfig, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.noise_variant = config.noise_variant
        self.noise_strength = config.noise_strength
        self.prior_steps = config.prior_steps
        self.refine_steps = config.refine_steps

        rng_key = rngs.default()
        rng1, rng2 = jax.random.split(rng_key)
        self.frozen_model = Pi0(config.base, rngs=nnx.Rngs(rng1))
        self.trainable_model = Pi0(config.base, rngs=nnx.Rngs(rng2))

    @override
    def generate_prior(self, rng, observation, actions):
        """Generate A_prior from frozen model. Uses unrolled loop to avoid compilation OOM."""
        prior_rng, variant_rng = jax.random.split(rng)
        a_prior = self._generate_prior_unrolled(
            self.frozen_model, prior_rng, observation, actions.shape, self.prior_steps
        )
        return self._apply_noise_variant(variant_rng, a_prior, actions.shape)

    @override
    def compute_loss(self, rng, observation, actions, *, train=False, prior=None):
        if prior is None:
            # Fallback: generate prior inline (may OOM during JIT for large models).
            # Prefer calling generate_prior() outside JIT and passing result here.
            preprocess_rng, prior_rng, variant_rng, train_rng = jax.random.split(rng, 4)
            observation = _model.preprocess_observation(preprocess_rng, observation, train=train)
            a_prior = jax.lax.stop_gradient(
                self.frozen_model.sample_actions(prior_rng, observation, num_steps=self.prior_steps)
            )
            a_prior = self._apply_noise_variant(variant_rng, a_prior, actions.shape)
            # Pass train=False since observation was already preprocessed with train=train above.
            return self.trainable_model.compute_loss(train_rng, observation, actions, train=False, prior=a_prior)
            # Note: this fallback path is rarely used (prior is normally precomputed outside JIT).
        else:
            # Prior precomputed outside JIT — just use it.
            return self.trainable_model.compute_loss(rng, observation, actions, train=train, prior=prior)

    @override
    def sample_actions(self, rng, observation, **kwargs):
        prior_rng, variant_rng, refine_rng = jax.random.split(rng, 3)
        a_prior = self.frozen_model.sample_actions(prior_rng, observation, num_steps=self.prior_steps)
        a_prior = self._apply_noise_variant(variant_rng, a_prior, a_prior.shape)
        return self.trainable_model.sample_actions(refine_rng, observation, num_steps=self.refine_steps, initial=a_prior)


class Pi0ZPriorOneVLM(_ZPriorBase):
    """Shared VLM + two action experts (frozen + trainable).

    The trainable model owns the VLM. The frozen model has its own LLM (including 2B weights)
    but we only use its action expert (300M) via suffix-only passes with the shared KV cache.
    The frozen 2B weights exist in memory but are never computed through.
    """

    def __init__(self, config: Pi0ZPriorConfig, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.noise_variant = config.noise_variant
        self.noise_strength = config.noise_strength
        self.prior_steps = config.prior_steps
        self.refine_steps = config.refine_steps

        rng_key = rngs.default()
        rng1, rng2 = jax.random.split(rng_key)
        # The trainable model owns the VLM (SigLIP + Gemma 2B + action expert 300M)
        self.trainable_model = Pi0(config.base, rngs=nnx.Rngs(rng1))
        # The frozen action expert is a full Pi0 but we only use its suffix (300M) passes.
        # Its 2B weights exist but are never computed through.
        self.frozen_action_expert = Pi0(config.base, rngs=nnx.Rngs(rng2))

    def _compute_shared_prefix(self, observation):
        """Compute prefix tokens + KV cache using the trainable model's VLM."""
        prefix_tokens, prefix_mask, prefix_ar_mask = self.trainable_model.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.trainable_model.PaliGemma.llm(
            [prefix_tokens, None], mask=prefix_attn_mask, positions=positions
        )
        return prefix_tokens, prefix_mask, prefix_ar_mask, kv_cache

    @override
    def generate_prior(self, rng, observation, actions):
        """Generate A_prior from frozen action expert with shared VLM prefix. Uses unrolled loop."""
        prior_rng, variant_rng = jax.random.split(rng)
        observation = _model.preprocess_observation(None, observation, train=False)
        shared_prefix = self._compute_shared_prefix(observation)
        a_prior = self._generate_prior_unrolled(
            self.frozen_action_expert, prior_rng, observation, actions.shape,
            self.prior_steps, shared_prefix=shared_prefix,
        )
        return self._apply_noise_variant(variant_rng, a_prior, actions.shape)

    @override
    def compute_loss(self, rng, observation, actions, *, train=False, prior=None):
        if prior is not None:
            # Prior precomputed outside JIT — use trainable model's loss directly.
            return self.trainable_model.compute_loss(rng, observation, actions, train=train, prior=prior)

        # Fallback: generate prior inline (may OOM during JIT for large models).
        preprocess_rng, prior_rng, variant_rng, time_rng = jax.random.split(rng, 4)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        # 1. Compute shared prefix from trainable VLM (once)
        prefix_tokens, prefix_mask, prefix_ar_mask = self.trainable_model.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        prefix_positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.trainable_model.PaliGemma.llm(
            [prefix_tokens, None], mask=prefix_attn_mask, positions=prefix_positions
        )
        shared_prefix = (prefix_tokens, prefix_mask, prefix_ar_mask, kv_cache)

        # 2. Generate prior from frozen action expert using shared KV cache (no gradients)
        a_prior = jax.lax.stop_gradient(
            self.frozen_action_expert.sample_actions(
                prior_rng, observation, num_steps=self.prior_steps,
                shared_prefix=shared_prefix,
            )
        )
        a_prior = self._apply_noise_variant(variant_rng, a_prior, actions.shape)

        # 3. Flow-matching loss through trainable model (reuse prefix_tokens, not KV cache)
        batch_shape = actions.shape[:-2]
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * a_prior + (1 - time_expanded) * actions
        u_t = a_prior - actions

        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.trainable_model.embed_suffix(
            observation, x_t, time
        )
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (prefix_out, suffix_out), _ = self.trainable_model.PaliGemma.llm(
            [prefix_tokens, suffix_tokens],
            mask=attn_mask,
            positions=positions,
            adarms_cond=[None, adarms_cond],
        )
        v_t = self.trainable_model.action_out_proj(suffix_out[:, -self.action_horizon:])

        return jnp.mean(jnp.square(v_t - u_t), axis=-1)

    @override
    def sample_actions(self, rng, observation, **kwargs):
        prior_rng, variant_rng, refine_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(None, observation, train=False)

        # Shared prefix from trainable VLM
        shared_prefix = self._compute_shared_prefix(observation)

        # Prior from frozen action expert using shared KV cache
        a_prior = self.frozen_action_expert.sample_actions(
            prior_rng, observation, num_steps=self.prior_steps,
            shared_prefix=shared_prefix,
        )
        a_prior = self._apply_noise_variant(variant_rng, a_prior, a_prior.shape)

        # Refine with trainable model using shared KV cache
        return self.trainable_model.sample_actions(
            refine_rng, observation, num_steps=self.refine_steps,
            initial=a_prior, shared_prefix=shared_prefix,
        )
