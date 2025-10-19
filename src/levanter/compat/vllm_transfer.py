# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""
Weight transfer utilities for converting Levanter models to vLLM format.

This module provides functionality to transfer trained model weights from Levanter
to vLLM during training for inference/generation. It handles parameter name mappings,
transpose operations, custom transformations, and device placement.

The design is inspired by Tunix's weight transfer mechanism but adapted for Levanter's
architecture using Haliax named arrays and Equinox modules.
"""

import logging
from dataclasses import dataclass
from typing import Callable, Dict, Optional

import jax
import jax.numpy as jnp
from jax._src.partition_spec import PartitionSpec
from jaxtyping import Array

from levanter.compat.hf_checkpoints import ModelWithHfSerializationMixin, _to_state_dict_with_dtype

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VllmMappingConfig:
    """
    Configuration for mapping Levanter model parameters to vLLM format.

    This class encapsulates all the information needed to transfer weights from
    Levanter to vLLM, including parameter name mappings, transpose operations,
    and custom transformation hooks.

    Attributes:
        to_vllm_mappings: Dictionary mapping vLLM parameter names to Levanter parameter names.
            Keys are vLLM names (with * for wildcards), values are Levanter names.
            Example: {"model.layers.*.self_attn.q_proj.weight": "transformer.layers.*.attn.q_proj.weight"}
        to_vllm_transpose_keys: Set of vLLM parameter names that require transposition.
            These parameters will be transposed after name mapping.
        to_vllm_hook_fns: Dictionary of custom transformation functions.
            Keys are vLLM parameter names, values are functions that take an array and return a transformed array.
            Example: {"model.layers.*.mlp.gate_proj.weight": lambda x: x * 0.5}
    """

    to_vllm_mappings: Dict[str, str]
    to_vllm_transpose_keys: set[str]
    to_vllm_hook_fns: Dict[str, Callable[[Array], Array]]

    def __init__(
        self,
        to_vllm_mappings: Optional[Dict[str, str]] = None,
        to_vllm_transpose_keys: Optional[set[str]] = None,
        to_vllm_hook_fns: Optional[Dict[str, Callable[[Array], Array]]] = None,
    ):
        object.__setattr__(self, "to_vllm_mappings", to_vllm_mappings or {})
        object.__setattr__(self, "to_vllm_transpose_keys", to_vllm_transpose_keys or set())
        object.__setattr__(self, "to_vllm_hook_fns", to_vllm_hook_fns or {})


def _match_pattern(pattern: str, key: str) -> Optional[Dict[str, str]]:
    """
    Match a key against a pattern with wildcards (*).

    Args:
        pattern: Pattern string with * wildcards (e.g., "layers.*.attn.*.weight")
        key: Key to match against the pattern

    Returns:
        Dictionary mapping wildcard positions to their matched values, or None if no match.
        Example: pattern="layers.*.attn.*.weight", key="layers.0.attn.q.weight"
                 returns {"0": "0", "1": "q"}
    """
    pattern_parts = pattern.split(".")
    key_parts = key.split(".")

    if len(pattern_parts) != len(key_parts):
        return None

    wildcards = {}
    wildcard_idx = 0

    for p_part, k_part in zip(pattern_parts, key_parts):
        if p_part == "*":
            wildcards[str(wildcard_idx)] = k_part
            wildcard_idx += 1
        elif p_part != k_part:
            return None

    return wildcards


def _apply_pattern(pattern: str, wildcards: Dict[str, str]) -> str:
    """
    Apply wildcard substitutions to a pattern.

    Args:
        pattern: Pattern string with * wildcards
        wildcards: Dictionary mapping wildcard indices to values

    Returns:
        Pattern with wildcards replaced by their values
    """
    result_parts = []
    wildcard_idx = 0

    for part in pattern.split("."):
        if part == "*":
            result_parts.append(wildcards[str(wildcard_idx)])
            wildcard_idx += 1
        else:
            result_parts.append(part)

    return ".".join(result_parts)


def transfer_weights_to_vllm(
    model: ModelWithHfSerializationMixin,
    mapping_config: VllmMappingConfig,
    dtype: Optional[jnp.dtype] = None,
    target_device_mesh: Optional[jax.sharding.Mesh] = None,
) -> Dict[str, Array]:
    """
    Transfer weights from a Levanter model to vLLM format.

    This function converts a Levanter model's weights to a format compatible with vLLM,
    applying parameter name mappings, transpose operations, and custom transformations.

    Args:
        model: Levanter model to transfer weights from
        mapping_config: Configuration specifying how to map parameters
        dtype: Optional dtype to convert floating-point arrays to
        target_device_mesh: Optional JAX mesh for resharding parameters to match vLLM's device placement

    Returns:
        Dictionary mapping vLLM parameter names to JAX arrays

    Example:
        >>> model = Gpt2LMHeadModel.init(Vocab, config, key=key)
        >>> mapping_config = config.vllm_mapping_config()
        >>> vllm_weights = transfer_weights_to_vllm(model, mapping_config)
    """
    logger.info("Starting weight transfer from Levanter to vLLM format")

    levanter_state_dict = _to_state_dict_with_dtype(model, dtype, None)

    vllm_state_dict = {}

    for vllm_pattern, levanter_pattern in mapping_config.to_vllm_mappings.items():
        for lev_key in levanter_state_dict.keys():
            wildcards = _match_pattern(levanter_pattern, lev_key)
            if wildcards is not None:
                vllm_key = _apply_pattern(vllm_pattern, wildcards)
                vllm_state_dict[vllm_key] = levanter_state_dict[lev_key]
                logger.debug(f"Mapped {lev_key} -> {vllm_key}")

    for vllm_key in mapping_config.to_vllm_transpose_keys:
        if vllm_key in vllm_state_dict:
            original_shape = vllm_state_dict[vllm_key].shape
            vllm_state_dict[vllm_key] = jnp.transpose(vllm_state_dict[vllm_key])
            logger.debug(f"Transposed {vllm_key}: {original_shape} -> {vllm_state_dict[vllm_key].shape}")

    for vllm_pattern, hook_fn in mapping_config.to_vllm_hook_fns.items():
        if vllm_pattern in vllm_state_dict:
            vllm_state_dict[vllm_pattern] = hook_fn(vllm_state_dict[vllm_pattern])
            logger.debug(f"Applied hook to {vllm_pattern}")
        else:
            for vllm_key in list(vllm_state_dict.keys()):
                wildcards = _match_pattern(vllm_pattern, vllm_key)
                if wildcards is not None:
                    vllm_state_dict[vllm_key] = hook_fn(vllm_state_dict[vllm_key])
                    logger.debug(f"Applied hook to {vllm_key}")

    if target_device_mesh is not None:
        logger.info(f"Resharding parameters to target device mesh: {target_device_mesh}")
        vllm_state_dict = jax.lax.with_sharding_constraint(vllm_state_dict, PartitionSpec())

    logger.info(f"Weight transfer complete. Transferred {len(vllm_state_dict)} parameters")
    return vllm_state_dict


def get_vllm_compatible_state_dict(
    model: ModelWithHfSerializationMixin,
    dtype: Optional[jnp.dtype] = None,
) -> Dict[str, Array]:
    """
    Get a vLLM-compatible state dict from a Levanter model.

    This is a convenience function that uses the model's config to determine
    the appropriate mapping configuration.

    Args:
        model: Levanter model to transfer weights from
        dtype: Optional dtype to convert floating-point arrays to

    Returns:
        Dictionary mapping vLLM parameter names to JAX arrays

    Raises:
        AttributeError: If the model's config doesn't provide vLLM mapping configuration
    """
    if not hasattr(model.config, "vllm_mapping_config"):
        raise AttributeError(
            f"Model config {type(model.config).__name__} does not provide vllm_mapping_config() method. "
            "Please implement this method to enable vLLM weight transfer."
        )

    mapping_config = model.config.vllm_mapping_config()
    return transfer_weights_to_vllm(model, mapping_config, dtype=dtype)
