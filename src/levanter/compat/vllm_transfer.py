# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""
Weight transfer utilities for converting Levanter models to vLLM format.

This module provides functionality to transfer trained model weights from Levanter
to vLLM during training for inference/generation. It leverages Levanter's existing
HuggingFace checkpoint conversion infrastructure since vLLM uses the same parameter
naming convention as HuggingFace Transformers.
"""

import logging
import tempfile
from pathlib import Path
from typing import Optional

import jax.numpy as jnp

from levanter.compat.hf_checkpoints import HFCheckpointConverter, ModelWithHfSerializationMixin

logger = logging.getLogger(__name__)


def save_for_vllm(
    model: ModelWithHfSerializationMixin,
    checkpoint_converter: HFCheckpointConverter,
    output_dir: str | Path,
    dtype: Optional[jnp.dtype] = None,
) -> None:
    """
    Save a Levanter model in vLLM-compatible format.

    This function saves the model using Levanter's existing HuggingFace checkpoint
    converter, which produces a format that vLLM can directly load since vLLM uses
    the same parameter naming convention as HuggingFace Transformers.

    Args:
        model: Levanter model to save
        checkpoint_converter: HFCheckpointConverter for the model
        output_dir: Directory to save the checkpoint
        dtype: Optional dtype to convert floating-point arrays to

    Example:
        >>> from levanter.models.gpt2 import Gpt2Config, Gpt2LMHeadModel
        >>> config = Gpt2Config()
        >>> converter = config.hf_checkpoint_converter()
        >>> model = Gpt2LMHeadModel.init(Vocab, config, key=key)
        >>> save_for_vllm(model, converter, "/path/to/checkpoint", dtype=jnp.float16)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving model for vLLM to {output_dir}")

    checkpoint_converter.save_pretrained(
        model,
        str(output_dir),
        dtype=dtype,
        save_reference_code=False,
        save_tokenizer=True,
    )

    logger.info(f"Model saved successfully to {output_dir}")


def load_with_vllm(
    checkpoint_path: str | Path,
    **vllm_kwargs,
):
    """
    Load a Levanter checkpoint with vLLM for inference.

    This function loads a checkpoint saved by save_for_vllm() using vLLM's LLM class.
    The checkpoint must be in HuggingFace format (which save_for_vllm() produces).

    Args:
        checkpoint_path: Path to the checkpoint directory
        **vllm_kwargs: Additional keyword arguments to pass to vLLM's LLM constructor
            (e.g., tensor_parallel_size, dtype, etc.)

    Returns:
        vLLM LLM instance ready for inference

    Example:
        >>> llm = load_with_vllm("/path/to/checkpoint", tensor_parallel_size=2)
        >>> outputs = llm.generate(["Hello, world!"], sampling_params=...)
    """
    try:
        from vllm import LLM
    except ImportError as e:
        raise ImportError(
            "vLLM is required for this function. Install it with: pip install vllm-tpu"
        ) from e

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    logger.info(f"Loading checkpoint from {checkpoint_path} with vLLM")

    llm = LLM(model=str(checkpoint_path), **vllm_kwargs)

    logger.info("Model loaded successfully with vLLM")
    return llm


def transfer_to_vllm_temp(
    model: ModelWithHfSerializationMixin,
    checkpoint_converter: HFCheckpointConverter,
    dtype: Optional[jnp.dtype] = None,
    **vllm_kwargs,
):
    """
    Transfer a Levanter model to vLLM via a temporary checkpoint.

    This is a convenience function that saves the model to a temporary directory
    and loads it with vLLM. Useful for quick testing or when you don't need to
    persist the checkpoint.

    Args:
        model: Levanter model to transfer
        checkpoint_converter: HFCheckpointConverter for the model
        dtype: Optional dtype to convert floating-point arrays to
        **vllm_kwargs: Additional keyword arguments to pass to vLLM's LLM constructor

    Returns:
        vLLM LLM instance ready for inference

    Example:
        >>> from levanter.models.gpt2 import Gpt2Config, Gpt2LMHeadModel
        >>> config = Gpt2Config()
        >>> converter = config.hf_checkpoint_converter()
        >>> model = Gpt2LMHeadModel.init(Vocab, config, key=key)
        >>> llm = transfer_to_vllm_temp(model, converter, dtype=jnp.float16)
        >>> outputs = llm.generate(["Hello!"], sampling_params=...)
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        save_for_vllm(model, checkpoint_converter, tmpdir, dtype=dtype)
        return load_with_vllm(tmpdir, **vllm_kwargs)
