# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Type

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

# Colocate vllm engine and worker in the main process
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

logger = logging.getLogger(__name__)

pytest.importorskip("torch")
pytest.importorskip("vllm")
pytest.importorskip("levanter")

from levanter.compat.hf_checkpoints import _to_state_dict_with_dtype  # noqa: E402
from levanter.models.llama import LlamaConfig, LlamaLMHeadModel  # noqa: E402
from levanter.models.qwen import Qwen3Config, Qwen3LMHeadModel  # noqa: E402
from tpu_inference.models.jax.utils.weight_utils import (  # noqa: E402
    shard_put,
    transfer_state_with_mappings,
)
from transformers import AutoModelForCausalLM  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402

QWEN_MODEL_ID = "Qwen/Qwen3-0.6B"
QWEN_MODEL_ID_BASE = "Qwen/Qwen3-0.6B-Base"
LLAMA_MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
LLAMA_MODEL_ID_BASE = "meta-llama/Llama-3.2-1B"
PROMPTS = (
    "Hello, my name is",
    "The capital of France is",
    "In a distant future, humanity",
)
MAX_NEW_TOKENS = 8

# Expected outputs from the reference implementation
EXPECTED_OUTPUTS_QWEN_CHAT = (
    " Lina. I'm a 2",
    " Paris. The capital of Italy is Rome",
    " has developed a new technology called \"Quant",
)

EXPECTED_OUTPUTS_QWEN_BASE = (
    " John. I am a student at the",
    " Paris. The capital of Germany is Berlin",
    " has developed a new technology that allows them",
)

EXPECTED_OUTPUTS_LLAMA_CHAT = (
    " Emily and I'm a huge fan of",
    " Paris. The Eiffel Tower is",
    " has colonized other planets and formed a",
)

EXPECTED_OUTPUTS_LLAMA_BASE = (
    " Kelsey and I am a 201",
    " Paris. It is the most visited city",
    " has been driven to the edge of the",
)


@dataclass(frozen=True)
class WeightTransferTestCase:
    name: str
    base_model_id: str
    chat_model_id: str
    expected_base_outputs: tuple[str, ...]
    expected_chat_outputs: tuple[str, ...]
    config_cls: Type
    lm_model_cls: Type


TEST_CASES: tuple[WeightTransferTestCase, ...] = (
    WeightTransferTestCase(
        name="llama3_1b",
        base_model_id=LLAMA_MODEL_ID_BASE,
        chat_model_id=LLAMA_MODEL_ID,
        expected_base_outputs=EXPECTED_OUTPUTS_LLAMA_BASE,
        expected_chat_outputs=EXPECTED_OUTPUTS_LLAMA_CHAT,
        config_cls=LlamaConfig,
        lm_model_cls=LlamaLMHeadModel,
    ),
    WeightTransferTestCase(
        name="qwen3_0_6b",
        base_model_id=QWEN_MODEL_ID_BASE,
        chat_model_id=QWEN_MODEL_ID,
        expected_base_outputs=EXPECTED_OUTPUTS_QWEN_BASE,
        expected_chat_outputs=EXPECTED_OUTPUTS_QWEN_CHAT,
        config_cls=Qwen3Config,
        lm_model_cls=Qwen3LMHeadModel,
    ),
)

def setup_module(module):
    """Initialize module-level test setup."""
    logger.info("Setting up vLLM tests - initializing TPU resources")


def teardown_module(module):
    """Clean up TPU resources after all tests in module."""
    logger.info("Tearing down vLLM tests - clearing JAX TPU cache and resources")
    try:
        jax.distributed.shutdown()
    except Exception as e:
        logger.warning("Error during jax.distributed.shutdown: %s", e)

    # Force garbage collection to free TPU memory
    import gc
    gc.collect()

    # Wait a bit for resources to be released
    time.sleep(1)

    logger.info("Teardown complete - TPU resources released")


@pytest.fixture(scope="module")
def vllm_llm() -> LLM:
    llm = LLM(
        # model=QWEN_MODEL_ID,
        # tokenizer=QWEN_MODEL_ID,
        model=LLAMA_MODEL_ID,
        tokenizer=LLAMA_MODEL_ID,
        tensor_parallel_size=1,
        seed=0,
        max_model_len=256,
    )
    yield llm
    # Cleanup: Delete the model and free TPU memory
    logger.info("Cleaning up vllm_llm fixture - freeing TPU resources")
    del llm
    import gc
    gc.collect()


@pytest.fixture(scope="module")
def vllm_llm_base() -> LLM:
    llm = LLM(
        # model=QWEN_MODEL_ID_BASE,
        # tokenizer=QWEN_MODEL_ID_BASE,
        model=LLAMA_MODEL_ID_BASE,
        tokenizer=LLAMA_MODEL_ID_BASE,
        tensor_parallel_size=1,
        seed=0,
        max_model_len=256,
    )
    yield llm
    # Cleanup: Delete the model and free TPU memory
    logger.info("Cleaning up vllm_llm_base fixture - freeing TPU resources")
    del llm
    import gc
    gc.collect()


# uv run --extra tpu pytest tests/test_vllm.py --log-cli-level=INFO -s
# # Swap out vllm_llm with vllm_llm_base to test Base model
def test_vllm_model_generation(vllm_llm_base: LLM) -> None:
    """Test that vLLM can generate from the model correctly."""

    sampling_params = SamplingParams(temperature=0.0, max_tokens=MAX_NEW_TOKENS)
    outputs = vllm_llm_base.generate(PROMPTS, sampling_params)

    assert len(outputs) == len(PROMPTS)
    for prompt, result, expected in zip(PROMPTS, outputs, EXPECTED_OUTPUTS_LLAMA_BASE, strict=True):
        completion = result.outputs[0]
        logger.info("Prompt %r -> %r", prompt, completion.text)
        assert completion.token_ids, f"No tokens generated for prompt: {prompt}"
        assert completion.text.strip(), f"No text generated for prompt: {prompt}"
        # Verify output matches expected
        assert completion.text == expected, (
            f"Output mismatch for prompt {prompt!r}: "
            f"expected {expected!r}, got {completion.text!r}"
        )

def _target_key_to_hf_key(path_str: str) -> str | None:
    if path_str.startswith("model.embed.embedding"):
        return "model.embed_tokens.weight"
    if path_str.startswith("model.") and path_str.endswith(".scale"):
        return path_str.replace(".scale", ".weight")
    if path_str.startswith("model.") and path_str.endswith(".kernel"):
        return path_str.replace(".kernel", ".weight")
    return None


def _reshape_weight(
    hf_key: str,
    value,
    target_shape: tuple[int, ...],
    lev_config,
):
    if value.shape == target_shape:
        return value
    if value.ndim == 2 and value.shape[::-1] == target_shape:
        return value.T

    hidden_size = lev_config.hidden_dim
    head_dim = lev_config.head_dim or (hidden_size // lev_config.num_heads)

    if hf_key.endswith("self_attn.q_proj.weight"):
        reshaped = value.reshape(lev_config.num_heads, head_dim, hidden_size)
        return jnp.transpose(reshaped, (2, 0, 1))

    if hf_key.endswith("self_attn.k_proj.weight") or hf_key.endswith("self_attn.v_proj.weight"):
        reshaped = value.reshape(lev_config.num_kv_heads, head_dim, hidden_size)
        return jnp.transpose(reshaped, (2, 0, 1))

    if hf_key.endswith("self_attn.o_proj.weight"):
        reshaped = value.reshape(hidden_size, lev_config.num_heads, head_dim)
        return jnp.transpose(reshaped, (1, 2, 0))

    raise ValueError(f"Unexpected reshape for {hf_key}: {value.shape} -> {target_shape}")


class _StateWrapper:
    def __init__(self, entries):
        self._entries = entries

    def flat_state(self):
        return self._entries


def _assert_outputs(outputs, expected_outputs: tuple[str, ...], phase: str, case: WeightTransferTestCase) -> None:
    assert len(outputs) == len(PROMPTS)
    for prompt, result, expected in zip(PROMPTS, outputs, expected_outputs, strict=True):
        completion = result.outputs[0]
        logger.info("%s[%s] prompt %r -> %r", phase, case.name, prompt, completion.text)
        assert completion.text == expected, (
            f"{phase} output mismatch for prompt {prompt!r} ({case.name}): "
            f"expected {expected!r}, got {completion.text!r}"
        )


def _run_weight_transfer_test(case: WeightTransferTestCase) -> None:
    logging.getLogger("tpu_inference").setLevel(logging.WARNING)

    logger.info("Loading vLLM model with base weights: %s (%s)", case.base_model_id, case.name)
    vllm_model = LLM(
        model=case.base_model_id,
        tokenizer=case.base_model_id,
        tensor_parallel_size=1,
        seed=0,
        max_model_len=256,
        trust_remote_code=True,
    )

    sampling_params = SamplingParams(temperature=0.0, max_tokens=MAX_NEW_TOKENS)

    logger.info("Validating base weights before transfer for %s", case.name)
    base_outputs = vllm_model.generate(PROMPTS, sampling_params)
    _assert_outputs(base_outputs, case.expected_base_outputs, "Base", case)

    runner = vllm_model.llm_engine.model_executor.driver_worker.model_runner
    target_state = runner.state
    target_flat = list(target_state.flat_state())

    logger.info("Loading Levanter source model: %s (%s)", case.chat_model_id, case.name)
    hf_model = AutoModelForCausalLM.from_pretrained(case.chat_model_id, trust_remote_code=True)
    hf_config = hf_model.config
    lev_config = case.config_cls.from_hf_config(hf_config)
    converter = lev_config.hf_checkpoint_converter(ref_checkpoint=case.chat_model_id)
    lev_model = converter.load_pretrained(
        lm_model_cls=case.lm_model_cls,
        ref=case.chat_model_id,
        resize_vocab_to_match_tokenizer=False,
    )
    lev_state = _to_state_dict_with_dtype(lev_model, None, None)

    del hf_model  # free HF model weights early

    logger.info("Preparing mapping for weight transfer (%s)", case.name)
    src_entries = []
    mappings: dict[str, tuple[str, Any]] = {}

    for target_path, param in target_flat:
        target_key = ".".join(str(part) for part in target_path)
        if target_key.startswith("rng."):
            continue

        hf_key = _target_key_to_hf_key(target_key)
        if hf_key is None:
            continue

        value = lev_state.get(hf_key)
        if value is None:
            logger.debug("No HF value for %s (target %s) [%s]", hf_key, target_key, case.name)
            continue

        reshaped = _reshape_weight(hf_key, value, param.value.shape, lev_config)
        if reshaped.dtype != param.value.dtype:
            reshaped = reshaped.astype(param.value.dtype)

        src_entries.append((tuple(hf_key.split(".")), nnx.Param(reshaped)))
        mappings[hf_key] = (target_key, param.value.sharding)

    assert mappings, f"No mappings constructed for weight transfer ({case.name})"

    src_state = _StateWrapper(src_entries)
    mesh = runner.mesh

    logger.info("Transferring weights from Levanter to vLLM (%s)", case.name)
    runner.state = transfer_state_with_mappings(
        src_state,
        target_state,
        mappings,
        shard=lambda array, sharding: shard_put(array, sharding, mesh),
    )

    logger.info("Running generation to validate transfer for %s", case.name)
    outputs = vllm_model.generate(PROMPTS, sampling_params)
    _assert_outputs(outputs, case.expected_chat_outputs, "Chat", case)

    logger.info("Cleaning up test - freeing TPU resources from vLLM model (%s)", case.name)
    del vllm_model
    import gc
    gc.collect()

    # Wait until vLLM is deallocated
    time.sleep(10)


@pytest.mark.parametrize("case", TEST_CASES, ids=lambda case: case.name)
def test_levanter_weight_transfer_to_vllm(case: WeightTransferTestCase) -> None:
    """
    Test weight transfer from Levanter to vLLM for different model families.
    """

    _run_weight_transfer_test(case)
