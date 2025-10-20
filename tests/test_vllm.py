# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
import logging
import tempfile
from pathlib import Path
import time
import os

import jax

# Colocate vllm engine and worker in the main process
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

logger = logging.getLogger(__name__)

pytest.importorskip("torch")
pytest.importorskip("vllm")

from vllm import LLM, SamplingParams  # noqa: E402

MODEL_ID = "Qwen/Qwen3-0.6B"
MODEL_ID_BASE = "Qwen/Qwen3-0.6B-Base"
PROMPTS = (
    "Hello, my name is",
    "The capital of France is",
    "In a distant future, humanity",
)
MAX_NEW_TOKENS = 8

# Expected outputs from the reference implementation
EXPECTED_OUTPUTS_CHAT = (
    " Lina. I'm a 2",
    " Paris. The capital of Italy is Rome",
    " has developed a new technology called \"Quant",
)

EXPECTED_OUTPUTS_BASE = (
    " John. I am a student at the",
    " Paris. The capital of Germany is Berlin",
    " has developed a new technology that allows them",
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
        model=MODEL_ID,
        tokenizer=MODEL_ID,
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
        model=MODEL_ID_BASE,
        tokenizer=MODEL_ID_BASE,
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
"""
Qwen/Qwen3-0.6B
'Hello, my name is' -> " Lina. I'm a 2"
'The capital of France is' -> ' Paris. The capital of Italy is Rome'
'In a distant future, humanity' -> ' has developed a new technology called "Quant'

Qwen/Qwen3-0.6B-Base
'Hello, my name is' -> ' John. I am a student at the'
'The capital of France is' -> ' Paris. The capital of Germany is Berlin'
'In a distant future, humanity' -> ' has developed a new technology that allows them'
"""


# def test_vllm_base_model_generation(vllm_llm_base: LLM) -> None:
#     """Test that vLLM can generate from the base model correctly."""

#     sampling_params = SamplingParams(temperature=0.0, max_tokens=MAX_NEW_TOKENS)
#     outputs = vllm_llm_base.generate(PROMPTS, sampling_params)

#     assert len(outputs) == len(PROMPTS)
#     for prompt, result, expected in zip(PROMPTS, outputs, EXPECTED_OUTPUTS_BASE, strict=True):
#         completion = result.outputs[0]
#         logger.info("Prompt %r -> %r", prompt, completion.text)
#         assert completion.token_ids, f"No tokens generated for prompt: {prompt}"
#         assert completion.text.strip(), f"No text generated for prompt: {prompt}"
#         # Verify output matches expected
#         assert completion.text == expected, (
#             f"Output mismatch for prompt {prompt!r}: "
#             f"expected {expected!r}, got {completion.text!r}"
#         )


def test_levanter_weight_transfer_to_vllm() -> None:
    """
    Test weight transfer from Levanter to vLLM.

    This test:
    1. Loads Qwen3-0.6B-Base model into Levanter
    2. Converts Levanter weights to HuggingFace format
    3. Loads the converted weights into vLLM
    4. Verifies that generation outputs match the expected outputs
    """

    pytest.importorskip("levanter")

    from levanter.models.qwen import Qwen3Config, Qwen3LMHeadModel
    from levanter.compat.hf_checkpoints import HFCheckpointConverter
    from transformers import AutoModelForCausalLM

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        hf_output_dir = tmpdir_path / "hf_output"
        hf_output_dir.mkdir(parents=True)

        # Step 1: Load HF model config and convert to Levanter
        logger.info("Loading HF config from %s", MODEL_ID_BASE)
        hf_model = AutoModelForCausalLM.from_pretrained(MODEL_ID_BASE, trust_remote_code=True)
        hf_config = hf_model.config

        # Create Levanter config from HF config
        lev_config = Qwen3Config.from_hf_config(hf_config)
        logger.info("Created Levanter config: %s", lev_config)

        # Step 2: Get the converter and load HF weights into Levanter
        converter = lev_config.hf_checkpoint_converter(ref_checkpoint=MODEL_ID_BASE)
        logger.info("Loading Levanter model from HF checkpoint: %s", MODEL_ID_BASE)
        lev_model = converter.load_pretrained(
            lm_model_cls=Qwen3LMHeadModel,
            ref=MODEL_ID_BASE,
            resize_vocab_to_match_tokenizer=False,
        )
        logger.info("Successfully loaded Levanter model")

        # Step 3: Convert Levanter model back to HF format
        logger.info("Converting Levanter model to HF format")
        converter.save_pretrained(
            lev_model,
            str(hf_output_dir),
        )
        logger.info("Successfully saved HF model to %s", hf_output_dir)

        # Step 4: Load the converted model into vLLM
        logger.info("Loading converted model into vLLM from %s", hf_output_dir)
        vllm_model = LLM(
            model=str(hf_output_dir),
            tokenizer=MODEL_ID_BASE,
            tensor_parallel_size=1,
            seed=0,
            max_model_len=256,
            trust_remote_code=True,
        )
        logger.info("Successfully loaded vLLM model")

        # Step 5: Run generation and verify outputs
        sampling_params = SamplingParams(temperature=0.0, max_tokens=MAX_NEW_TOKENS)
        outputs = vllm_model.generate(PROMPTS, sampling_params)

        assert len(outputs) == len(PROMPTS)
        for prompt, result, expected in zip(PROMPTS, outputs, EXPECTED_OUTPUTS_BASE, strict=True):
            completion = result.outputs[0]
            logger.info("Prompt %r -> %r", prompt, completion.text)
            # Verify output matches expected
            assert completion.text == expected, (
                f"Output mismatch for prompt {prompt!r}: "
                f"expected {expected!r}, got {completion.text!r}"
            )

        # Cleanup: Explicitly delete the vLLM model to free TPU resources
        logger.info("Cleaning up test - freeing TPU resources from vLLM model")
        del vllm_model
        import gc
        gc.collect()
