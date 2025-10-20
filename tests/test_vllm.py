# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
import logging

logger = logging.getLogger(__name__)

pytest.importorskip("torch")
pytest.importorskip("vllm")

from vllm import LLM, SamplingParams  # noqa: E402

MODEL_ID = "Qwen/Qwen3-0.6B"
PROMPTS = (
    "Hello, my name is",
    "The capital of France is",
    "In a distant future, humanity",
)
MAX_NEW_TOKENS = 8


@pytest.fixture(scope="module")
def vllm_llm() -> LLM:
    return LLM(
        model=MODEL_ID,
        tokenizer=MODEL_ID,
        tensor_parallel_size=1,
        seed=0,
        max_model_len=256,
    )

# uv run --extra tpu pytest tests/test_vllm.py --log-cli-level=INFO
def test_vllm_tpu_greedy_generation(vllm_llm: LLM) -> None:
    sampling_params = SamplingParams(temperature=0.0, max_tokens=MAX_NEW_TOKENS)
    outputs = vllm_llm.generate(PROMPTS, sampling_params)

    assert len(outputs) == len(PROMPTS)
    for prompt, result in zip(PROMPTS, outputs, strict=True):
        completion = result.outputs[0]
        logger.info("Prompt %r -> %r", prompt, completion.text)
        assert completion.token_ids, f"No tokens generated for prompt: {prompt}"
        assert completion.text.strip(), f"No text generated for prompt: {prompt}"
