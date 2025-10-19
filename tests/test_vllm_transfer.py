# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""
Integration tests for vLLM weight transfer.

These tests verify that Levanter models can be successfully saved and loaded with vLLM.
"""

import tempfile

import jax
import jax.numpy as jnp
import pytest

from levanter.compat.vllm_transfer import load_with_vllm, save_for_vllm
from levanter.models.gpt2 import Gpt2Config, Gpt2LMHeadModel
from levanter.models.llama import LlamaConfig, LlamaLMHeadModel
from test_utils import skip_if_no_torch


try:
    import vllm  # noqa: F401

    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False


@pytest.fixture
def setup_jax_mesh():
    """Set up a JAX mesh for tests that require it."""
    devices = jax.devices()
    mesh = jax.sharding.Mesh(devices, ("data",))
    with mesh:
        yield mesh


@skip_if_no_torch
@pytest.mark.skipif(not VLLM_AVAILABLE, reason="vLLM not installed")
def test_gpt2_save_and_load_with_vllm(setup_jax_mesh):
    """Test saving a GPT-2 model and loading it with vLLM."""
    config = Gpt2Config(
        num_layers=2,
        num_heads=4,
        hidden_dim=128,
        seq_len=128,
        gradient_checkpointing=False,
        use_bias=True,
    )

    Vocab = config.Vocab
    converter = config.hf_checkpoint_converter()

    key = jax.random.PRNGKey(0)
    model = Gpt2LMHeadModel.init(Vocab, config, key=key)

    with tempfile.TemporaryDirectory() as tmpdir:
        save_for_vllm(model, converter, tmpdir, dtype=jnp.float32)

        llm = load_with_vllm(tmpdir, max_model_len=128, enforce_eager=True)

        assert llm is not None

        outputs = llm.generate(["Hello, world!"], max_tokens=10)
        assert len(outputs) == 1
        assert len(outputs[0].outputs) > 0


@skip_if_no_torch
@pytest.mark.skipif(not VLLM_AVAILABLE, reason="vLLM not installed")
def test_llama_save_and_load_with_vllm(setup_jax_mesh):
    """Test saving a Llama model and loading it with vLLM."""
    config = LlamaConfig(
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        hidden_dim=128,
        intermediate_dim=256,
        seq_len=128,
        gradient_checkpointing=False,
        rope=True,
    )

    Vocab = config.Vocab
    converter = config.hf_checkpoint_converter()

    key = jax.random.PRNGKey(0)
    model = LlamaLMHeadModel.init(Vocab, config, key=key)

    with tempfile.TemporaryDirectory() as tmpdir:
        save_for_vllm(model, converter, tmpdir, dtype=jnp.float32)

        llm = load_with_vllm(tmpdir, max_model_len=128, enforce_eager=True)

        assert llm is not None

        outputs = llm.generate(["Hello, world!"], max_tokens=10)
        assert len(outputs) == 1
        assert len(outputs[0].outputs) > 0


@skip_if_no_torch
@pytest.mark.skipif(not VLLM_AVAILABLE, reason="vLLM not installed")
def test_gpt2_inference_consistency(setup_jax_mesh):
    """Test that vLLM inference produces reasonable outputs."""
    config = Gpt2Config(
        num_layers=2,
        num_heads=4,
        hidden_dim=128,
        seq_len=128,
        gradient_checkpointing=False,
        use_bias=True,
    )

    Vocab = config.Vocab
    converter = config.hf_checkpoint_converter()

    key = jax.random.PRNGKey(42)
    model = Gpt2LMHeadModel.init(Vocab, config, key=key)

    with tempfile.TemporaryDirectory() as tmpdir:
        save_for_vllm(model, converter, tmpdir, dtype=jnp.float32)

        llm = load_with_vllm(tmpdir, max_model_len=128, enforce_eager=True)

        prompt = "The quick brown fox"
        outputs = llm.generate([prompt], max_tokens=20)

        assert len(outputs) == 1
        generated_text = outputs[0].outputs[0].text
        assert len(generated_text) > 0


@skip_if_no_torch
def test_save_for_vllm_creates_checkpoint(setup_jax_mesh):
    """Test that save_for_vllm creates a valid checkpoint directory."""
    config = Gpt2Config(
        num_layers=2,
        num_heads=4,
        hidden_dim=128,
        seq_len=128,
        gradient_checkpointing=False,
        use_bias=True,
    )

    Vocab = config.Vocab
    converter = config.hf_checkpoint_converter()

    key = jax.random.PRNGKey(0)
    model = Gpt2LMHeadModel.init(Vocab, config, key=key)

    with tempfile.TemporaryDirectory() as tmpdir:
        save_for_vllm(model, converter, tmpdir, dtype=jnp.float32)

        import os

        assert os.path.exists(tmpdir)
        assert os.path.exists(os.path.join(tmpdir, "config.json"))
        assert any(f.endswith(".safetensors") or f.endswith(".bin") for f in os.listdir(tmpdir))


@skip_if_no_torch
@pytest.mark.skipif(not VLLM_AVAILABLE, reason="vLLM not installed")
def test_llama_with_untied_embeddings(setup_jax_mesh):
    """Test Llama model with untied embeddings."""
    config = LlamaConfig(
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        hidden_dim=128,
        intermediate_dim=256,
        seq_len=128,
        gradient_checkpointing=False,
        rope=True,
        tie_word_embeddings=False,
    )

    Vocab = config.Vocab
    converter = config.hf_checkpoint_converter()

    key = jax.random.PRNGKey(0)
    model = LlamaLMHeadModel.init(Vocab, config, key=key)

    with tempfile.TemporaryDirectory() as tmpdir:
        save_for_vllm(model, converter, tmpdir, dtype=jnp.float32)

        llm = load_with_vllm(tmpdir, max_model_len=128, enforce_eager=True)

        assert llm is not None

        outputs = llm.generate(["Test prompt"], max_tokens=10)
        assert len(outputs) == 1
        assert len(outputs[0].outputs) > 0
