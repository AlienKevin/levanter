# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""
Tests for vLLM weight transfer functionality.
"""

import jax
import jax.numpy as jnp
import jax.random as jrandom
import pytest

from haliax import Axis
from levanter.compat.vllm_transfer import (
    VllmMappingConfig,
    _apply_pattern,
    _match_pattern,
    transfer_weights_to_vllm,
)
from levanter.models.gpt2 import Gpt2Config, Gpt2LMHeadModel
from levanter.models.llama import LlamaConfig, LlamaLMHeadModel


@pytest.fixture(autouse=True)
def setup_jax_mesh():
    """Set up a JAX mesh context for tests that require it."""
    devices = jax.devices("cpu")
    mesh = jax.sharding.Mesh(devices, ("data",))
    with mesh:
        yield mesh


class TestPatternMatching:
    """Test pattern matching utilities."""

    def test_match_pattern_exact(self):
        pattern = "layers.0.attn.q.weight"
        key = "layers.0.attn.q.weight"
        result = _match_pattern(pattern, key)
        assert result == {}

    def test_match_pattern_single_wildcard(self):
        pattern = "layers.*.attn.weight"
        key = "layers.0.attn.weight"
        result = _match_pattern(pattern, key)
        assert result == {"0": "0"}

    def test_match_pattern_multiple_wildcards(self):
        pattern = "layers.*.attn.*.weight"
        key = "layers.0.attn.q.weight"
        result = _match_pattern(pattern, key)
        assert result == {"0": "0", "1": "q"}

    def test_match_pattern_no_match(self):
        pattern = "layers.*.attn.weight"
        key = "layers.0.mlp.weight"
        result = _match_pattern(pattern, key)
        assert result is None

    def test_match_pattern_length_mismatch(self):
        pattern = "layers.*.attn.weight"
        key = "layers.0.weight"
        result = _match_pattern(pattern, key)
        assert result is None

    def test_apply_pattern_single_wildcard(self):
        pattern = "model.layers.*.weight"
        wildcards = {"0": "5"}
        result = _apply_pattern(pattern, wildcards)
        assert result == "model.layers.5.weight"

    def test_apply_pattern_multiple_wildcards(self):
        pattern = "model.layers.*.attn.*.weight"
        wildcards = {"0": "3", "1": "q_proj"}
        result = _apply_pattern(pattern, wildcards)
        assert result == "model.layers.3.attn.q_proj.weight"


class TestVllmMappingConfig:
    """Test VllmMappingConfig dataclass."""

    def test_empty_config(self):
        config = VllmMappingConfig()
        assert config.to_vllm_mappings == {}
        assert config.to_vllm_transpose_keys == set()
        assert config.to_vllm_hook_fns == {}

    def test_config_with_mappings(self):
        mappings = {"vllm.key": "levanter.key"}
        config = VllmMappingConfig(to_vllm_mappings=mappings)
        assert config.to_vllm_mappings == mappings

    def test_config_with_transpose_keys(self):
        transpose_keys = {"key1", "key2"}
        config = VllmMappingConfig(to_vllm_transpose_keys=transpose_keys)
        assert config.to_vllm_transpose_keys == transpose_keys

    def test_config_with_hook_fns(self):
        hook_fn = lambda x: x * 2
        hook_fns = {"key": hook_fn}
        config = VllmMappingConfig(to_vllm_hook_fns=hook_fns)
        assert config.to_vllm_hook_fns == hook_fns


class TestGpt2VllmTransfer:
    """Test vLLM weight transfer for GPT-2 models."""

    @pytest.fixture
    def gpt2_config(self):
        return Gpt2Config(
            seq_len=128,
            hidden_dim=64,
            num_layers=2,
            num_heads=4,
            mlp_scale=2,
        )

    @pytest.fixture
    def gpt2_model(self, gpt2_config):
        Vocab = Axis("vocab", 100)
        key = jrandom.PRNGKey(0)
        return Gpt2LMHeadModel.init(Vocab, gpt2_config, key=key)

    def test_gpt2_has_vllm_mapping_config(self, gpt2_config):
        mapping_config = gpt2_config.vllm_mapping_config()
        assert isinstance(mapping_config, VllmMappingConfig)
        assert len(mapping_config.to_vllm_mappings) > 0

    def test_gpt2_transfer_weights(self, gpt2_model):
        mapping_config = gpt2_model.config.vllm_mapping_config()
        vllm_weights = transfer_weights_to_vllm(gpt2_model, mapping_config)

        assert isinstance(vllm_weights, dict)
        assert len(vllm_weights) > 0

        assert "transformer.wte.weight" in vllm_weights
        assert "transformer.wpe.weight" in vllm_weights
        assert "transformer.ln_f.weight" in vllm_weights

        for i in range(gpt2_model.config.num_layers):
            assert f"transformer.h.{i}.attn.c_attn.weight" in vllm_weights
            assert f"transformer.h.{i}.mlp.c_fc.weight" in vllm_weights

    def test_gpt2_transfer_preserves_shapes(self, gpt2_model):
        mapping_config = gpt2_model.config.vllm_mapping_config()
        vllm_weights = transfer_weights_to_vllm(gpt2_model, mapping_config)

        wte_shape = vllm_weights["transformer.wte.weight"].shape
        assert wte_shape == (100, 64)

    def test_gpt2_transfer_with_dtype(self, gpt2_model):
        mapping_config = gpt2_model.config.vllm_mapping_config()
        vllm_weights = transfer_weights_to_vllm(gpt2_model, mapping_config, dtype=jnp.float16)

        for key, value in vllm_weights.items():
            if jnp.issubdtype(value.dtype, jnp.floating):
                assert value.dtype == jnp.float16


class TestLlamaVllmTransfer:
    """Test vLLM weight transfer for Llama models."""

    @pytest.fixture
    def llama_config(self):
        return LlamaConfig(
            seq_len=128,
            hidden_dim=64,
            intermediate_dim=128,
            num_layers=2,
            num_heads=4,
            num_kv_heads=2,
        )

    @pytest.fixture
    def llama_model(self, llama_config):
        Vocab = Axis("vocab", 100)
        key = jrandom.PRNGKey(0)
        return LlamaLMHeadModel.init(Vocab, llama_config, key=key)

    def test_llama_has_vllm_mapping_config(self, llama_config):
        mapping_config = llama_config.vllm_mapping_config()
        assert isinstance(mapping_config, VllmMappingConfig)
        assert len(mapping_config.to_vllm_mappings) > 0

    def test_llama_transfer_weights(self, llama_model):
        mapping_config = llama_model.config.vllm_mapping_config()
        vllm_weights = transfer_weights_to_vllm(llama_model, mapping_config)

        assert isinstance(vllm_weights, dict)
        assert len(vllm_weights) > 0

        assert "model.embed_tokens.weight" in vllm_weights
        assert "model.norm.weight" in vllm_weights

        for i in range(llama_model.config.num_layers):
            assert f"model.layers.{i}.self_attn.q_proj.weight" in vllm_weights
            assert f"model.layers.{i}.self_attn.k_proj.weight" in vllm_weights
            assert f"model.layers.{i}.self_attn.v_proj.weight" in vllm_weights
            assert f"model.layers.{i}.self_attn.o_proj.weight" in vllm_weights
            assert f"model.layers.{i}.mlp.gate_proj.weight" in vllm_weights
            assert f"model.layers.{i}.mlp.up_proj.weight" in vllm_weights
            assert f"model.layers.{i}.mlp.down_proj.weight" in vllm_weights

    def test_llama_transfer_with_tied_embeddings(self):
        config = LlamaConfig(
            seq_len=128,
            hidden_dim=64,
            intermediate_dim=128,
            num_layers=2,
            num_heads=4,
            num_kv_heads=2,
            tie_word_embeddings=True,
        )
        Vocab = Axis("vocab", 100)
        key = jrandom.PRNGKey(0)
        model = LlamaLMHeadModel.init(Vocab, config, key=key)

        mapping_config = model.config.vllm_mapping_config()
        vllm_weights = transfer_weights_to_vllm(model, mapping_config)

        assert "lm_head.weight" not in vllm_weights

    def test_llama_transfer_with_untied_embeddings(self):
        config = LlamaConfig(
            seq_len=128,
            hidden_dim=64,
            intermediate_dim=128,
            num_layers=2,
            num_heads=4,
            num_kv_heads=2,
            tie_word_embeddings=False,
        )
        Vocab = Axis("vocab", 100)
        key = jrandom.PRNGKey(0)
        model = LlamaLMHeadModel.init(Vocab, config, key=key)

        mapping_config = model.config.vllm_mapping_config()
        vllm_weights = transfer_weights_to_vllm(model, mapping_config)

        assert "lm_head.weight" in vllm_weights


class TestTransposeAndHooks:
    """Test transpose operations and custom hooks."""

    def test_transpose_operation(self):
        from levanter.models.gpt2 import Gpt2Config, Gpt2LMHeadModel

        gpt2_config = Gpt2Config(seq_len=128, hidden_dim=64, num_layers=1, num_heads=4)
        Vocab = Axis("vocab", 100)
        key = jrandom.PRNGKey(0)
        model = Gpt2LMHeadModel.init(Vocab, gpt2_config, key=key)

        test_config = VllmMappingConfig(
            to_vllm_mappings={"test.weight": "wte.weight"},
            to_vllm_transpose_keys={"test.weight"},
        )

        vllm_weights = transfer_weights_to_vllm(model, test_config)
        original_shape = (100, 64)
        transposed_shape = vllm_weights["test.weight"].shape

        assert transposed_shape == (original_shape[1], original_shape[0])

    def test_custom_hook(self):
        from levanter.models.gpt2 import Gpt2Config, Gpt2LMHeadModel

        gpt2_config = Gpt2Config(seq_len=128, hidden_dim=64, num_layers=1, num_heads=4)
        Vocab = Axis("vocab", 100)
        key = jrandom.PRNGKey(0)
        model = Gpt2LMHeadModel.init(Vocab, gpt2_config, key=key)

        scale_factor = 2.0
        test_config = VllmMappingConfig(
            to_vllm_mappings={"test.weight": "wte.weight"},
            to_vllm_hook_fns={"test.weight": lambda x: x * scale_factor},
        )

        vllm_weights = transfer_weights_to_vllm(model, test_config)

        from levanter.compat.hf_checkpoints import _to_state_dict_with_dtype

        original_state_dict = _to_state_dict_with_dtype(model, None, None)
        original_weight = original_state_dict["wte.weight"]

        assert jnp.allclose(vllm_weights["test.weight"], original_weight * scale_factor)
