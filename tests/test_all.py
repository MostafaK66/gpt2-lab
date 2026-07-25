"""35 offline tests for gpt2-lab."""
from __future__ import annotations

import math
import numpy as np
import pytest
import torch

from stub_tokenizer import StubTokenizer

from gpt2lab.config.sections import (
    ModelConfig, DataConfig, TrainingConfig,
    EvalConfig, CheckpointConfig, VizConfig, SamplingConfig,
)
from gpt2lab.config.experiment import ExperimentConfig
from gpt2lab.models.gpt import GPT
from gpt2lab.models.attention import CausalSelfAttention
from gpt2lab.models.mlp import MLP
from gpt2lab.models.block import Block
from gpt2lab.models.initialization import init_weights
from gpt2lab.data.loader import DataLoader
from gpt2lab.training.optim import build_optimizer, cosine_lr
from gpt2lab.training.metrics import Metrics
from gpt2lab.training.callbacks import PrintCallback
from gpt2lab.training.runtime import select_device


# ── Helpers ──────────────────────────────────────────────

def _small_cfg(**kw) -> ModelConfig:
    defaults = dict(vocab_size=64, n_layer=2, n_head=2, n_embd=32,
                    block_size=16, dropout=0.0, bias=False)
    defaults.update(kw)
    return ModelConfig(**defaults)


def _small_model(cfg=None) -> GPT:
    return GPT(cfg or _small_cfg())


# ═══════════════════════ CONFIG TESTS ════════════════════

class TestModelConfig:
    def test_defaults(self):
        cfg = ModelConfig()
        assert cfg.n_embd % cfg.n_head == 0

    def test_validation_n_embd_mod_n_head(self):
        with pytest.raises(AssertionError):
            ModelConfig(n_embd=10, n_head=3)

    def test_validation_dropout_range(self):
        with pytest.raises(AssertionError):
            ModelConfig(dropout=1.0)

    def test_validation_negative_layers(self):
        with pytest.raises(AssertionError):
            ModelConfig(n_layer=0)


class TestDataConfig:
    def test_defaults(self):
        cfg = DataConfig()
        assert cfg.batch_size > 0

    def test_validation(self):
        with pytest.raises(AssertionError):
            DataConfig(batch_size=0)


class TestTrainingConfig:
    def test_defaults(self):
        cfg = TrainingConfig()
        assert cfg.learning_rate > 0

    def test_validation_betas(self):
        with pytest.raises(AssertionError):
            TrainingConfig(beta1=1.0)


class TestOtherConfigs:
    def test_eval_config(self):
        cfg = EvalConfig()
        assert cfg.eval_interval > 0

    def test_checkpoint_config(self):
        cfg = CheckpointConfig()
        assert cfg.save_interval > 0

    def test_viz_config(self):
        cfg = VizConfig()
        assert cfg.plot_interval > 0

    def test_sampling_config(self):
        cfg = SamplingConfig()
        assert cfg.temperature > 0

    def test_sampling_validation(self):
        with pytest.raises(AssertionError):
            SamplingConfig(temperature=-1)


class TestExperimentConfig:
    def test_from_flat_dict(self):
        cfg = ExperimentConfig.from_flat_dict({"n_layer": 4, "batch_size": 32})
        assert cfg.model.n_layer == 4
        assert cfg.data.batch_size == 32

    def test_unknown_keys_ignored(self):
        cfg = ExperimentConfig.from_flat_dict({"nonexistent_key": 999})
        assert cfg.model.n_layer == 6  # default


# ═══════════════════════ MODEL TESTS ═════════════════════

class TestAttention:
    def test_output_shape(self):
        cfg = _small_cfg()
        attn = CausalSelfAttention(cfg)
        x = torch.randn(2, cfg.block_size, cfg.n_embd)
        y = attn(x)
        assert y.shape == x.shape

    def test_causal_mask(self):
        cfg = _small_cfg()
        attn = CausalSelfAttention(cfg)
        assert attn.mask.shape == (1, 1, cfg.block_size, cfg.block_size)


class TestMLP:
    def test_output_shape(self):
        cfg = _small_cfg()
        mlp = MLP(cfg)
        x = torch.randn(2, 8, cfg.n_embd)
        assert mlp(x).shape == x.shape


class TestBlock:
    def test_output_shape(self):
        cfg = _small_cfg()
        block = Block(cfg)
        x = torch.randn(2, cfg.block_size, cfg.n_embd)
        assert block(x).shape == x.shape


class TestGPT:
    def test_forward(self):
        cfg = _small_cfg()
        model = _small_model(cfg)
        idx = torch.randint(0, cfg.vocab_size, (2, cfg.block_size))
        logits, loss = model(idx)
        assert logits.shape == (2, cfg.block_size, cfg.vocab_size)
        assert loss is None

    def test_forward_with_targets(self):
        cfg = _small_cfg()
        model = _small_model(cfg)
        idx = torch.randint(0, cfg.vocab_size, (2, cfg.block_size))
        _, loss = model(idx, targets=idx)
        assert loss is not None and loss.item() > 0

    def test_generate(self):
        cfg = _small_cfg()
        model = _small_model(cfg)
        model.eval()
        idx = torch.zeros(1, 1, dtype=torch.long)
        out = model.generate(idx, max_new_tokens=5)
        assert out.shape == (1, 6)

    def test_weight_tying(self):
        model = _small_model()
        assert model.transformer.wte.weight is model.lm_head.weight

    def test_num_parameters(self):
        model = _small_model()
        assert model.num_parameters() > 0

    def test_block_size_assertion(self):
        cfg = _small_cfg()
        model = _small_model(cfg)
        idx = torch.randint(0, cfg.vocab_size, (1, cfg.block_size + 1))
        with pytest.raises(AssertionError):
            model(idx)


class TestInitialization:
    def test_linear_init(self):
        m = torch.nn.Linear(10, 10)
        init_weights(m)
        assert m.weight.std().item() < 0.1


# ═══════════════════════ DATA TESTS ══════════════════════

class TestStubTokenizer:
    def test_encode_decode(self):
        tok = StubTokenizer()
        text = "hello"
        ids = tok.encode(text)
        assert len(ids) == len(text.encode("utf-8"))

    def test_roundtrip_ascii(self):
        tok = StubTokenizer()
        text = "test123"
        assert tok.decode(tok.encode(text)) == text


class TestDataLoader:
    def test_batch_shape(self):
        data = np.arange(100, dtype=np.uint16)
        loader = DataLoader(data, block_size=8, batch_size=4, device="cpu")
        x, y = loader.get_batch()
        assert x.shape == (4, 8)
        assert y.shape == (4, 8)

    def test_targets_shifted(self):
        data = np.arange(100, dtype=np.uint16)
        loader = DataLoader(data, block_size=8, batch_size=1, device="cpu")
        x, y = loader.get_batch()
        # y should be x shifted by 1 in the original data
        assert x.shape == y.shape


# ═══════════════════════ TRAINING TESTS ══════════════════

class TestOptim:
    def test_build_optimizer(self):
        model = _small_model()
        cfg = TrainingConfig()
        opt = build_optimizer(model, cfg)
        assert len(opt.param_groups) == 2

    def test_cosine_lr_warmup(self):
        cfg = TrainingConfig(warmup_steps=10, learning_rate=1e-3)
        assert cosine_lr(0, cfg) == 0.0
        assert cosine_lr(5, cfg) == pytest.approx(5e-4, rel=1e-5)

    def test_cosine_lr_decay(self):
        cfg = TrainingConfig(warmup_steps=0, lr_decay_steps=100,
                             learning_rate=1e-3, min_lr=1e-4)
        lr_mid = cosine_lr(50, cfg)
        assert 1e-4 < lr_mid < 1e-3


class TestMetrics:
    def test_log_train(self):
        m = Metrics()
        m.log_train(0, 4.5, 1e-3)
        assert len(m.train_losses) == 1

    def test_log_val(self):
        m = Metrics()
        m.log_val(100, 3.2)
        assert m.val_steps == [100]


class TestCallbacks:
    def test_print_callback(self, capsys):
        cb = PrintCallback(log_interval=1)
        m = Metrics()
        m.log_train(1, 4.0, 1e-3)
        cb.on_step_end(1, m)
        assert "loss" in capsys.readouterr().out


class TestRuntime:
    def test_select_cpu(self):
        d = select_device("cpu")
        assert d.type == "cpu"

