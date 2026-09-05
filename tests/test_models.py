from __future__ import annotations

from typing import Any, cast

import pytest
import torch
import torch.nn as nn

from gpt2lab.config import ModelConfig
from gpt2lab.models import GPT, MLP, CausalSelfAttention, TransformerBlock
from gpt2lab.models.initialization import (
    WeightInitializer,
    is_residual_projection,
    mark_residual_projection,
)


@pytest.fixture
def model_config() -> ModelConfig:
    return ModelConfig(
        block_size=4,
        n_layer=1,
        n_head=2,
        n_embd=8,
        mlp_ratio=2,
        dropout=0.0,
    )


def test_model_components_preserve_shape(model_config: ModelConfig) -> None:
    inputs = torch.randn(2, 3, 8)
    assert CausalSelfAttention(model_config)(inputs).shape == inputs.shape
    assert MLP(model_config)(inputs).shape == inputs.shape
    assert TransformerBlock(model_config)(inputs).shape == inputs.shape


def test_attention_is_causal(model_config: ModelConfig) -> None:
    attention = CausalSelfAttention(model_config).eval()
    first = torch.randn(1, 3, 8)
    changed_future = first.clone()
    changed_future[:, 1:, :] += 100
    assert torch.allclose(
        attention(first)[:, 0, :], attention(changed_future)[:, 0, :], atol=1e-5
    )


def test_bias_and_dropout_configuration() -> None:
    config = ModelConfig(
        n_layer=1, n_head=1, n_embd=4, block_size=4, bias=False, dropout=0.5
    )
    attention = CausalSelfAttention(config)
    mlp = MLP(config)
    assert cast(Any, attention.c_attn).bias is None
    assert cast(Any, attention.c_proj).bias is None
    assert cast(Any, mlp.c_fc).bias is None
    assert cast(Any, mlp.c_proj).bias is None
    attention.train()
    assert attention(torch.ones(2, 3, 4)).shape == (2, 3, 4)


def test_weight_initializer_marks_and_initializes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[float] = []

    def fake_normal_(tensor: torch.Tensor, *, mean: float, std: float) -> torch.Tensor:
        assert mean == 0.0
        observed.append(std)
        return tensor

    initializer = WeightInitializer(0.02, 2)
    linear = mark_residual_projection(nn.Linear(2, 2))
    embedding = nn.Embedding(3, 2)
    monkeypatch.setattr(nn.init, "normal_", fake_normal_)
    assert is_residual_projection(linear)
    assert not is_residual_projection(embedding)
    initializer(linear)
    initializer(embedding)
    initializer(nn.LayerNorm(2))
    assert initializer.init_std == 0.02
    assert initializer.residual_std == 0.01
    assert observed == [0.01, 0.02]
    assert torch.count_nonzero(linear.bias) == 0


@pytest.mark.parametrize("init_std,n_layer", [(0.0, 1), (0.1, 0)])
def test_weight_initializer_validation(init_std: float, n_layer: int) -> None:
    with pytest.raises(ValueError, match="positive"):
        WeightInitializer(init_std, n_layer)


def test_gpt_forward_loss_tying_and_parameter_count(model_config: ModelConfig) -> None:
    model = GPT(model_config, vocab_size=16)
    tokens = torch.tensor([[1, 2, 3], [4, 5, 6]])
    logits, loss = model(tokens, tokens)
    assert logits.shape == (2, 3, 16)
    assert loss is not None and loss.ndim == 0
    assert model.weights_are_tied
    assert model.num_parameters() > model.num_parameters(include_embeddings=False) > 0
    logits_without_targets, no_loss = model(tokens)
    assert logits_without_targets.shape == logits.shape
    assert no_loss is None


@pytest.mark.parametrize(
    ("idx", "targets", "message"),
    [
        (torch.ones(3, dtype=torch.long), None, "shape"),
        (torch.ones((1, 2)), None, "long"),
        (torch.ones((1, 0), dtype=torch.long), None, "empty"),
        (torch.ones((1, 5), dtype=torch.long), None, "block_size"),
        (
            torch.ones((1, 2), dtype=torch.long),
            torch.ones((1, 1), dtype=torch.long),
            "same shape",
        ),
        (
            torch.ones((1, 2), dtype=torch.long),
            torch.ones((1, 2)),
            "targets.*long",
        ),
    ],
)
def test_gpt_input_validation(
    model_config: ModelConfig,
    idx: torch.Tensor,
    targets: torch.Tensor | None,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        GPT(model_config, 8)(idx, targets)


def test_gpt_requires_positive_vocabulary(model_config: ModelConfig) -> None:
    with pytest.raises(ValueError, match="vocab_size"):
        GPT(model_config, 0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_new_tokens": 0},
        {"max_new_tokens": 1, "temperature": 0.0},
        {"max_new_tokens": 1, "top_k": 0},
    ],
)
def test_generate_validation(model_config: ModelConfig, kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        GPT(model_config, 8).generate(torch.ones((1, 1), dtype=torch.long), **kwargs)


def test_generate_extends_context_and_restores_mode(
    model_config: ModelConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = GPT(model_config, 8)
    model.train()

    def choose_first(probabilities: torch.Tensor, num_samples: int) -> torch.Tensor:
        assert num_samples == 1
        return probabilities.argmax(dim=-1, keepdim=True)

    monkeypatch.setattr(torch, "multinomial", choose_first)
    output = model.generate(
        torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long),
        max_new_tokens=2,
        temperature=0.5,
        top_k=999,
    )
    assert output.shape == (1, 7)
    assert model.training


def test_generate_restores_eval_mode(model_config: ModelConfig) -> None:
    model = GPT(model_config, 8).eval()
    model.generate(torch.ones((1, 1), dtype=torch.long), 1, top_k=None)
    assert not model.training
