from types import SimpleNamespace

from vllm.model_executor.models.gpt_oss import _get_layer_sliding_window


def test_gpt_oss_window_uses_layer_types_instead_of_layer_parity() -> None:
    config = SimpleNamespace(
        sliding_window=64,
        layer_types=["full_attention", "sliding_attention"],
    )

    assert _get_layer_sliding_window(config, 0) is None
    assert _get_layer_sliding_window(config, 1) == 64


def test_gpt_oss_window_keeps_parity_fallback_for_legacy_configs() -> None:
    config = SimpleNamespace(sliding_window=128)

    assert _get_layer_sliding_window(config, 0) == 128
    assert _get_layer_sliding_window(config, 1) is None
