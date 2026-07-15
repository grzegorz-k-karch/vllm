from types import SimpleNamespace

import torch

import vllm.model_executor.models.gpt_oss as gpt_oss


def test_unquantized_loader_remaps_modular_routed_expert_names(monkeypatch):
    checkpoint_name = "layers.0.mlp.experts.w2_weight"
    parameter_name = "layers.0.mlp.experts.routed_experts.w2_weight"
    parameter = torch.nn.Parameter(torch.empty(2, 4, 3), requires_grad=False)
    loaded_weight = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)

    model = SimpleNamespace(
        config=SimpleNamespace(intermediate_size=3),
        parallel_config=SimpleNamespace(enable_expert_parallel=False),
        named_parameters=lambda: [(parameter_name, parameter)],
    )
    monkeypatch.setattr(gpt_oss, "is_pp_missing_parameter", lambda *_: False)
    monkeypatch.setattr(gpt_oss, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(
        gpt_oss,
        "get_dp_group",
        lambda: SimpleNamespace(world_size=1, rank_in_group=0),
    )
    monkeypatch.setattr(
        gpt_oss,
        "get_pcp_group",
        lambda: SimpleNamespace(world_size=1, rank_in_group=0),
    )
    monkeypatch.setattr(
        gpt_oss,
        "FusedMoEParallelConfig",
        SimpleNamespace(flatten_tp_across_dp_and_pcp=lambda **_: (1, 0)),
    )

    loaded = gpt_oss.GptOssModel._load_weights_other(
        model,
        ep_rank_end=2,
        ep_rank_start=0,
        heads_per_rank=1,
        head_start=0,
        weights=[(checkpoint_name, loaded_weight)],
        stacked_params_mapping=[],
    )

    assert loaded == {parameter_name}
    torch.testing.assert_close(parameter, loaded_weight.permute(0, 2, 1))


def test_unquantized_loader_uses_layer_local_sink_heads(monkeypatch):
    name = "layers.0.attn.sinks"
    parameter = torch.nn.Parameter(torch.empty(4), requires_grad=False)
    loaded_weight = torch.arange(8, dtype=torch.float32)
    model = SimpleNamespace(
        config=SimpleNamespace(intermediate_size=4),
        parallel_config=SimpleNamespace(enable_expert_parallel=False),
        named_parameters=lambda: [(name, parameter)],
    )
    monkeypatch.setattr(gpt_oss, "is_pp_missing_parameter", lambda *_: False)
    monkeypatch.setattr(gpt_oss, "get_tensor_model_parallel_world_size", lambda: 2)
    monkeypatch.setattr(gpt_oss, "get_tensor_model_parallel_rank", lambda: 1)
    monkeypatch.setattr(
        gpt_oss,
        "get_dp_group",
        lambda: SimpleNamespace(world_size=1, rank_in_group=0),
    )
    monkeypatch.setattr(
        gpt_oss,
        "get_pcp_group",
        lambda: SimpleNamespace(world_size=1, rank_in_group=0),
    )
    monkeypatch.setattr(
        gpt_oss,
        "FusedMoEParallelConfig",
        SimpleNamespace(flatten_tp_across_dp_and_pcp=lambda **_: (2, 1)),
    )

    loaded = gpt_oss.GptOssModel._load_weights_other(
        model,
        ep_rank_end=2,
        ep_rank_start=0,
        heads_per_rank=8,
        head_start=8,
        weights=[(name, loaded_weight)],
        stacked_params_mapping=[],
    )

    assert loaded == {name}
    torch.testing.assert_close(parameter, loaded_weight[4:])
