# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# RL-specific parallelize function for vLLM-compatible TP plan.
# Applies tensor parallelism so vLLM's flash attention kernels receive
# local tensors (DTensor unwrap/wrap is handled in the attention modules),
# and optionally FSDP for the trainer.

import logging

import torch
import torch.nn as nn

from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import Replicate, Shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    parallelize_module,
    PrepareModuleInput,
    RowwiseParallel,
    SequenceParallel,
)

from torchtitan.config import ParallelismConfig
from torchtitan.config.configs import CompileConfig
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.tensor_parallel import NoParallel

logger = logging.getLogger(__name__)


def _apply_compile_with_dynamic_dims(
    model: nn.Module,
    compile_config: CompileConfig,
    dynamic_dims: list[tuple[int, int]],
) -> None:
    """Compile each TransformerBlock with specified dynamic dimensions.

    Like ``apply_compile_dense``, but wraps each block's compiled forward with
    ``mark_dynamic`` on the specified (arg_index, dim) pairs. This avoids
    recompiles when those dimensions vary between calls (e.g. sequence length
    in RL training).

    Uses the same wrapper pattern as MoE expert-parallel in
    ``torchtitan/distributed/compile.py``: ``mark_dynamic`` runs in eager mode,
    then the compiled forward sees the dynamic marks.

    Args:
        dynamic_dims: List of (arg_index, dim) tuples. Each entry marks
            ``args[arg_index]`` dimension ``dim`` as dynamic via
            ``torch._dynamo.mark_dynamic``, provided the arg exists and
            is not None.
    """
    torch._dynamo.config.skip_fwd_side_effects_in_bwd_under_checkpoint = (
        True  # pyrefly: ignore [bad-assignment]
    )

    # pyrefly: ignore [missing-attribute]
    for layer_id, block in model.layers.named_children():
        compiled_fwd = torch.compile(
            block.forward, backend=compile_config.backend, fullgraph=True
        )

        def _make_dynamic_fwd(cf, dims):
            def forward(*args, **kwargs):
                for arg_idx, dim in dims:
                    if arg_idx < len(args) and args[arg_idx] is not None:
                        torch._dynamo.mark_dynamic(args[arg_idx], dim)
                return cf(*args, **kwargs)

            return forward

        block.forward = _make_dynamic_fwd(compiled_fwd, dynamic_dims)
        # pyrefly: ignore [missing-attribute]
        model.layers.register_module(layer_id, block)

    logger.info("Compiling each TransformerBlock with dynamic dims")


def parallelize_qwen3(
    model: nn.Module,
    *,
    parallel_dims: ParallelDims,
    parallelism: ParallelismConfig,
    compile_config: CompileConfig | None = None,
    has_position_id: bool = False,
):
    """
    Apply tensor parallelism to the Qwen3 dense model for RL training/inference.

    NOTE: The function signature is intentionally simpler than core torchtitan's
    parallelize_qwen3 — it only accepts the configs needed for TP.
    TODO: Change to core torchtitan's Qwen3 parallel plan when full DTensor is ready

    Args:
        compile_config: If provided and enabled, applies per-layer torch.compile
            after TP (matching the pattern in torchtitan/models/llama3/parallelize.py).
        has_position_id: Whether position IDs are passed as an explicit argument
            to the attention module. True for vLLM inference (generator),
            False for training (trainer).
    """

    if parallel_dims.tp_enabled:
        tp_mesh = parallel_dims.get_mesh("tp")
        apply_non_moe_tp(
            model,
            tp_mesh,
            enable_loss_parallel=not parallelism.disable_loss_parallel,
            enable_float8_tensorwise_tp=False,
            enable_async_tp=parallelism.enable_async_tensor_parallel,
            enable_sp=parallelism.enable_sequence_parallel,
            has_position_id=has_position_id,
        )

    if (
        compile_config is not None
        and compile_config.enable
        and "model" in compile_config.components
    ):
        if parallel_dims.tp_enabled:
            # Eagerly init local_map on inner_attention so dynamo never sees
            # the _local_map_fn=None lazy-init branch (avoids a recompile).
            # Derive qkv placements from the parallelized wq weight:
            # ColwiseParallel shards the output-features dim (weight dim 0).
            # After attention reshape [bs, seq, n_heads, head_dim], the sharded
            # output features map to the n_heads dim (index 2).
            tp_mesh = parallel_dims.get_mesh("tp")
            first_block = next(
                iter(model.layers.values())
            )  # pyrefly: ignore [not-callable]
            qkv_placements = tuple(
                Shard(2) if isinstance(p, Shard) else p
                for p in first_block.attention.wq.weight.placements
            )
            for block in model.layers.values():  # pyrefly: ignore [not-callable]
                block.attention.inner_attention.init_local_map(qkv_placements, tp_mesh)
        # Mark seq_len (dim 1) as dynamic on hidden states (arg 0) and
        # positions (arg 3) to avoid per-episode recompiles from variable
        # sequence lengths in RL training.
        _apply_compile_with_dynamic_dims(
            model, compile_config, dynamic_dims=[(0, 1), (3, 1)]
        )

    return model


def apply_non_moe_tp(
    model: nn.Module,
    tp_mesh: DeviceMesh,
    enable_loss_parallel: bool,
    enable_float8_tensorwise_tp: bool,
    enable_async_tp: bool,
    enable_sp: bool = True,
    has_position_id: bool = False,
):
    """Apply tensor parallelism to the Qwen3 dense model.

    This is a temporary TP plan used while we resolve composability issues in the
    main torchtitan codebase. Once DTensor is fully supported across the TP
    region, this separate plan should be removed.
    """

    sp_layout = Shard(1) if enable_sp else Replicate()
    norm_plan = SequenceParallel(use_local_output=False) if enable_sp else NoParallel()

    parallelize_module(
        model,
        tp_mesh,
        {
            "tok_embeddings": RowwiseParallel(
                input_layouts=Replicate(),
                output_layouts=sp_layout,
                use_local_output=False,
            ),
            "norm": norm_plan,
            "output": ColwiseParallel(
                input_layouts=sp_layout,
                output_layouts=Replicate(),
                use_local_output=True,  # return logits and plain tensor
            ),
        },
    )

    # Apply tensor + sequence parallelism to every transformer block
    # NOTE: At the cost of model code change, we can accelerate Sequence Parallel
    #       by folding (and unfolding) the batch dimension and the sequence dimension.
    #       Examples can be found at https://github.com/pytorch/torchtitan/pull/437
    positions_layout = Replicate() if has_position_id else None

    # pyrefly: ignore [not-callable]
    for transformer_block in model.layers.values():
        qk_norm_plan = SequenceParallel(sequence_dim=2)
        layer_plan = {
            "attention_norm": norm_plan,
            "attention": PrepareModuleInput(
                input_layouts=(
                    sp_layout,
                    Replicate(),
                    None,
                    positions_layout,
                ),
                desired_input_layouts=(
                    Replicate(),
                    Replicate(),
                    None,
                    positions_layout,
                ),
            ),
            "attention.wq": ColwiseParallel(use_local_output=False),
            "attention.wk": ColwiseParallel(use_local_output=False),
            "attention.wv": ColwiseParallel(use_local_output=False),
            "attention.q_norm": qk_norm_plan,
            "attention.k_norm": qk_norm_plan,
            "attention.wo": RowwiseParallel(
                output_layouts=sp_layout,
                use_local_output=False,
            ),
            "ffn_norm": norm_plan,
        }

        # pyrefly: ignore [missing-attribute]
        if not transformer_block.moe_enabled:
            layer_plan.update(
                {
                    "feed_forward": PrepareModuleInput(
                        input_layouts=(sp_layout,),
                        desired_input_layouts=(Replicate(),),
                    ),
                    "feed_forward.w1": ColwiseParallel(use_local_output=False),
                    "feed_forward.w2": RowwiseParallel(
                        output_layouts=sp_layout,
                        use_local_output=False,
                    ),
                    "feed_forward.w3": ColwiseParallel(use_local_output=False),
                }
            )
        else:
            raise ValueError(
                "Running vLLM inference with torchtitan Qwen3 MoE model is not supported yet."
            )

        parallelize_module(
            # pyrefly: ignore [bad-argument-type]
            module=transformer_block,
            device_mesh=tp_mesh,
            # pyrefly: ignore [bad-argument-type]
            parallelize_plan=layer_plan,
        )
