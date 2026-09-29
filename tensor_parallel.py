import math
from typing import Optional, List
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
import process_group_manager as pgm


# =============================================================================
#                        TP COMMUNICATION PRIMITIVES
# =============================================================================

def split_tensor_along_last_dim(
    tensor: torch.Tensor,
    num_partitions: int
) -> tuple[torch.Tensor, ...]:
    """Split a tensor along its last dimension into num_partitions chunks.

    Args:
        tensor: Input tensor whose last dim is divisible by num_partitions.
        num_partitions: Number of equal splits along the last dimension.

    Returns:
        Tuple of tensors, each with last_dim_size = tensor.size(-1) // num_partitions.
    """
    last_dim = tensor.dim() - 1
    assert tensor.size(last_dim) % num_partitions == 0, (
        f"tensor.size({last_dim})={tensor.size(last_dim)} is not divisible "
        f"by num_partitions={num_partitions}"
    )
    last_dim_size = tensor.size(last_dim) // num_partitions
    return torch.split(tensor, last_dim_size, dim=last_dim)


class Reduce(torch.autograd.Function):
    """All-reduce SUM primitive: aggregates sharded partial sums into a full tensor.

    Used after RowParallelLinear and VocabParallelEmbedding.
    Forward: in-place all-reduce SUM over the TP group.
    Backward: identity (each rank's local shard receives its gradient naturally).

    Note: mutates the input in-place via all_reduce.
    """

    @staticmethod
    def forward(ctx, input: torch.Tensor) -> torch.Tensor:
        if pgm.process_group_manager.tp_world_size == 1:
            return input
        dist.all_reduce(input, op=dist.ReduceOp.SUM, group=pgm.process_group_manager.tp_group)
        return input

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        return grad_output


class Gather(torch.autograd.Function):
    """All-gather primitive: concatenates column-parallel shards along the last dim.

    Used where the consumer needs the full tensor (e.g. the final logits head
    with gather_output=True). Prefer Column->Row zero-copy handoff when possible,
    since all-gather moves N x more data than all-reduce.
    Forward: all-gather + concat along last dim.
    Backward: split grad along last dim, return this rank's shard (no comms).
    """

    @staticmethod
    def forward(ctx, input: torch.Tensor) -> torch.Tensor:
        if pgm.process_group_manager.tp_world_size == 1:
            return input
        last_dim = input.dim() - 1
        # Need contiguous tensors for collective
        input = input.contiguous()
        tensor_list: List[torch.Tensor] = [
            torch.empty_like(input)
            for _ in range(pgm.process_group_manager.tp_world_size)
        ]
        tensor_list[pgm.process_group_manager.tp_rank] = input
        dist.all_gather(tensor_list, input, group=pgm.process_group_manager.tp_group)
        output = torch.cat(tensor_list, dim=last_dim).contiguous()
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        if pgm.process_group_manager.tp_world_size == 1:
            return grad_output
        # Split gradient according to TP size along last dim
        chunks = split_tensor_along_last_dim(
            grad_output, pgm.process_group_manager.tp_world_size
        )
        return chunks[pgm.process_group_manager.tp_rank].contiguous()


class Copy(torch.autograd.Function):
    """Input-replication primitive bridging replicated inputs to column-parallel layers.

    Forward: identity (input is already replicated on every rank).
    Backward: all-reduce SUM of partial input gradients.
    Copy and Reduce are autograd inverses of each other.
    """

    @staticmethod
    def forward(ctx, input: torch.Tensor) -> torch.Tensor:
        return input

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        if pgm.process_group_manager.tp_world_size == 1:
            return grad_output
        dist.all_reduce(grad_output, op=dist.ReduceOp.SUM, group=pgm.process_group_manager.tp_group)
        return grad_output


# =============================================================================
#                     TENSOR PARALLEL MODULES
# =============================================================================

class ColumnParallelLinear(nn.Module):
    """Linear layer sharded along the output dimension: W = [W_0; ...; W_{N-1}].

    Used for Q/K/V, MLP up/gate projections, and the final logits head
    (with gather_output=True). With gather_output=False (default) the sharded
    output feeds directly into a RowParallelLinear with zero communication.
    Weights are initialized from full fan-in statistics, then sliced per rank.
    Memory per rank: (out/N) x in.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool,
        gather_output: bool = False,
    ):
        super().__init__()

        self.tp_world_size = pgm.process_group_manager.tp_world_size
        self.tp_rank = pgm.process_group_manager.tp_rank

        self.in_features = in_features
        self.out_features = out_features

        assert out_features % self.tp_world_size == 0, (
            f"out_features ({out_features}) must be divisible by "
            f"tp_world_size ({self.tp_world_size})"
        )
        self.output_size_per_partition = out_features // self.tp_world_size
        self.gather_output = gather_output

        # Note: F.linear performs X @ W^T + b, so weight shape is [out, in]
        self.weight = nn.Parameter(
            torch.Tensor(self.output_size_per_partition, self.in_features)
        )
        if bias:
            self.bias = nn.Parameter(torch.Tensor(self.output_size_per_partition))
            with torch.no_grad():
                self.bias.zero_()
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize with full fan-in statistics, then slice for this rank."""
        if self.tp_world_size == 1:
            k = 1.0 / self.weight.size(1)
            bound = math.sqrt(k)
            torch.nn.init.uniform_(self.weight, -bound, bound)
            return

        # Initialize master weight with GLOBAL dimensions for correct variance
        master_weight = torch.empty(
            self.out_features, self.in_features,
            dtype=self.weight.dtype, requires_grad=False,
        )
        k = 1.0 / master_weight.size(1)
        bound = math.sqrt(k)
        torch.nn.init.uniform_(master_weight, -bound, bound)

        # Slice this rank's partition
        weight_list = torch.split(master_weight, self.output_size_per_partition, dim=0)
        self.weight.data.copy_(weight_list[self.tp_rank])

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Ensure full input is available; backward aggregates partial ∇X
        input_parallel = Copy.apply(input)
        # Y_i = X · W_i^T + b_i
        output = F.linear(input_parallel, self.weight, self.bias)
        if self.gather_output:
            output = Gather.apply(output)
        return output


class RowParallelLinear(nn.Module):
    """Linear layer sharded along the input dimension: Y = sum_i(X_i @ W_i^T) + b.

    Used for attention out_proj and MLP down_proj. Expects the sharded input
    produced by a preceding ColumnParallelLinear (zero-copy handoff). The bias
    is added AFTER the all-reduce so it isn't counted N times.
    Memory per rank: out x (in/N), plus a full bias.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool):
        super().__init__()

        self.tp_world_size = pgm.process_group_manager.tp_world_size
        self.tp_rank = pgm.process_group_manager.tp_rank

        self.in_features = in_features
        self.out_features = out_features

        assert in_features % self.tp_world_size == 0, (
            f"in_features ({in_features}) must be divisible by "
            f"tp_world_size ({self.tp_world_size})"
        )
        self.input_size_per_partition = in_features // self.tp_world_size

        self.weight = nn.Parameter(
            torch.Tensor(self.out_features, self.input_size_per_partition)
        )
        if bias:
            self.bias = nn.Parameter(torch.Tensor(self.out_features))
            # Always initialize bias to zero to prevent double-counting
            with torch.no_grad():
                self.bias.zero_()
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize with full fan-in statistics, then slice for this rank."""
        if self.tp_world_size == 1:
            k = 1.0 / self.weight.size(1)
            bound = math.sqrt(k)
            torch.nn.init.uniform_(self.weight, -bound, bound)
            return

        # Initialize master weight with GLOBAL dimensions for correct variance
        master_weight = torch.empty(
            self.out_features, self.in_features,
            dtype=self.weight.dtype, requires_grad=False,
        )
        k = 1.0 / master_weight.size(1)
        bound = math.sqrt(k)
        torch.nn.init.uniform_(master_weight, -bound, bound)

        # Slice this rank's partition along input dimension
        weight_list = torch.split(master_weight, self.input_size_per_partition, dim=1)
        self.weight.data.copy_(weight_list[self.tp_rank])

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Y_i = X_i · W_i^T (no bias yet)
        output_parallel = F.linear(input, self.weight)
        # AllReduce SUM to reconstruct full Y
        output = Reduce.apply(output_parallel)
        # Add bias AFTER AllReduce to avoid summing it N times
        if self.bias is not None:
            output = output + self.bias
        return output


class VocabParallelEmbedding(nn.Module):
    """Embedding table sharded along the vocab dimension; rank i owns rows [start_i, end_i).

    Forward: mask out-of-range tokens, look up local rows, zero invalid positions,
    then all-reduce SUM (exactly one rank contributes per token). Returns the full
    embedding, identical to a non-parallel nn.Embedding. Gradients for non-owned
    tokens are naturally zero, so no explicit grad masking is needed.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
        max_norm: Optional[float] = None,
        norm_type: float = 2.0,
        scale_grad_by_freq: bool = False,
        sparse: bool = False,
    ):
        super().__init__()

        self.tp_world_size = pgm.process_group_manager.tp_world_size
        self.tp_rank = pgm.process_group_manager.tp_rank

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.sparse = sparse

        # Compute this rank's vocabulary slice
        self.vocab_start_index, self.vocab_end_index = (
            self._vocab_range_from_global_vocab_size(
                self.num_embeddings,
                pgm.process_group_manager.tp_rank,
                pgm.process_group_manager.tp_world_size,
            )
        )
        self.num_embeddings_per_partition = (
            self.vocab_end_index - self.vocab_start_index
        )

        self.weight = nn.Parameter(
            torch.Tensor(self.num_embeddings_per_partition, self.embedding_dim)
        )
        self.reset_parameters()

    @staticmethod
    def _vocab_range_from_global_vocab_size(
        global_vocab_size: int, rank: int, world_size: int
    ) -> tuple[int, int]:
        """Compute [start, end) index range for a given rank's vocab partition."""
        assert global_vocab_size % world_size == 0, (
            f"global_vocab_size ({global_vocab_size}) is not divisible "
            f"by world_size ({world_size})"
        )
        per_partition_vocab_size = global_vocab_size // world_size
        index_start = rank * per_partition_vocab_size
        index_end = index_start + per_partition_vocab_size
        return index_start, index_end

    def reset_parameters(self):
        """Initialize with global stats, then slice for this rank."""
        if self.tp_world_size == 1:
            torch.nn.init.normal_(self.weight, mean=0.0, std=1.0)
            return

        # Initialize master embedding with GLOBAL dimensions
        master_weight = torch.empty(
            self.num_embeddings, self.embedding_dim,
            dtype=self.weight.dtype, requires_grad=False,
        )
        torch.nn.init.normal_(master_weight, mean=0.0, std=1.0)

        # Slice this rank's partition
        weight_list = torch.split(
            master_weight, self.num_embeddings_per_partition, dim=0
        )
        self.weight.data.copy_(weight_list[self.tp_rank])

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Range-masked lookup + all-reduce SUM to recover the global embedding."""
        # ① Build validity mask
        input_mask = (input < self.vocab_start_index) | (
            input >= self.vocab_end_index
        )

        # ② Shift to local indices; clamp invalid to safe index 0
        masked_input = input.clone() - self.vocab_start_index
        masked_input[input_mask] = 0

        # ③ Local embedding lookup
        output_parallel = F.embedding(
            masked_input,
            self.weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )

        # ④ Zero out invalid positions so they don't pollute AllReduce
        output_parallel[input_mask, :] = 0.0

        # ⑤ AllReduce SUM — exactly one rank contributes per token
        output = Reduce.apply(output_parallel)
        return output


# =============================================================================
#                    MODEL CONVERSION UTILITY
# =============================================================================

def apply_tensor_parallel(model: nn.Module) -> nn.Module:
    """Convert a standard transformer model to tensor-parallel execution.

    Replaces linear layers and embeddings with their TP equivalents according
    to the canonical Megatron-LM sharding pattern:
        • Q/K/V/up/gate projections → ColumnParallel
        • out/down projections      → RowParallel
        • Input embedding           → VocabParallelEmbedding
        • Final logits projection   → ColumnParallel(gather_output=True)

    Args:
        model: Standard (non-parallel) transformer model with attributes
               `decoder_layers`, `embedding`, and `final_proj`.

    Returns:
        Model with TP modules substituted in-place.
    """

    def _replace_module(
        _module: nn.Module,
        _linear_proj_name: str,
        _style: str,
        args: Optional[dict] = None,
    ):
        if args is None:
            args = {}
        assert _style in ("column", "row", "vocab"), f"Unknown TP style: {_style}"

        linear_layer = getattr(_module, _linear_proj_name)

        if _style == "column":
            new_layer = ColumnParallelLinear(
                in_features=linear_layer.in_features,
                out_features=linear_layer.out_features,
                bias=linear_layer.bias is not None,
                gather_output=args.get("gather_output", False),
            )
        elif _style == "row":
            new_layer = RowParallelLinear(
                in_features=linear_layer.in_features,
                out_features=linear_layer.out_features,
                bias=linear_layer.bias is not None,
            )
        else:  # vocab
            new_layer = VocabParallelEmbedding(
                num_embeddings=linear_layer.num_embeddings,
                embedding_dim=linear_layer.embedding_dim,
            )

        setattr(_module, _linear_proj_name, new_layer)

    # Canonical Megatron-LM TP mapping
    module_linear_name_style_mapping = [
        ("attention", "q_proj", "column"),
        ("attention", "k_proj", "column"),
        ("attention", "v_proj", "column"),
        ("attention", "out_proj", "row"),
        ("mlp", "up_proj", "column"),
        ("mlp", "gate_proj", "column"),
        ("mlp", "down_proj", "row"),
    ]

    for layer in model.decoder_layers:
        for module_name, linear_proj_name, style in module_linear_name_style_mapping:
            _replace_module(getattr(layer, module_name), linear_proj_name, style)

    _replace_module(model, "embedding", "vocab")
    _replace_module(model, "final_proj", "column", args={"gather_output": True})

    return model