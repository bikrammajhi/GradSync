# GradSync — From-Scratch Distributed Training

GradSync is a compact, educational distributed-training framework for LLMs built
directly on `torch.distributed`. The parallel primitives are implemented by hand
— no Megatron, no DeepSpeed — so you can read exactly what a tensor-parallel
training step does under the hood.

**Status:** tensor parallelism (TP) is implemented and smoke-tested on 2×A100.
Data- and pipeline-parallelism are planned (see [Roadmap](#roadmap)).

The current build ships:

- A Llama-style model (`model.py`): flash-attention, rotary embeddings, GQA,
  Triton RMSNorm.
- A tensor-parallel layer suite (`tensor_parallel.py`):
  `ColumnParallelLinear`, `RowParallelLinear`, `VocabParallelEmbedding`, and the
  `Copy`/`Reduce`/`Gather` communication primitives behind them.
- A 3D process-group grid (`process_group_manager.py`) that carves
  `dp × pp × tp` groups out of the global world. The TP groups are exercised
  today; the DP/PP groups are created to prepare the next steps.
- A chunked-text dataloader (`dataloader.py`) that tokenizes a Hugging Face
  corpus into fixed-length blocks (full-dataset tokenization — not streaming).
- A cloud runner (`modal_train.py`) that launches multi-GPU runs on
  [Modal](https://modal.com) with zero local GPU requirements.

## Quick Start

One-step smoke test (the config that has been run and validated), on Modal
2 × A100 with TP=2:

```bash
modal run --detach modal_train.py --tp-size 2 \
  --max-tokens 64 --seq-len 64 --micro-batch-size 1 \
  --gradient-accumulation-steps 1 --num-hidden-layers 8 --num-proc 8
```

Or, locally, the same run with `torchrun --nproc_per_node 2` and the same
arguments on `--tp_size`, `--max_tokens`, etc. Both terminate after exactly one
optimizer step (64 tokens).

## Results — TP smoke test

Run on **Modal (2 × A100-40GB)**, TP=2 · DP=1 · PP=1, 2026-08-08:

| Metric | Value |
|---|---|
| Model | TinyLlama/TinyLlama_v1.1 (8 decoder layers, 32 heads, 4 KV heads, bf16) |
| Sequence length | 64 |
| Micro-batch × grad-accum | 1 × 1 (64 tokens/step) |
| Steps run | 1 (smoke test) |
| **Loss** | **10.5625** |
| **GPU memory / GPU** | **2.56 GB** |

A randomly initialized 32k-vocab model has expected loss `ln(32000) ≈ 10.37`;
the measured 10.56 is consistent with that. The single step exercised the whole
TP machinery — `Copy`/`Reduce`/`Gather` collectives, sharded embeddings, and a
full backward + optimizer step — and completed with no errors; it is a
smoke test, not a correctness proof.

[wandb run](https://wandb.ai/iiserkbikram/picotron_tutorial/runs/h68lwer2)

## Roadmap

- [x] Model + process-group grid + chunked dataloader
- [x] Tensor parallelism — smoke-tested on 2×A100
- [ ] Data parallelism — naive and bucketed gradient all-reduce
- [ ] Pipeline parallelism — 1F1B / AFAB schedules
- [ ] 3D-parallel run + convergence-curve validation

## References

- [Picotron tutorial repo](https://github.com/huggingface/picotron_tutorial) — the
  canonical "from-scratch" series this repo builds on
- [Modal](https://modal.com) — the managed GPU runner used for validation
- [wandb run](https://wandb.ai/iiserkbikram/picotron_tutorial/runs/h68lwer2) — the
  TP smoke test

*Built for understanding — every communication call lands on a single
`torch.distributed` collective; nothing is hidden behind a framework.*