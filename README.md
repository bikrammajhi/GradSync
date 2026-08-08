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

## Repository Layout

| File                      | Purpose                                                        |
|---------------------------|----------------------------------------------------------------|
| `model.py`                | Llama-style transformer (flash-attn, GQA, rotary embeddings)   |
| `tensor_parallel.py`      | TP primitives, parallel linears, and `apply_tensor_parallel`   |
| `process_group_manager.py`| DP × PP × TP grid and subgroup creation                         |
| `dataloader.py`           | Chunked-text dataset loader                                    |
| `train.py`                | `torchrun`-launched training entrypoint                         |
| `modal_train.py`          | Modal runner (TP2/TP4, parameterized config)                    |
| `utils.py`                | Seeding, locked printing, readable number formatting            |

## Quick Start

### One-step smoke test (the config that has been run and validated)

On Modal (2 × A100):

```bash
modal run --detach modal_train.py --tp-size 2 \
  --max-tokens 64 --seq-len 64 --micro-batch-size 1 \
  --gradient-accumulation-steps 1 --num-hidden-layers 8 --num-proc 8
```

Same run, on a local GPU box (`torchrun --nproc_per_node 2 ...`), same
arguments:

```bash
torchrun --nproc_per_node 2 train.py --tp_size 2 \
  --max_tokens 64 --seq_len 64 --micro_batch_size 1 \
  --gradient_accumulation_steps 1 --num_hidden_layers 8 --num_proc 8
```

Both terminate after exactly one optimizer step (64 tokens).

### Default production config (aimed at next)

```bash
modal run --detach modal_train.py          # TP4, seq_len 1024, mb 4, grad_acc 8
```

produces 32,768-token steps and stops after `4,096,000` tokens (~125 steps).
This larger sweep is what will be used for the convergence validation below.

## Configuration

| Argument | Default | Meaning |
|---|---|---|
| `--tp_size` | 1 | Tensor-parallel degree (sharded weights, all-reduce output) |
| `--dp_size` | 1 | Data-parallel degree — process groups created, gradient sync **not yet implemented** |
| `--pp_size` | 1 | Pipeline-parallel degree — process groups created, pipelining **not yet implemented** |
| `--micro_batch_size` × `--gradient_accumulation_steps` | 1 × 1 | Micro-batch per optimizer step |
| `--seq_len` | 32 | Context length (also caps `max_position_embeddings`) |
| `--max_tokens` | 1,000,000 | Stop condition — total tokens trained |
| `--model_name` | HuggingFaceTB/SmolLM-360M-Instruct | Base config + tokenizer (HuggingFace id) |

One optimizer step consumes `micro_batch_size × gradient_accumulation_steps ×
dp_world_size × seq_len` tokens; the training loop exits when that count reaches
`max_tokens`.

## Results — TP smoke test (first run)

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

Run: https://wandb.ai/iiserkbikram/picotron_tutorial/runs/h68lwer2

### Validation protocol (pending)

The sanity criterion for this framework is convergence parity: any parallel
scheme must land on the same loss curve as the single-GPU baseline. The figure
below is the **reference plot from the [picotron tutorial](https://github.com/huggingface/picotron_tutorial)** — the target this repo aims to reproduce.
Re-running the smoke test with larger budgets (e.g. the default config above)
should stay on that curve.

![Reference convergence plot from the picotron tutorial — the baseline this project aims to match](images/sanity_check.png)

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