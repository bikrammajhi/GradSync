"""Run the GradSync TP training on Modal.

Usage:
    modal run modal_train.py --tp-size 2 --max-tokens 204800 --seq-len 256 \\
        --micro-batch-size 2 --gradient-accumulation-steps 4 \\
        --num-hidden-layers 8 --num-proc 8

    Optional wandb logging:
    modal secret create gradsync-wandb WANDB_API_KEY=xxx
    GRADSYNC_WANDB_SECRET=gradsync-wandb modal run modal_train.py --tp-size 2 ...
"""
import os
import subprocess
import modal

app = modal.App("gradsync-tp-train")

FLASH_ATTN_WHEEL = (
    "https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.0/"
    "flash_attn-2.5.0%2Bcu122torch2.1cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"
)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.1.0", index_url="https://download.pytorch.org/whl/cu121")
    .pip_install(
        "numpy==1.26.4",
        "datasets==2.19.1",
        "transformers==4.41.1",
        "sentencepiece",
        "wandb",
    )
    .pip_install(FLASH_ATTN_WHEEL)
    .add_local_file("model.py", "/root/model.py")
    .add_local_file("train.py", "/root/train.py")
    .add_local_file("dataloader.py", "/root/dataloader.py")
    .add_local_file("process_group_manager.py", "/root/process_group_manager.py")
    .add_local_file("tensor_parallel.py", "/root/tensor_parallel.py")
    .add_local_file("utils.py", "/root/utils.py")
)

# Optional wandb secret. Set GRADSYNC_WANDB_SECRET to a Modal secret name holding
# WANDB_API_KEY to enable --use_wandb; unset (default) runs without wandb.
# NOTE: Secret.from_name resolves lazily at deploy time, so this must be gated
# on an env var — a try/except around from_name cannot catch a missing secret.
_WANDB_SECRET_NAME = os.environ.get("GRADSYNC_WANDB_SECRET", "")
_WANDB_SECRETS = (
    [modal.Secret.from_name(_WANDB_SECRET_NAME)] if _WANDB_SECRET_NAME else []
)


def _build_cmd(
    tp_size,
    max_tokens,
    seq_len,
    micro_batch_size,
    gradient_accumulation_steps,
    num_hidden_layers,
    num_proc,
):
    cmd = [
        "torchrun",
        "--nproc_per_node", str(tp_size),
        "train.py",
        "--tp_size", str(tp_size),
        "--micro_batch_size", str(micro_batch_size),
        "--gradient_accumulation_steps", str(gradient_accumulation_steps),
        "--seq_len", str(seq_len),
        "--max_tokens", str(max_tokens),
        "--num_proc", str(num_proc),
        "--model_name", "TinyLlama/TinyLlama_v1.1",
        "--num_hidden_layers", str(num_hidden_layers),
        "--num_attention_heads", "32",
        "--num_key_value_heads", "4",
        "--run_name", "tp_1B",
    ]
    if _WANDB_SECRET_NAME:
        cmd.append("--use_wandb")
    return cmd


def _run(
    max_tokens: int = 4096000,
    seq_len: int = 1024,
    micro_batch_size: int = 4,
    gradient_accumulation_steps: int = 8,
    num_hidden_layers: int = 22,
    num_proc: int = 16,
    tp_size: int = 4,
):
    cmd = _build_cmd(
        max_tokens=max_tokens,
        seq_len=seq_len,
        micro_batch_size=micro_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_hidden_layers=num_hidden_layers,
        num_proc=num_proc,
        tp_size=tp_size,
    )
    subprocess.run(cmd, cwd="/root", check=True)


@app.function(
    image=image,
    gpu="a100-40gb:2",
    cpu=16.0,
    timeout=2 * 60 * 60,
    secrets=_WANDB_SECRETS,
)
def run_tp2(**kwargs):
    _run(tp_size=2, **kwargs)


@app.function(
    image=image,
    gpu="a100-40gb:4",
    cpu=32.0,
    timeout=2 * 60 * 60,
    secrets=_WANDB_SECRETS,
)
def run_tp4(**kwargs):
    _run(tp_size=4, **kwargs)


@app.local_entrypoint()
def main(
    tp_size: int = 4,
    max_tokens: int = 4096000,
    seq_len: int = 1024,
    micro_batch_size: int = 4,
    gradient_accumulation_steps: int = 8,
    num_hidden_layers: int = 22,
    num_proc: int = 16,
):
    kwargs = dict(
        max_tokens=max_tokens,
        seq_len=seq_len,
        micro_batch_size=micro_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_hidden_layers=num_hidden_layers,
        num_proc=num_proc,
    )
    if tp_size == 2:
        run_tp2.remote(**kwargs)
    elif tp_size == 4:
        run_tp4.remote(**kwargs)
    else:
        raise ValueError(f"Unsupported tp_size {tp_size}, use 2 or 4")