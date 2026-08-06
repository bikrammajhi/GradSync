import torch
import torch.nn as nn
import torch.nn.functional as F
from flash_attn.flash_attn_interface import flash_attn_func
from flash_attn.layers.rotary import apply_rotary_emb
from flash_attn.ops.triton.layer_norm import layer_norm_fn

def flash_attention(query, key, value, causal=True):
    # [Batch_size, Seq_len, Num_heads, Head_dim] -> [Batch_size, Num_heads, Seq_len, Head_dim]
    query = query.permute(0, 2, 1, 3).contiguous()
    key = key.permute(0, 2, 1, 3).contiguous()
    value = value.permute(0, 2, 1, 3).contiguous()

    # Apply flash attention
    output = flash_attn_func(query, key, value, causal=causal)

    return output

def get_cos_sin(seq_len, head_dim, base=500000.0):
    assert head_dim % 2 == 0, "Head dimension must be even for rotary embeddings."
    # freq should be calculated on CPU to match transformer implementations
    theta = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.int64).float().to('cpu') / head_dim))
    dtype = torch.bfloat16
    device = torch.device("cuda")
    position = torch.arange(seq_len).to(device).unsqueeze(1).to(dtype) # [seq_len, 1]
    # To match transformer implementations, we need to compute the outer product of position and theta on GPU
    theta = theta.to(device)
    return torch.cos(position.float() * theta.float()).to(dtype).repeat(1,2), 
           torch.sin(position.float() * theta.float()).to(dtype).repeat(1,2) # [seq_len, head_dim], [seq_len, head_dim]

class TritonRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.register_parameter("bias", None)  # No bias for RMSNorm
        
    def forward(self, hidden_states, residual=None, dropout_p=0.0, prenorm=False, residual_in_fp32=False, return_dropout_mask=False):
        
        return layer_norm_fn(
            hidden_states, self.weight,
            None, residual=residual, 
            eps=self.eps, dropout_p=dropout_p, prenorm=prenorm,
            residual_in_fp32=residual_in_fp32, is_rms_norm=True, 
            return_dropout_mask=return_dropout_mask
        )

class Attention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_values = config.num_key_value_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_local_heads = config.num_attention_heads
        self.num_local_kv_heads = config.num_key_value_heads
        
        self.wq = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(self.hidden_size, self.num_key_values * self.head_dim, bias=False)
        self.wv = nn.Linear(self.hidden_size, self.num_key_values * self.head_dim, bias=False)
        self.wo = nn.Linear(self.hidden_size, self.hidden_size, bias=False)  
        self.layer_idx = layer_idx
        
    def forward(self, x, cos, sin, attention_mask=None, position_ids=None):
        batch_size, seq_len, hidden_dim = x.size()
        q = self.wq(x)  # [Batch_size, Seq_len, Num_heads * Head_dim]
        k = self.wk(x)  # [Batch_size, Seq_len, Num_key_values * Head_dim]
        v = self.wv(x)  # [Batch_size, Seq_len, Num_key_values * Head_dim]

        q = q.view(batch_size, seq_len, self.num_local_heads, self.head_dim)    # [Batch_size, Seq_len, Num_heads, Head_dim]
        k = k.view(batch_size, seq_len, self.num_local_kv_heads, self.head_dim) # [Batch_size, Seq_len, Num_key_values, Head_dim]
        v = v.view(batch_size, seq_len, self.num_local_kv_heads, self.head_dim) # [Batch_size, Seq_len, Num_key_values, Head_dim]
        
        # Apply rotary embeddings
        q = apply_rotary_emb(q,cos[:, :self.head_dim // 2], sin[:, :self.head_dim // 2], interleaved=False)
        k = apply_rotary_emb(k,cos[:, :self.head_dim // 2], sin[:, :self.head_dim // 2], interleaved=False)
        
        q = q.transpose(1, 2).contiguous()  # [Batch_size, Num_heads, Seq_len, Head_dim]
        k = k.transpose(1, 2).contiguous()  # [Batch_size, Num_key_values, Seq_len, Head_dim]
        v = v.view(batch_size, seq_len, self.num_local_kv_heads, self.head_dim).transpose(1, 2).contiguous()  # [Batch_size, Num_key_values, Seq_len, Head_dim]
        
        causal = True if q.size(2) == k.size(2) else False  # Causal if query and key have the same sequence length
        
        out = flash_attention(q, k, v, causal=causal)  # [Batch_size, Num_heads, Seq_len, Head_dim]
        out = out.reshape(batch_size, seq_len, self.num_local_heads * self.head_dim) # [Batch_size, Seq_len, Num_heads * Head_dim]
        out = self.wo(out)  # [Batch_size, Seq_len, Hidden_size]
        return out

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
    
    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

class DecoderLayer(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.input_layernorm = TritonRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = TritonRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attention = Attention(config, layer_idx)
        self.mlp = MLP(config)
        self.layer_idx = layer_idx
        head_dim = config.hidden_size // config.num_attention_heads
        self.cos, self.sin = get_cos_sin(config.max_position_embeddings, head_dim, base=config.rope_theta) # [max_position_embeddings, head_dim]
        
    def forward(self, x, attention_mask=None, position_ids=None):
        cos, sin = self.cos, self.sin
        x = x + self.attention(self.input_layernorm(x), cos, sin, attention_mask, position_ids)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x

class Llama(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.hidden_size % config.num_attention_heads == 0, "hidden_size must be divisible by num_attention_heads"
        assert config.num_attention_heads % config.num_key_value_heads == 0, "num_attention_heads must be divisible by num_key_value_heads"
        
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_values = config.num_key_value_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.num_layers = config.num_hidden_layers
        self.model_config = config
        
        # modules
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.decode_layers = nn.ModuleList([DecoderLayer(config, layer_idx) for layer_idx in range(config.num_layers)])
        self.final_proj = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.final_layernorm = TritonRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
    def forward(self, input_ids, attention_mask=None, position_ids: torch.Tensor=None):
        x = self.embedding(input_ids)  # [Batch_size, Seq_len, Hidden_size]
        for layer in self.decode_layers:
            x = layer(x)               # [Batch_size, Seq_len, Hidden_size]
        x = self.final_layernorm(x)
        logits = self.final_proj(x)
        return logits                  # [Batch_size, Seq_len, Vocab_size]

