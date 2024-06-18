import jax
import numpy as np
from jax import numpy as jnp
from flax.linen.attention import dot_product_attention

from transformers import (
    AutoTokenizer,
)

class LlamaConfig:
    def __init__(
        self,
        dim:         int,
        hidden_dim:  int,
        n_layers:    int,
        n_heads:     int,
        n_kv_heads:  int,
        vocab_size:  int,
        seq_len:     int,
    ):
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.vocab_size = vocab_size
        self.seq_len = seq_len

class LlamaModel:
    def __init__(
        self, 
        config: LlamaConfig,
        model_path: str,
        tokenizer,
    ):
        self.config = config
        self.tokenizer = tokenizer
        self.model_path = model_path
        self.read_model_weights()
    
    def read_model_weights(self):
        pass


# @jax.jit
def create_sinusoidal_positions(num_pos, dim):
    inv_freq = 1.0 / (10000 ** (np.arange(0, dim, 2) / dim))
    freqs = np.einsum("i , j -> i j", np.arange(num_pos), inv_freq).astype("float32")

    emb = np.concatenate((freqs, freqs), axis=-1)
    out = np.concatenate((np.sin(emb)[:, None, :], np.cos(emb)[:, None, :]), axis=-1)
    return jnp.array(out[:, :, :num_pos])


@jax.jit
def rotate_half(tensor):
    """Rotates half the hidden dims of the input."""
    rotate_half_tensor = jnp.concatenate(
        (-tensor[..., tensor.shape[-1] // 2 :], tensor[..., : tensor.shape[-1] // 2]), axis=-1
    )
    return rotate_half_tensor


@jax.jit
def apply_rotary_pos_emb(tensor, sin_pos, cos_pos):
    return (tensor * cos_pos) + (rotate_half(tensor) * sin_pos)


@jax.jit
def rotary_emb(k, q, position_ids, sincos):
    sincos = sincos[position_ids]
    sin_pos, cos_pos = jnp.split(sincos, 2, axis=-1)

    k = apply_rotary_pos_emb(k, sin_pos, cos_pos)
    q = apply_rotary_pos_emb(q, sin_pos, cos_pos)

    k = jnp.asarray(k, dtype=jnp.float16)
    q = jnp.asarray(q, dtype=jnp.float16)
    return k, q


def create_split_heads(num_heads, head_dim):
    def split_heads(hidden_states):
        return hidden_states.reshape(hidden_states.shape[:1] + (num_heads, head_dim))
    return jax.jit(split_heads)


def create_merge_heads(embed_dim):
    def merge_heads(hidden_states):
        return hidden_states.reshape(hidden_states.shape[:1] + (embed_dim,))
    return jax.jit(merge_heads)


@jax.jit
def rms_norm(hidden_states, weight, rms_norm_eps=1e-6):
    variance = jnp.asarray(hidden_states, dtype=jnp.float16)
    variance = jnp.power(variance, 2)
    variance = variance.mean(-1, keepdims=True)
    hidden_states = hidden_states / jnp.sqrt(variance + rms_norm_eps)
    return weight * jnp.asarray(hidden_states, dtype=jnp.float16)


@jax.jit
def quantize_input(hidden_states, gamma, eps=1e-6):
    gamma = 127 / jnp.abs(hidden_states).max(axis=-1, keepdims=True).clip(min=eps)
    quantized_x = (hidden_states * gamma).round().clip(min=-128, max=127) / gamma
    return jnp.asarray(quantized_x, dtype=jnp.int8)


def create_llama_running(config: LlamaConfig):
    kv_head_dim = config.dim // config.n_kv_heads
    head_dim = config.dim // config.n_heads
    num_key_value_groups = config.n_heads // config.n_kv_heads
    sincos = create_sinusoidal_positions(config.seq_len, head_dim)
    mask = jnp.ones((config.seq_len, config.seq_len))
    mask = jnp.triu(mask, 1)
    mask = jnp.logical_not(mask)
    split_heads = create_split_heads(config.n_heads, head_dim)
    split_heads_kv = create_split_heads(config.n_kv_heads, head_dim)
    merge_heads = create_merge_heads(config.dim)

    def run_llama(
        llama_kv, 
        llama_qo, 
        llama_normal_attn, 
        llama_gamma_attn,
        llama_mlp_ug,
        llama_mlp_d,
        llama_gamma_mlp,
        llama_normal_mlp_ug,
        llama_normal_mlp_d,
        llama_head,
        llama_norm,
        hidden_states,
        kv_cache=None,
    ):
        sequence_length = hidden_states.shape[0]
        position_ids = jnp.arange(sequence_length)
        if kv_cache is None:
            use_cache = False
            kv_cache = [[] for i in range(config.n_layers)]
        else:
            use_cache = True
        
        for i in range(config.n_layers):
            q = rms_norm(hidden_states, llama_normal_attn[i, 0])
            q = quantize_input(q, llama_gamma_attn[i, 0])
            q = q @ llama_qo[i, 0].T
            q = split_heads(q)


            k = rms_norm(hidden_states, llama_normal_attn[i, 1])
            k = quantize_input(k, llama_gamma_attn[i, 1])
            k = k @ llama_kv[i, 0].T
            k = split_heads_kv(k)

            v = rms_norm(hidden_states, llama_normal_attn[i, 2])
            v = quantize_input(v, llama_gamma_attn[i, 2])
            v = v @ llama_kv[i, 1].T
            v = split_heads_kv(v)

            if use_cache:
                k = jnp.concat([kv_cache[i][0], k], axis=0)
                v = jnp.concat([kv_cache[i][1], v], axis=0)
                kv_cache[i] = (k, v)
            else:
                kv_cache[i] = (k, v)
            k, q = rotary_emb(k, q, position_ids, sincos)
            # if use_cache:
            #     pass
            # else:
            #     pass
            
            k = jnp.repeat(k, num_key_value_groups, axis=1)
            v = jnp.repeat(v, num_key_value_groups, axis=1)

            attn_output = dot_product_attention(
                q,
                k,
                v,
                mask=mask[:sequence_length, :sequence_length],
                deterministic=True,
                dtype=jnp.float16,
            )
            
            attn_output = merge_heads(attn_output)

            attn_output = rms_norm(attn_output, llama_normal_attn[i, 3])
            attn_output = quantize_input(attn_output, llama_gamma_attn[i, 3])
            attn_output = attn_output @ llama_qo[i, 1].T

            hidden_states += attn_output

            up = rms_norm(hidden_states, llama_normal_mlp_ug[i, 0])
            up = quantize_input(up, llama_gamma_mlp[i, 0])
            up = up @ llama_mlp_ug[i, 0].T

            gate = rms_norm(hidden_states, llama_normal_mlp_ug[i, 1])
            gate = quantize_input(gate, llama_gamma_mlp[i, 1])
            gate = gate @ llama_mlp_ug[i, 1].T
            gate = jax.nn.silu(gate)

            down = rms_norm(up * gate, llama_normal_mlp_d[i])
            down = quantize_input(down, llama_gamma_mlp[i, 2])
            down = down @ llama_mlp_d[i].T

            hidden_states += down

            
        
        hidden_states = rms_norm(hidden_states, llama_norm)
        hidden_states = hidden_states @ llama_head.T
        return hidden_states, kv_cache

    return jax.jit(run_llama)


