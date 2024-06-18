# from transformers.models.llama.modeling_flax_llama import LlamaModel
import torch
import jax.numpy as jnp

from model.llama import LlamaConfig, LlamaModel, create_llama_running
from codeGenerator.generator.models.Bitnet import (
    BitnetForCausalLM, 
    BitnetAttentionConfig,
    BitnetFFNConfig,
    BitnetConfig,
)

def convert_bit_linear(layer):
    norm = jnp.array(layer.norm.weight.data.numpy())
    gamma = 1.0 / torch.abs(layer.weight).mean().clamp_(min=layer.eps)
    return jnp.array((layer.weight * gamma).round().clamp_(-1, 1).detach()), gamma.item(), norm

dim:         int = 256
hidden_dim:  int = 512
n_layers:    int = 12
n_heads:     int = 16
n_kv_heads:  int = 4
vocab_size:  int = 32
seq_len:     int = 256

attention_config = BitnetAttentionConfig(kv_n_heads=n_kv_heads)
ffn_config = BitnetFFNConfig(ffn_hidden_size=hidden_dim)
config = BitnetConfig(
    d_model=dim,
    n_heads=n_heads,
    n_layers=n_layers,
    max_seq_len=seq_len,
    vocab_size=vocab_size,
    ffn_config=ffn_config,
    attn_config=attention_config,
)
config._attn_implementation = "eager"

model = BitnetForCausalLM(config=config)


run_config = LlamaConfig(
    dim=dim,
    hidden_dim=hidden_dim,
    n_layers=n_layers,
    n_heads=n_heads,
    n_kv_heads=n_kv_heads,
    vocab_size=vocab_size,
    seq_len=seq_len
)

print(model.model)

emb = model.model.embed_tokens
llama_head = jnp.array(model.lm_head.weight.data.numpy())
llama_norm = jnp.array(model.model.norm.weight.data.numpy())
# print(last_linear.shape)

llama_kv = [] 
llama_qo = []
llama_normal_attn = [] 
llama_gamma_attn = []
llama_mlp_ug = []
llama_mlp_d = []
llama_gamma_mlp = []
llama_normal_mlp_ug = []
llama_normal_mlp_d = []

for i in range(run_config.n_layers):
    q, q_gamma, q_norm = convert_bit_linear(model.model.layers[i].self_attn.q_proj)
    k, k_gamma, k_norm = convert_bit_linear(model.model.layers[i].self_attn.k_proj)
    v, v_gamma, v_norm = convert_bit_linear(model.model.layers[i].self_attn.v_proj)
    o, o_gamma, o_norm = convert_bit_linear(model.model.layers[i].self_attn.o_proj)
    llama_kv.append(jnp.stack([k, v]))
    llama_qo.append(jnp.stack([q, o]))
    llama_normal_attn.append(jnp.stack([q_norm, k_norm, v_norm, o_norm]))
    llama_gamma_attn.append(jnp.stack([q_gamma, k_gamma, v_gamma, o_gamma]))

    gate, gate_gamma, gate_norm = convert_bit_linear(model.model.layers[i].mlp.gate_proj)
    up, up_gamma, up_norm = convert_bit_linear(model.model.layers[i].mlp.up_proj)
    down, down_gamma, down_norm = convert_bit_linear(model.model.layers[i].mlp.down_proj)
    llama_mlp_ug.append(jnp.stack([up, gate]))
    llama_mlp_d.append(down)
    llama_gamma_mlp.append(jnp.stack([up_gamma, gate_gamma, down_gamma]))
    llama_normal_mlp_ug.append(jnp.stack([up_norm, gate_norm]))
    llama_normal_mlp_d.append(down_norm)


llama_kv = jnp.stack(llama_kv)
llama_qo = jnp.stack(llama_qo)
llama_normal_attn = jnp.stack(llama_normal_attn)
llama_gamma_attn = jnp.stack(llama_gamma_attn)
llama_mlp_ug = jnp.stack(llama_mlp_ug)
llama_mlp_d = jnp.stack(llama_mlp_d)
llama_gamma_mlp = jnp.stack(llama_gamma_mlp)
llama_normal_mlp_ug = jnp.stack(llama_normal_mlp_ug)
llama_normal_mlp_d = jnp.stack(llama_normal_mlp_d)

hidden_states = jnp.array(emb(torch.Tensor([0, 1, 2, 3]).type(torch.long)).detach().numpy())
run_llama = create_llama_running(run_config)
answer = run_llama(
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
)
res2 = model(torch.Tensor([[0, 1, 2, 3]]).type(torch.long))
print("a")