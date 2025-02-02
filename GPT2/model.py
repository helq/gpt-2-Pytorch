'''
    code by TaeHwan Jung(@graykode)
    Original Paper and repository here : https://github.com/openai/gpt-2
    GPT2 Pytorch Model : https://github.com/huggingface/pytorch-pretrained-BERT
'''
from __future__ import annotations

import copy
import torch
import math
import torch.nn as nn
from torch.nn.parameter import Parameter

from .config import GPT2Config

from typing import Optional, Tuple, List, Union


def gelu(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * x * (1 + torch.tanh(  # type: ignore
                          math.sqrt(2 / math.pi)
                          * (x + 0.044715 * torch.pow(x, 3))))  # type: ignore


class LayerNorm(nn.Module):  # type: ignore
    def __init__(self, hidden_size: int, eps: float = 1e-12) -> None:
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))  # type: ignore
        self.bias = nn.Parameter(torch.zeros(hidden_size))  # type: ignore
        self.variance_epsilon = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)  # type: ignore
        return self.weight * x + self.bias  # type: ignore


class Conv1D(nn.Module):  # type: ignore
    def __init__(self, nf: int, nx: int) -> None:
        super(Conv1D, self).__init__()
        self.nf = nf
        w = torch.empty(nx, nf)  # type: ignore
        nn.init.normal_(w, std=0.02)
        self.weight = Parameter(w)
        self.bias = Parameter(torch.zeros(nf))  # type: ignore

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)  # type: ignore
        x = x.view(*size_out)
        return x


class Attention(nn.Module):  # type: ignore
    def __init__(self, nx: int, n_ctx: int, config: GPT2Config, scale: bool = False) -> None:
        super(Attention, self).__init__()
        n_state = nx  # in Attention: n_state=768 (nx=n_embd)
        # [switch nx => n_state from Block to Attention to keep identical to TF implem]
        assert n_state % config.n_head == 0
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(n_ctx, n_ctx)).view(1, 1, n_ctx, n_ctx))  # type: ignore
        self.n_head = config.n_head
        self.split_size = n_state
        self.scale = scale
        self.c_attn = Conv1D(n_state * 3, nx)
        self.c_proj = Conv1D(n_state, nx)

    def _attn(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        w = torch.matmul(q, k)
        if self.scale:
            w = w / math.sqrt(v.size(-1))
        nd, ns = w.size(-2), w.size(-1)
        b = self.bias[:, :, ns-nd:ns, :ns]
        w = w * b - 1e10 * (1 - b)
        w = nn.Softmax(dim=-1)(w)  # type: ignore
        return torch.matmul(w, v)

    def merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_x_shape)  # in Tensorflow implem: fct merge_states

    def split_heads(self, x: torch.Tensor, k: bool = False) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
        x = x.view(*new_x_shape)  # in Tensorflow implem: fct split_states
        if k:
            return x.permute(0, 2, 3, 1)  # (batch, head, head_features, seq_length)
        else:
            return x.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

    def forward(
        self, x: torch.Tensor, layer_past: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.c_attn(x)
        query, key, value = x.split(self.split_size, dim=2)  # type: ignore
        query = self.split_heads(query)
        key = self.split_heads(key, k=True)
        value = self.split_heads(value)
        if layer_past is not None:
            # transpose back cf below
            past_key, past_value = layer_past[0].transpose(-2, -1), layer_past[1]

            key = torch.cat((past_key, key), dim=-1)  # type: ignore
            value = torch.cat((past_value, value), dim=-2)  # type: ignore

        # transpose to have same shapes for stacking
        present = torch.stack((key.transpose(-2, -1), value))

        a = self._attn(query, key, value)
        a = self.merge_heads(a)
        a = self.c_proj(a)
        return a, present


class MLP(nn.Module):  # type: ignore
    # in MLP: n_state=3072 (4 * n_embd)
    def __init__(self, n_state: int, config: GPT2Config) -> None:
        super(MLP, self).__init__()
        nx = config.n_embd
        self.c_fc = Conv1D(n_state, nx)
        self.c_proj = Conv1D(nx, n_state)
        self.act = gelu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.c_fc(x))
        h2 = self.c_proj(h)
        return h2  # type: ignore


class Block(nn.Module):  # type: ignore
    def __init__(self, n_ctx: int, config: GPT2Config, scale: bool = False) -> None:
        super(Block, self).__init__()
        nx = config.n_embd
        self.ln_1 = LayerNorm(nx, eps=config.layer_norm_epsilon)
        self.attn = Attention(nx, n_ctx, config, scale)
        self.ln_2 = LayerNorm(nx, eps=config.layer_norm_epsilon)
        self.mlp = MLP(4 * nx, config)

    def forward(
        self, x: torch.Tensor, layer_past: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        a, present = self.attn(self.ln_1(x), layer_past=layer_past)
        x = x + a
        m = self.mlp(self.ln_2(x))
        x = x + m
        return x, present


class GPT2Model(nn.Module):  # type: ignore
    def __init__(self, config: GPT2Config) -> None:
        super(GPT2Model, self).__init__()
        self.n_layer = config.n_layer
        self.n_embd = config.n_embd
        self.n_vocab = config.vocab_size

        self.wte = nn.Embedding(config.vocab_size, config.n_embd)  # type: ignore
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)  # type: ignore
        block = Block(config.n_ctx, config, scale=True)
        self.h = nn.ModuleList(  # type: ignore
            [copy.deepcopy(block) for _ in range(config.n_layer)])
        self.ln_f = LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

    def set_embeddings_weights(self, model_embeddings_weights: torch.Tensor) -> None:
        embed_shape = model_embeddings_weights.shape
        self.decoder = nn.Linear(embed_shape[1], embed_shape[0], bias=False)  # type: ignore
        self.decoder.weight = model_embeddings_weights  # Tied weights

    def forward(
            self,
            input_ids: torch.Tensor,
            position_ids: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            past: Optional[List[Optional[torch.Tensor]]] = None
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        if past is None:
            past_length = 0
            past = [None] * len(self.h)
        else:
            assert past[0] is not None
            past_length = past[0][0].size(-2)
        if position_ids is None:
            position_ids = torch.arange(  # type: ignore
                past_length, input_ids.size(-1) + past_length,
                dtype=torch.long,  # type: ignore
                device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_ids.size(-1))
        position_ids = position_ids.view(-1, position_ids.size(-1))

        inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1))
            token_type_embeds = self.wte(token_type_ids)
        else:
            token_type_embeds = 0
        hidden_states = inputs_embeds + position_embeds + token_type_embeds
        presents = []
        for block, layer_past in zip(self.h, past):
            hidden_states, present = block(hidden_states, layer_past)
            presents.append(present)
        hidden_states = self.ln_f(hidden_states)
        output_shape = input_shape + (hidden_states.size(-1),)
        return hidden_states.view(*output_shape), presents


class GPT2LMHead(nn.Module):  # type: ignore
    def __init__(self, model_embeddings_weights: torch.Tensor, config: GPT2Config) -> None:
        super(GPT2LMHead, self).__init__()
        self.n_embd = config.n_embd
        self.set_embeddings_weights(model_embeddings_weights)

    def set_embeddings_weights(self, model_embeddings_weights: torch.Tensor) -> None:
        embed_shape = model_embeddings_weights.shape
        self.decoder = nn.Linear(embed_shape[1], embed_shape[0], bias=False)  # type: ignore
        self.decoder.weight = model_embeddings_weights  # Tied weights

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        # Truncated Language modeling logits (we remove the last token)
        # h_trunc = h[:, :-1].contiguous().view(-1, self.n_embd)
        lm_logits = self.decoder(hidden_state)
        return lm_logits  # type: ignore


class GPT2LMHeadModel(nn.Module):  # type: ignore
    def __init__(self, config: GPT2Config) -> None:
        super(GPT2LMHeadModel, self).__init__()
        self.transformer = GPT2Model(config)
        self.lm_head = GPT2LMHead(self.transformer.wte.weight, config)

    def set_tied(self) -> None:
        """ Make sure we are sharing the embeddings
        """
        self.lm_head.set_embeddings_weights(self.transformer.wte.weight)

    def forward(self,
                input_ids: torch.Tensor,
                position_ids: Optional[torch.Tensor] = None,
                token_type_ids: Optional[torch.Tensor] = None,
                lm_labels: Optional[torch.Tensor] = None,
                past: Optional[List[Optional[torch.Tensor]]] = None
                ) -> Union[Tuple[torch.Tensor, List[torch.Tensor]], float]:
        hidden_states, presents = self.transformer(input_ids, position_ids, token_type_ids, past)
        lm_logits = self.lm_head(hidden_states)
        if lm_labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)  # type: ignore
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), lm_labels.view(-1))
            return loss  # type: ignore
        return lm_logits, presents
