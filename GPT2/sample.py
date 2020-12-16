'''
    code by TaeHwan Jung(@graykode)
    Original Paper and repository here : https://github.com/openai/gpt-2
    GPT2 Pytorch Model : https://github.com/huggingface/pytorch-pretrained-BERT
'''
from __future__ import annotations

import torch
import torch.nn.functional as F
from tqdm import trange

from typing import Any, Optional
from .model import GPT2LMHeadModel
# from .utils import interact


def top_k_logits(logits: torch.Tensor, k: int) -> torch.Tensor:
    if k == 0:
        return logits
    values, _ = torch.topk(logits, k)  # type: ignore
    min_values = values[:, -1].reshape((-1, 1))
    return torch.where(logits < min_values,  # type: ignore
                       torch.ones_like(logits, dtype=logits.dtype) * -1e10,  # type: ignore
                       logits)


def sample_sequence(
    model: GPT2LMHeadModel,
    length: Optional[int] = None,
    start_token: Any = None,
    batch_size: Optional[int] = None,
    context: Any = None,
    temperature: float = 1,
    top_k: int = 0,
    device: str = 'cuda',
    sample: bool = True,
    quiet: bool = True
) -> torch.Tensor:
    if length is None:
        length = 1024

    if start_token is None:
        assert context is not None, 'Specify exactly one of start_token and context!'
        context = torch.tensor(  # type: ignore
            context, device=device, dtype=torch.long  # type: ignore
        ).unsqueeze(0).repeat(batch_size, 1)
    else:
        assert context is None, 'Specify exactly one of start_token and context!'
        context = torch.full(  # type: ignore
            (batch_size, 1), start_token, device=device, dtype=torch.long)  # type: ignore

    # print("context.shape =", context.shape)
    # interact(locals())
    if context.shape[1] >= 1024:
        raise ValueError("The size of the input text exceeds the capacity of the network.\n"
                         "GPT-2 is a CNN in nature. It doesn't have an internal state.\n"
                         "It takes an input of size < 1024 and predicts the next value.")

    length = min(length, 1024 - int(context.shape[1]))

    prev = context
    output = context
    past = None
    with torch.no_grad():
        myrange = trange(length, unit="char") if not quiet else range(length)
        for i in myrange:
            logits, past = model(prev, past=past)
            logits = logits[:, -1, :] / temperature
            logits = top_k_logits(logits, k=top_k)
            log_probs = F.softmax(logits, dim=-1)
            if sample:
                prev = torch.multinomial(log_probs, num_samples=1)  # type: ignore
            else:
                _, prev = torch.topk(log_probs, k=1, dim=-1)  # type: ignore
            output = torch.cat((output, prev), dim=1)  # type: ignore
    return output  # type: ignore
