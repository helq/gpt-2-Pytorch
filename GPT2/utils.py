'''
    code by TaeHwan Jung(@graykode)
    Original Paper and repository here : https://github.com/openai/gpt-2
    GPT2 Pytorch Model : https://github.com/huggingface/pytorch-pretrained-BERT
'''
from __future__ import annotations

import logging

import torch.nn as nn
from .model import GPT2LMHeadModel, GPT2Model
from typing import Any, Dict, Union, List

logger = logging.getLogger(__name__)


def interact(locals: Any) -> None:
    import code
    code.InteractiveConsole(locals=locals).interact()


def load_weight(  # noqa: C901
    model: GPT2LMHeadModel,
    state_dict: Dict[str, Any]
) -> GPT2LMHeadModel:
    old_keys = []
    new_keys = []
    for key in state_dict.keys():
        new_key = None
        if key.endswith(".g"):
            new_key = key[:-2] + ".weight"
        elif key.endswith(".b"):
            new_key = key[:-2] + ".bias"
        elif key.endswith(".w"):
            new_key = key[:-2] + ".weight"
        if new_key:
            old_keys.append(key)
            new_keys.append(new_key)
    for old_key, new_key in zip(old_keys, new_keys):
        state_dict[new_key] = state_dict.pop(old_key)

    missing_keys = []  # type: List[str]
    unexpected_keys = []  # type: List[str]
    error_msgs = []  # type: List[str]
    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, "_metadata", None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata  # type: ignore

    def load(module: nn.Module, prefix: str = "") -> None:  # type: ignore
        local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
        module._load_from_state_dict(
            state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs
        )
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + ".")

    start_model = model  # type: Union[GPT2Model, GPT2LMHeadModel]
    if hasattr(model, "transformer") \
            and all(not s.startswith('transformer.') for s in state_dict.keys()):
        start_model = model.transformer
    load(start_model, prefix="")

    # Make sure we are still sharing the output and input embeddings after loading weights
    model.set_tied()
    return model
