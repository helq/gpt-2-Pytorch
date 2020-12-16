'''
    code by TaeHwan Jung(@graykode)
    Original Paper and repository here : https://github.com/openai/gpt-2
    GPT2 Pytorch Model : https://github.com/huggingface/pytorch-pretrained-BERT
'''
from __future__ import annotations

import os
import sys
import torch
import random
import argparse
import numpy as np
from GPT2.model import GPT2LMHeadModel
import GPT2.utils as utils
from GPT2.config import GPT2Config
from GPT2.sample import sample_sequence
from GPT2.encoder import get_encoder

from typing import Any, Dict, Optional, Generator


def get_args() -> Any:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text", type=str, default=None)
    group.add_argument('--unconditional',
                       action='store_true', help='If true, unconditional generation.')
    parser.add_argument("--quiet", action='store_true')
    parser.add_argument("--nsamples", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--length", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_k", type=int, default=40)
    args = parser.parse_args()

    if args.quiet is False:
        print(args)

    return args


class TextGenerator:
    def __init__(self, state_dict: Dict[str, Any]) -> None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # type: ignore

        # Load Model
        config = GPT2Config()
        enc = get_encoder()
        model = GPT2LMHeadModel(config)
        model = utils.load_weight(model, state_dict)
        model.to(device)
        model.eval()

        self.device = device
        self.model = model
        self.config = config
        self.enc = enc

    def start_seed(self, seed: Optional[int]) -> None:
        if seed is None:
            seed = random.randint(0, 2147483647)
        np.random.seed(seed)
        torch.random.manual_seed(seed)
        torch.cuda.manual_seed(seed)  # type: ignore

    def generate(
        self,
        text: Optional[str] = None,
        nsamples: int = 1,
        unconditional: bool = False,
        batch_size: Optional[int] = None,
        length: Optional[int] = None,
        temperature: float = 0.7,
        top_k: int = 40,
        seed: Optional[int] = None,
        quiet: bool = True,
    ) -> Generator[str, None, None]:
        assert (text is None and unconditional) or (text is not None and not unconditional)
        if batch_size is None:
            batch_size = 1
        assert nsamples % batch_size == 0

        self.start_seed(seed)

        if length is not None and length > self.config.n_ctx:
            raise ValueError(f"Can't get samples longer than window size: {self.config.n_ctx}")

        if text is not None:
            context_tokens = self.enc.encode(text)
        else:
            eof = self.enc.encoder['<|endoftext|>']

        for _ in range(nsamples // batch_size):
            out = sample_sequence(
                model=self.model, length=length,
                context=None if unconditional else context_tokens,
                start_token=eof if unconditional else None,
                batch_size=batch_size,
                temperature=temperature, top_k=top_k, device=self.device,
                quiet=quiet
            )
            if unconditional:
                new_seq = out[:, len('<|endoftext|>'):].tolist()
            else:
                new_seq = out[:, len(context_tokens):].tolist()

            for i in range(batch_size):
                yield self.enc.decode(new_seq[i])


if __name__ == '__main__':
    if not os.path.exists('gpt2-pytorch_model.bin'):
        print('Please download gpt2-pytorch_model.bin')
        sys.exit()

    state_dict = torch.load('gpt2-pytorch_model.bin',  # type: ignore
                            map_location='cpu' if not torch.cuda.is_available() else None)
    args = get_args()

    print(args.text)
    tg = TextGenerator(state_dict)
    for i, text in enumerate(tg.generate(
        text=args.text, quiet=args.quiet, nsamples=args.nsamples,
        unconditional=args.unconditional, batch_size=args.batch_size, length=args.length,
            temperature=args.temperature, top_k=args.top_k)):
        if args.quiet is False:
            print("="*40 + f" SAMPLE {i+1} " + "="*40)
        print(text)
