'''
    code by TaeHwan Jung(@graykode)
    Original Paper and repository here : https://github.com/openai/gpt-2
    GPT2 Pytorch Model : https://github.com/huggingface/pytorch-pretrained-BERT
'''
from __future__ import annotations

import os
import sys
import torch
import argparse
from GPT2.textgen import TextGenerator

# from GPT2.utils import interact

from typing import Any


def get_args() -> Any:
    parser = argparse.ArgumentParser()

    group = parser.add_argument_group("Sequence generation")
    groupexc = group.add_mutually_exclusive_group(required=True)
    groupexc.add_argument("--text", type=str, default=None)
    groupexc.add_argument('--unconditional',
                          action='store_true', help='If true, unconditional generation.')
    group.add_argument("--nsamples", type=int, default=1)
    group.add_argument("--batch_size", type=int, default=None)

    group2 = parser.add_argument_group("Next word")
    group2.add_argument("--next_word", action='store_true',
                        help='Activates next word only. It only shows the next most probable words')

    parser.add_argument("--length", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument(
        "--top_k", type=int, default=40,
        help='How many "words" to consider when sampling. 0 means consider everything')

    parser.add_argument("--quiet", action='store_true')
    args = parser.parse_args()

    if args.quiet is False:
        print(args)

    return args


if __name__ == '__main__':
    if not os.path.exists('gpt2-pytorch_model.bin'):
        print('Please download gpt2-pytorch_model.bin')
        sys.exit(1)

    args = get_args()

    state_dict = torch.load('gpt2-pytorch_model.bin',  # type: ignore
                            map_location='cpu' if not torch.cuda.is_available() else None)

    tg = TextGenerator(state_dict)

    if args.next_word:
        for word in tg.generate_next_options(
            args.text, args.temperature, args.top_k, length=args.length
        ):
            print(repr(word))
    else:
        print(args.text)
        for i, text in enumerate(tg.generate(
            text=args.text, quiet=args.quiet, nsamples=args.nsamples,
            unconditional=args.unconditional, batch_size=args.batch_size, length=args.length,
                temperature=args.temperature, top_k=args.top_k)):
            if args.quiet is False:
                print("="*40 + f" SAMPLE {i+1} " + "="*40)
            print(text)
