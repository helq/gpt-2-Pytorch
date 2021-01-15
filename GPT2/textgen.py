import torch
import random
import numpy as np
from GPT2.model import GPT2LMHeadModel
import GPT2.utils as utils
from GPT2.config import GPT2Config
from GPT2.sample import sample_sequence, predict_next
from GPT2.encoder import get_encoder

from typing import Any, Dict, Optional, Generator, List


class TextGenerator:
    def __init__(self, state_dict: Dict[str, Any],
                 seed: Optional[int] = None,
                 ) -> None:
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

        self.start_seed(seed)

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
        quiet: bool = True,
    ) -> Generator[str, None, None]:
        assert (text is None and unconditional) or (text is not None and not unconditional)
        if batch_size is None:
            batch_size = 1
        assert nsamples % batch_size == 0

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

    def generate_next_options(
        self,
        text: str,
        temperature: float = 1,
        top_k: int = 0,
        length: Optional[int] = None
    ) -> List[str]:
        context_tokens = self.enc.encode(text)
        out = predict_next(
            self.model, context=context_tokens, temperature=temperature,
            top_k=top_k,
            length=1 if length is None else length,
            device=self.device
        )
        return [self.enc.decode(opt) for opt in out.tolist()]
