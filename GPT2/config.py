'''
    code by TaeHwan Jung(@graykode)
    Original Paper and repository here : https://github.com/openai/gpt-2
    GPT2 Pytorch Model : https://github.com/huggingface/pytorch-pretrained-BERT
'''


class GPT2Config(object):
    def __init__(
            self,
            vocab_size_or_config_json_file: int = 50257,
            n_positions: int = 1024,
            n_ctx: int = 1024,
            n_embd: int = 768,
            n_layer: int = 12,
            n_head: int = 12,
            layer_norm_epsilon: float = 1e-5,
            initializer_range: float = 0.02,
    ) -> None:
        self.vocab_size = vocab_size_or_config_json_file
        self.n_ctx = n_ctx
        self.n_positions = n_positions
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
