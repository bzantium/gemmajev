"""Pinned official backbones; importing this registry performs no downloads."""

MODELS = {
    "gemma-3-270m": {
        "model_id": "google/gemma-3-270m",
        "revision": "9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1",
        "config": "gemma3_270m",
    },
    "gemma-3-270m-it": {
        "model_id": "google/gemma-3-270m-it",
        "revision": "ac82b4e820549b854eebf28ce6dedaf9fdfa17b3",
        "config": "gemma3_270m_it",
    },
    "gemma-3-1b-it": {
        "model_id": "google/gemma-3-1b-it",
        "revision": "dcc83ea841ab6100d6b47a070329e1ba4cf78752",
        "config": "gemma3_1b_it",
    },
}


class ChatTokenizer:
    """Apply the official user-turn template before candidate-head pooling.

    No answer label is supplied. The final hidden state is the generation-prompt
    boundary after a user message containing the observed state and candidate.
    """

    def __init__(self, tokenizer):
        if not tokenizer.chat_template:
            raise ValueError("Chat input requires an official tokenizer chat template")
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id

    def encode(self, text, add_special_tokens=True):
        if not add_special_tokens:
            return self.tokenizer.encode(text, add_special_tokens=False)
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": text}], tokenize=True, add_generation_prompt=True
        )
