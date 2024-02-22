"""A code-golfy refactoring of Karpathy's implementation:
https://github.com/karpathy/minbpe."""

from pathlib import Path
from collections import Counter


class BPETokenizer:
    """Simple Byte-Pair Encoding that transforms text to tokens."""

    def __init__(self):
        """Copy a previous-defined vocab or create a new one."""
        self.vocab = {idx: bytes([idx]) for idx in range(256)}
        self.merges = {}
        self.special_tokens = {}

    def encode(self, text: str) -> list[int]:
        """Encode / tokenize the input text."""
        tokens = list(text.encode("utf-8"))
        while len(tokens) >= 2:
            left, right = min(c := Counter(zip(tokens, tokens[1:])), key=c.get)

            if (left, right) not in self.merges:
                break

            idx = self.merges[(left, right)]
            tokens = self._merge(tokens, left, right, idx)
        return tokens

    def decode(self, tokens: list[str]) -> str:
        """Decode the input tokens."""
        return (b"".join(self.vocab[idx] for idx in tokens)).decode(
            encoding="utf-8", errors="replace"
        )

    def train(self, corpus_path: Path, vocab_size: int) -> None:
        """'train' the BPE tokenizer on a corpus of data."""
        text = corpus_path.open(mode="r", encoding="utf-8").read()
        tokens = list(text.encode("utf-8"))
        for i in range(vocab_size - 256):
            left, right = max(c := Counter(zip(tokens, tokens[1:])), key=c.get)
            tokens = self._merge(tokens, left, right, 256 + i)
            self.merges[(left, right)] = 256 + i
            self.vocab[256 + i] = self.vocab[left] + self.vocab[right]

    def add_special(self, str_token: str) -> int:
        """Add a string token to the vocabulary. Returns the index
        at which it is placed."""
        self.vocab[max(self.vocab.keys()) + 1] = str_token
        self.special_tokens[str_token] = str_token
        return max(self.vocab.keys())

    def _merge(
        self, tokens: list[int], left: int, right: int, new_id: int
    ) -> list[int]:
        new_tokens, i = [], 0
        while i < len(tokens) - 1:
            cond = tokens[i] == left and tokens[i + 1] == right
            new_tokens.append(new_id if cond else tokens[i])
            i += 1 + cond
        return new_tokens

    def load(self, tokenizer_path: Path) -> None:
        ...

    def save(self, save_path: Path | str) -> None:
        with Path(save_path).open(mode="w", encoding="utf-8") as f:
            for special, idx in self.special_tokens.items():
                f.write(f"{special} {idx}\n")
            for idx1, idx2 in self.merges:
                f.write(f"{idx1} {idx2}\n")
