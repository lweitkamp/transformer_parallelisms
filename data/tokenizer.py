"""A code-golfy refactoring of Karpathy's implementation:
https://github.com/karpathy/minbpe."""

from pathlib import Path
from collections import Counter


class BPETokenizer:
    def __init__(self):
        self.vocab = None
        self.merges = None

    def encode(self, text: str) -> list[int]:
        ids = list(text.encode("utf-8"))
        while len(ids) >= 2:
            left, right = min(c := Counter(zip(ids, ids[1:])), key=c.get)

            if (left, right) not in self.merges:
                break

            idx = self.merges[(left, right)]
            ids = self._merge(ids, left, right, idx)
        return ids

    def decode(self, ids: list[str]) -> str:
        return (b"".join(self.vocab[idx] for idx in ids)).decode(
            encoding="utf-8", errors="replace"
        )

    def train(self, corpus_path: Path, vocab_size: int) -> None:
        text = corpus_path.open(mode="r", encoding="utf-8").read()
        ids = list(text.encode("utf-8"))

        vocab, merges = {idx: bytes([idx]) for idx in range(256)}, {}
        for i in range(vocab_size - 256):
            left, right = max(c := Counter(zip(ids, ids[1:])), key=c.get)
            ids = self._merge(ids, left, right, 256 + i)
            merges[(left, right)] = 256 + i
            vocab[256 + i] = vocab[left] + vocab[right]

        self.vocab = vocab
        self.merges = merges

    def _merge(self, ids: list[int], left: int, right: int, new_id: int) -> list[int]:
        new_ids, i = [], 0
        while i < len(ids) - 1:
            cond = ids[i] == left and ids[i + 1] == right
            new_ids.append(new_id if cond else ids[i])
            i += 1 + cond
        return new_ids

    def load(self, tokenizer_path: Path) -> None:
        ...

    def save(self, save_path: Path) -> None:
        assert self.vocab
        assert self.merges
        ...
