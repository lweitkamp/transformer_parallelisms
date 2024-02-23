"""A code-golfy refactoring of Karpathy's implementation:
https://github.com/karpathy/minbpe."""

from pathlib import Path
from collections import Counter


class BPETokenizer:
    """Simple Byte-Pair Encoding that transforms text to tokens."""

    def __init__(self):
        self.vocab = {idx: bytes([idx]) for idx in range(256)}
        self.merges = {}
        self.special_tokens = {}
        self.reverse_special = {}

    def encode(self, text: str) -> list[int]:
        """Encode / tokenize the input text."""
        tokens = list(text.encode("utf-8"))
        while len(tokens) >= 2:
            left, right = min(c := Counter(zip(tokens, tokens[1:])), key=c.get)

            if (left, right) not in self.merges:
                break

            tokens = self._merge(tokens, left, right, self.merges[(left, right)])
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
        self.special_tokens[str_token] = max(self.vocab)

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
        self.vocab = {idx: bytes([idx]) for idx in range(256)}
        self.special_tokens, self.merges = {}, {}
        with Path(tokenizer_path).open(mode="r", encoding="utf-8") as f:
            num_special = int(f.readline().strip())
            for _ in range(num_special):
                special, special_idx = f.readline().strip().split()
                self.special_tokens[special] = int(special_idx)
            for i, line in enumerate(f):
                idx1, idx2 = map(int, line.split())
                self.merges[(idx1, idx2)] = 256 + i

        # Build the vocab
        for (p0, p1), idx in self.merges.items():
            self.vocab[idx] = self.vocab[p0] + self.vocab[p1]
        for special, idx in self.special_tokens.items():
            self.vocab[idx] = special.encode("utf-8")

    def save(self, save_path: Path | str) -> None:
        with Path(save_path).open(mode="w", encoding="utf-8") as f:
            f.write(f"{len(self.special_tokens)}\n")
            for special, idx in self.special_tokens.items():
                f.write(f"{special} {idx}\n")
            for idx1, idx2 in self.merges:
                f.write(f"{idx1} {idx2}\n")

    @classmethod
    def from_saved(cls, path):
        tokenizer = cls()
        tokenizer.load(path)
        return tokenizer


def train_shakespeare():
    tokenizer = BPETokenizer()
    tokenizer.train(Path("data") / "shakespeare.txt", vocab_size=256 + 255)
    tokenizer.add_special("<|endoftext|>")  # 512 total tokens
    tokenizer.save(Path("data") / "tokenizer.model")

    txt = "Hello world, will this text be recovered properly?"
    encoded_text = tokenizer.encode(txt)
    tokenizer.load(Path("data") / "tokenizer.model")
    assert tokenizer.decode(encoded_text) == txt


if __name__ == "__main__":
    train_shakespeare()
