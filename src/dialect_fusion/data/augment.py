from __future__ import annotations

import random


class LightAugmenter:
    """
    Applies a random light perturbation to whitespace-tokenized text:
    word deletion, adjacent-word swap, or both. Used only on the training
    split to regularize the character-BiLSTM branches.
    """

    def __init__(self, p_del: float = 0.10, p_swap: float = 0.10, prob: float = 0.50):
        self.p_del = p_del
        self.p_swap = p_swap
        self.prob = prob

    def _delete(self, words: list[str]) -> list[str]:
        if len(words) <= 2:
            return words
        out = [w for w in words if random.random() > self.p_del]
        return out or [random.choice(words)]

    def _swap(self, words: list[str]) -> list[str]:
        if len(words) < 2:
            return words
        words = words[:]
        i = random.randint(0, len(words) - 2)
        words[i], words[i + 1] = words[i + 1], words[i]
        return words

    def __call__(self, text: str) -> str:
        if random.random() > self.prob:
            return text
        words = str(text).split()
        op = random.choice(["del", "swap", "both"])
        if op == "del":
            words = self._delete(words)
        elif op == "swap":
            words = self._swap(words)
        else:
            words = self._swap(self._delete(words))
        return " ".join(words) if words else text
