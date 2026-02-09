import torch
import torch.nn as nn

chars = sorted(set("".join(names)))
char_to_ix = {c: i for i, c in enumerate(chars)}
ix_to_char = {i: c for c, i in char_to_ix.items()}

vocab_size = len(chars)
num_classes = len(names)

def word_to_vector(word):
    vec = torch.zeros(vocab_size)
    for ch in word:
        vec[char_to_ix[ch]] += 1.0
    return vec

X = torch.stack([word_to_vector(name) for name in names])
y = torch.tensor([0, 1, 2, 3, 4])  # labels for names

