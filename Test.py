import torch

from LRU_pytorch import LRU

# Create a single Linear Recurrent Unit, that takes in inputs of size (batch_size, seq_length, 30) (or (seq_length,
# 30)), with internal state-space variable of size 10, and returns outputs of (batch_size, seq_length,
# 50) (or (seq_length, 50)).

lru = LRU(
    in_features=30,
    out_features=50,
    state_features=10
)

preds = lru(torch.randn([2, 70, 30]))  # Get predictions


preds