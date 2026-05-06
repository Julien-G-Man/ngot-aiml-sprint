# scripts/bnb_config_demo.py
from transformers import BitsAndBytesConfig
import torch

# BitsAndBytesConfig controls how the base model is loaded
bnb_config = BitsAndBytesConfig(
    # ── Load in 4-bit ─────────────────────────────────────────
    load_in_4bit=True,

    # ── NF4: Normal Float 4 quantisation type ─────────────────
    # NF4 is optimised for neural network weights
    # (they follow a normal distribution, NF4 represents this efficiently)
    bnb_4bit_quant_type='nf4',

    # ── Compute dtype: precision for forward pass calculations ─
    # bfloat16 is preferred on Ampere+ GPUs (RTX 3080, A100)
    # float16 on older GPUs (RTX 2080, T4)
    bnb_4bit_compute_dtype=torch.bfloat16,

    # ── Double quantisation: compress the quantisation constants
    # Saves ~0.3 GB extra memory with negligible quality loss
    bnb_4bit_use_double_quant=True,
)

print('BitsAndBytes 4-bit quantisation config created')
print('Base model will be loaded in NF4 format')
print('Expected memory reduction: 75% vs float32')
