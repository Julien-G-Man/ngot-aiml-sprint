# Verify that the model loads correctly before running full training

import os, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from dotenv import load_dotenv

load_dotenv()

MODEL_ID = 'mistralai/Mistral-7B-Instruct-v0.2'
# Alternative if Mistral requires acceptance:
#   MODEL_ID = 'meta-llama/Llama-2-7b-chat-hf'  (need Meta access)
#   MODEL_ID = 'microsoft/phi-2'                  (no auth needed, 3B params)

# ── 4-bit quantisation config ─────────────────────────────────
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_compute_dtype=torch.bfloat16
        if torch.cuda.get_device_capability()[0] >= 8  # Ampere+ GPUs
        else torch.float16,
    bnb_4bit_use_double_quant=True,
)

# ── Load tokeniser ────────────────────────────────────────────
print(f'Loading tokeniser from {MODEL_ID}...')
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    token=os.getenv('HF_TOKEN'),
)
# Add padding token if missing (Mistral does not have one by default)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

# ── Load model in 4-bit ───────────────────────────────────────
print('Loading model in 4-bit precision (this takes ~2 minutes first time)...')
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map='auto',       # Automatically spread across available GPUs
    trust_remote_code=True,
    token=os.getenv('HF_TOKEN'),
)

# ── Verify memory usage ───────────────────────────────────────
used  = torch.cuda.memory_allocated() / 1e9
total = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f'GPU memory used: {used:.1f} GB / {total:.1f} GB total')
print('Model loaded successfully!')

# ── Quick test inference (before fine-tuning) ─────────────────
test_prompt = '<s>[INST] Extract chemicals and diseases from: Aspirin inhibits COX and reduces fever. [/INST]'
inputs = tokenizer(test_prompt, return_tensors='pt').to('cuda')
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=100, temperature=0.1,
do_sample=True)
print('\nBase model output (before fine-tuning):')
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
