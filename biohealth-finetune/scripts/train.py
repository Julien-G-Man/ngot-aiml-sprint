# scripts/train.py
# Full QLoRA fine-tuning — this is the main training script

import os, torch
from pathlib import Path
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    BitsAndBytesConfig, TrainingArguments,
)
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from dotenv import load_dotenv

load_dotenv()

# ── Configuration ────────────────────────────────────────────
MODEL_ID     = 'mistralai/Mistral-7B-Instruct-v0.2'
OUTPUT_DIR   = './outputs/biohealth-ner-adapter'
MAX_SEQ_LEN  = 1024   # Truncate inputs to this length
BATCH_SIZE   = 2      # Per-GPU batch size (increase if VRAM allows)
GRAD_ACCUM   = 8      # Effective batch = BATCH_SIZE * GRAD_ACCUM = 16
EPOCHS       = 3
LR           = 2e-4

# ── Load data ─────────────────────────────────────────────────
print('Loading formatted dataset...')
train_ds = load_from_disk('data/bc5cdr_train_formatted')
val_ds   = load_from_disk('data/bc5cdr_val_formatted')
print(f'Train: {len(train_ds)}, Val: {len(val_ds)}')

# ── 4-bit config ─────────────────────────────────────────────
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_compute_dtype=torch.bfloat16
        if torch.cuda.get_device_capability()[0] >= 8 else torch.float16,
    bnb_4bit_use_double_quant=True,
)

# ── Load base model ──────────────────────────────────────────
print('Loading base model in 4-bit...')
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, quantization_config=bnb_config,
    device_map='auto', trust_remote_code=True, token=os.getenv('HF_TOKEN'),
)
# CRITICAL: prepare_model_for_kbit_training sets up gradient checkpointing
# and ensures 4-bit layers can be trained properly
model = prepare_model_for_kbit_training(model)

# ── LoRA config ──────────────────────────────────────────────
lora_config = LoraConfig(
    r=16,           # Slightly higher r for better quality on NER
    lora_alpha=32,  # 2 * r
    target_modules=[
        'q_proj', 'k_proj', 'v_proj', 'o_proj',
        'gate_proj', 'up_proj', 'down_proj',
    ],
    lora_dropout=0.05,
    bias='none',
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, lora_config)

# Print how many parameters we're actually training
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in model.parameters())
print(f'Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)')

# ── Tokeniser ────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID, trust_remote_code=True, token=os.getenv('HF_TOKEN'),
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ── Training arguments ────────────────────────────────────────
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,

    # Batching
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,

    # Training duration
    num_train_epochs=EPOCHS,

    # Learning rate
    learning_rate=LR,
    lr_scheduler_type='cosine',   # Cosine decay — standard for LLMs
    warmup_ratio=0.05,            # 5% of steps for warmup

    # Memory optimisation
    gradient_checkpointing=True,  # Trade compute for memory — REQUIRED for QLoRA
    bf16=torch.cuda.get_device_capability()[0] >= 8,   # bfloat16 on Ampere+
    fp16=torch.cuda.get_device_capability()[0] < 8,    # float16 on older GPUs
    tf32=True,                    # Use TF32 on Ampere+ for faster compute

    # Logging and checkpointing
    logging_steps=10,
    eval_strategy='epoch',
    save_strategy='epoch',
    save_total_limit=2,           # Keep only the 2 best checkpoints
    load_best_model_at_end=True,
    metric_for_best_model='eval_loss',
    report_to='none',             # Disable W&B / MLflow for this sprint
)

# ── SFTTrainer ────────────────────────────────────────────────
# SFTTrainer (Supervised Fine-Tuning Trainer) from TRL simplifies
# instruction-tuning — it handles packing, loss masking, and more
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    args=training_args,
    dataset_text_field='text',    # Column name in our dataset
    max_seq_length=MAX_SEQ_LEN,
    packing=False,                # Don't pack multiple examples into one sequence
)

# ── Train! ────────────────────────────────────────────────────
print('Starting QLoRA fine-tuning...')
print(f'Effective batch size: {BATCH_SIZE * GRAD_ACCUM}')
print(f'Training on {len(train_ds)} examples for {EPOCHS} epochs')
print(f'Estimated time: ~30–90 minutes depending on GPU')
print(f'Loss curve will appear below:')
print()

trainer.train()

# ── Save the adapter ─────────────────────────────────────────
print('Saving LoRA adapter...')
trainer.model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f'Adapter saved to {OUTPUT_DIR}')
print(f'Adapter size: {sum(f.stat().st_size for f in Path(OUTPUT_DIR).rglob("*") if
f.is_file()) / 1e6:.1f} MB')
# Expected: ~50–100 MB (vs 14 GB for the full model!)
