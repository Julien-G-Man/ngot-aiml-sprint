from peft import LoraConfig, TaskType

# LoraConfig defines HOW the LoRA adapters are added to the model
lora_config = LoraConfig(
    # ── Rank: the 'r' in our ΔW ≈ A×B formula ──────────────
    # Higher r = more capacity to learn = more parameters = less memory saving
    # r=8: good starting point for most tasks
    # r=16: better for complex tasks needing more expressivity
    # r=64: almost full fine-tuning quality, less memory saving
    r=8,

    # ── Alpha: scaling factor for the LoRA update ───────────
    # Applied as: W_update * (alpha/r)
    # Rule of thumb: lora_alpha = 2 * r  (so alpha=16 for r=8)
    # Higher alpha = larger learning rate effect = faster but less stable
    lora_alpha=16,

    # ── Target modules: which weight matrices to add LoRA to ─
    # 'q_proj', 'v_proj': query and value projections in attention
    # Adding more modules (k, o, gate, up, down) improves quality
    # but increases trainable parameters
    target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj',
                    'gate_proj', 'up_proj', 'down_proj'],

    # ── Dropout: regularisation to prevent overfitting ────────
    lora_dropout=0.05,

    # ── Bias: whether to train bias terms ────────────────────
    bias='none',   # 'none' is the standard choice

    # ── Task type ────────────────────────────────────────────
    task_type=TaskType.CAUSAL_LM,  # Autoregressive text generation
)

print('LoRA config created:')
print(f'  r={lora_config.r}, alpha={lora_config.lora_alpha}')
print(f'  Target modules: {lora_config.target_modules}')
print(f'  Approx trainable params for 7B model: ~8M (0.12% of total)')
