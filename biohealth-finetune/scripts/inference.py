# Compare base model vs fine-tuned model on biomedical NER

import os, torch, json
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from dotenv import load_dotenv

load_dotenv()

MODEL_ID    = 'mistralai/Mistral-7B-Instruct-v0.2'
ADAPTER_DIR = './outputs/biohealth-ner-adapter'
SYSTEM_PROMPT = '''You are a biomedical NER system. Extract chemicals and diseases
from the text.
Return JSON: {"chemicals": [...], "diseases": [...]}'''

def load_model(use_adapter: bool):
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                             bnb_4bit_compute_dtype=torch.bfloat16)
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, quantization_config=bnb, device_map='auto',
token=os.getenv('HF_TOKEN'))
    if use_adapter:
        return PeftModel.from_pretrained(base, ADAPTER_DIR)
    return base

def run_ner(model, tokenizer, text: str) -> str:
    prompt = (f'<s>[INST] <<SYS>>\n{SYSTEM_PROMPT}\n<</SYS>>\n\n'
              f'Extract entities: {text} [/INST]')
    inputs  = tokenizer(prompt, return_tensors='pt').to('cuda')
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=150, temperature=0.1,
do_sample=True)
    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    # Extract just the response part (after [/INST])
    return decoded.split('[/INST]')[-1].strip()

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=os.getenv('HF_TOKEN'))
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

test_cases = [
    'Methotrexate-induced hepatotoxicity was observed in 12% of rheumatoid arthritis patients.',
    'Cisplatin nephrotoxicity and peripheral neuropathy are dose-limiting toxicities.',
    'Aspirin and clopidogrel dual antiplatelet therapy after myocardial infarction.',
]

print('=== BASE MODEL (no fine-tuning) ===')
base_model = load_model(use_adapter=False)
for text in test_cases:
    print(f'Input:  {text[:80]}...')
    print(f'Output: {run_ner(base_model, tokenizer, text)}')
    print()

del base_model; torch.cuda.empty_cache()

print('=== FINE-TUNED MODEL (QLoRA adapter) ===')
ft_model = load_model(use_adapter=True)
for text in test_cases:
    print(f'Input:  {text[:80]}...')
    output = run_ner(ft_model, tokenizer, text)
    print(f'Output: {output}')
    # Try to parse as JSON — fine-tuned model should be consistent
    try:
        parsed = json.loads(output)
        print(f'  ✅ Valid JSON: chemicals={parsed.get("chemicals",[])} diseases={parsed.get("diseases",[])}')
    except json.JSONDecodeError:
        print(f'  ⚠️  Not valid JSON')
    print()
