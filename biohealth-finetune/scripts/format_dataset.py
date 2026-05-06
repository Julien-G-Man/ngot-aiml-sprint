from datasets import load_dataset, Dataset
import json
from pathlib import Path

# ── Instruction templates ─────────────────────────────────────
SYSTEM_PROMPT = '''You are a biomedical named entity recognition (NER) system.
Given a biomedical text, extract all chemical and disease entities.
Return a JSON object with two keys: - "chemicals": list of chemical entity strings - "diseases": list of disease entity strings
Return ONLY the JSON object, no explanation.'''

def format_example(doc: dict) -> dict:
    """Convert one BC5CDR document to instruction format."""
    # Combine title and abstract into one text
    text_parts = []
    for passage in doc['passages']:
        for t in passage['text']:
            if t.strip():
                text_parts.append(t.strip())
    full_text = ' '.join(text_parts)

    # Collect entities by type
    chemicals, diseases = [], []
    seen = set()
    for ent in doc['entities']:
        name = ent['text'][0].strip()
        key = (name.lower(), ent['type'])
        if key not in seen:
            seen.add(key)
            if ent['type'] == 'Chemical':
                chemicals.append(name)
            elif ent['type'] == 'Disease':
                diseases.append(name)

    expected_output = json.dumps({'chemicals': chemicals, 'diseases': diseases})

    # Mistral instruction format: [INST] ... [/INST]
    instruction = (
        f'<s>[INST] <<SYS>>\n{SYSTEM_PROMPT}\n<</SYS>>\n\n'
        f'Extract entities from this biomedical text:\n\n{full_text[:1200]} [/INST]'
        f' {expected_output} </s>'
    )
    return {'text': instruction}

# ── Process the dataset ───────────────────────────────────────
raw = load_dataset('bigbio/bc5cdr', name='bc5cdr_bigbio_kb', trust_remote_code=True)

train_formatted = [format_example(doc) for doc in raw['train']]
val_formatted   = [format_example(doc) for doc in raw['validation']]

train_ds = Dataset.from_list(train_formatted)
val_ds   = Dataset.from_list(val_formatted)

# Save to disk for reuse
Path('data').mkdir(exist_ok=True)
train_ds.save_to_disk('data/bc5cdr_train_formatted')
val_ds.save_to_disk('data/bc5cdr_val_formatted')

print(f'Formatted {len(train_ds)} training examples')
print(f'Formatted {len(val_ds)} validation examples')
print('\nFirst example (truncated):')
print(train_ds[0]['text'][:500])
