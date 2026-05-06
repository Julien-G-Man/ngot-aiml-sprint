from datasets import load_dataset
from collections import Counter

print('Loading BC5CDR dataset from HuggingFace...')
# The dataset is publicly available — no token needed
dataset = load_dataset('bigbio/bc5cdr', name='bc5cdr_bigbio_kb',
trust_remote_code=True)

print(f'Splits: {list(dataset.keys())}')
print(f'Train size:      {len(dataset["train"])}')
print(f'Validation size: {len(dataset["validation"])}')
print(f'Test size:       {len(dataset["test"])}')

# Inspect one example
example = dataset['train'][0]
print('\n=== EXAMPLE DOCUMENT ===')
print(f'ID: {example["id"]}')
print(f'Passages: {len(example["passages"])} (usually title + abstract)')
print(f'Entities: {len(example["entities"])}')
print(f'\nTitle: {example["passages"][0]["text"][0][:120]}')
print(f'\nEntities:')
for ent in example['entities'][:5]:
    print(f'  [{ent["type"]}]: {ent["text"][0]}')

# Count entity types
all_types = [ent['type'] for split in ['train','validation','test']
             for doc in dataset[split] for ent in doc['entities']]
print(f'\nEntity type distribution: {Counter(all_types)}')
# Expected: Counter({'Disease': ~4000, 'Chemical': ~4000})
