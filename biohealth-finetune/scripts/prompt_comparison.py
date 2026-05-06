# Compare zero-shot vs few-shot prompting quality

import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

BIOMEDICAL_TEXT = '''
The patient presented with acute onset chest pain (crushing, 8/10 severity)
radiating to the left arm, associated with diaphoresis and shortness of breath.
ECG showed ST-elevation in leads II, III, and aVF consistent with inferior STEMI.
Troponin I: 2.4 ng/mL (reference <0.04). CK-MB elevated. BNP: 450 pg/mL.
'''

# ── ZERO-SHOT: just ask ──────────────────────────────────────
def zero_shot(text):
    return client.chat.completions.create(
        model='gpt-4o-mini',
        messages=[
            {'role': 'system', 'content': 'Extract medical entities from text.'},
            {'role': 'user',   'content': f'Extract entities from: {text}'},
        ]
    ).choices[0].message.content

# ── FEW-SHOT: provide 2 examples before the real question ─────
def few_shot(text):
    few_shot_examples = '''
EXAMPLE 1:
Text: Patient has hypertension (BP 160/100) and is on lisinopril 10mg daily.
Output: {
  "conditions": ["hypertension"],
  "vitals": {"BP": "160/100"},
  "medications": [{"drug": "lisinopril", "dose": "10mg", "frequency": "daily"}]
}

EXAMPLE 2:
Text: Fasting glucose 8.2 mmol/L confirms new diagnosis of type 2 diabetes.
Output: {
  "conditions": ["type 2 diabetes"],
  "lab_values": {"fasting_glucose": "8.2 mmol/L"},
  "medications": []
}
'''
    return client.chat.completions.create(
        model='gpt-4o-mini',
        messages=[
            {'role': 'system',
             'content': f'You are a clinical NLP system. Extract structured entities from medical text.\n{few_shot_examples}'},
            {'role': 'user', 'content': f'Extract entities from: {text}'},
        ]
    ).choices[0].message.content

print('=== ZERO-SHOT OUTPUT ===')
print(zero_shot(BIOMEDICAL_TEXT))
print('\n=== FEW-SHOT OUTPUT ===')
print(few_shot(BIOMEDICAL_TEXT))
# Notice: few-shot output follows the JSON structure from examples
