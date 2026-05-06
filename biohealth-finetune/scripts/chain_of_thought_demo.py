# scripts/chain_of_thought_demo.py

import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

CLINICAL_QUESTION = '''
A 68-year-old male presents with: progressive dyspnoea on exertion for 3 months,
bilateral ankle oedema, elevated JVP, S3 gallop, bilateral basal crackles.
Recent echo shows EF 32%. BNP 890 pg/mL.
What is the most likely diagnosis and what is the first-line treatment?
'''

# WITHOUT Chain-of-Thought
def no_cot(question):
    return client.chat.completions.create(
        model='gpt-4o-mini',
        messages=[
            {'role': 'system', 'content': 'You are a medical AI. Answer concisely.'},
            {'role': 'user',   'content': question}
        ],
        temperature=0.1,
    ).choices[0].message.content

# WITH Chain-of-Thought
def with_cot(question):
    cot_system = '''You are a clinical reasoning AI.
For every clinical question, ALWAYS follow this reasoning process:

STEP 1 — IDENTIFY KEY FINDINGS: List the significant symptoms, signs, and
investigations.
STEP 2 — PATTERN RECOGNITION: What syndrome or pathophysiology do these findings
suggest?
STEP 3 — DIFFERENTIAL DIAGNOSIS: List 2-3 possible diagnoses, most likely first.
STEP 4 — MOST LIKELY DIAGNOSIS: State the diagnosis and explain your reasoning.
STEP 5 — FIRST-LINE TREATMENT: Based on current guidelines, state initial
management.
STEP 6 — CONFIDENCE: Rate your confidence 1-10 and explain any uncertainty.

Always show ALL steps, even for straightforward cases.'''

    return client.chat.completions.create(
        model='gpt-4o-mini',
        messages=[
            {'role': 'system', 'content': cot_system},
            {'role': 'user',   'content': question}
        ],
        temperature=0.1,
    ).choices[0].message.content

print('WITHOUT CoT (often misses nuances):')
print(no_cot(CLINICAL_QUESTION))
print('\nWITH Chain-of-Thought (structured, comprehensive):')
print(with_cot(CLINICAL_QUESTION))
