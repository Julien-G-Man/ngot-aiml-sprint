# Three independent reasoning paths + synthesis

import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

DRUG_QUESTION = '''
A 45-year-old female with Type 2 diabetes, hypertension, CKD stage 3 (eGFR 42)
and a recent MI 6 months ago needs glucose-lowering therapy.
Current HbA1c is 8.9% on metformin 1g twice daily.
What drug should be added next?
'''

def tree_of_thought(question: str) -> str:
    # Step 1: Generate 3 independent expert perspectives
    perspective_prompt = f'''
    You are 3 independent clinical pharmacology experts.
    Each expert analyses this case INDEPENDENTLY without knowing what others think.
    Give Expert 1, Expert 2, and Expert 3's analysis separately.

    Case: {question}

    Expert 1 (focuses on cardiorenal protection):
    Expert 2 (focuses on glycaemic efficacy):
    Expert 3 (focuses on safety and tolerability):
    '''
    perspectives = client.chat.completions.create(
        model='gpt-4o-mini',
        messages=[{'role': 'user', 'content': perspective_prompt}],
        temperature=0.4,   # Slightly more variation between experts
    ).choices[0].message.content

    # Step 2: Synthesise the 3 perspectives into a final answer
    synthesis_prompt = f'''
    Three clinical experts have independently analysed this case:
    {perspectives}

    Now synthesise these perspectives:
    1. Where do the experts AGREE? This is your most confident recommendation.
    2. Where do they DISAGREE? What accounts for the disagreement?
    3. Final recommendation with evidence level (A/B/C).
    '''
    synthesis = client.chat.completions.create(
        model='gpt-4o-mini',
        messages=[{'role': 'user', 'content': synthesis_prompt}],
        temperature=0.0,
    ).choices[0].message.content

    return f'=== Expert Perspectives ===\n{perspectives}\n\n=== Synthesis ===\n{synthesis}'

print(tree_of_thought(DRUG_QUESTION))
