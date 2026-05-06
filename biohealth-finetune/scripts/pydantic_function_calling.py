# Use Pydantic V2 model as a function schema for OpenAI

import os
import json
import time
from openai import OpenAI
from pydantic import BaseModel, ValidationError, Field
from typing import Literal
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# ── Define your output schema as a Pydantic model ─────────────
class ClinicalExtraction(BaseModel):
    """Structured clinical entity extraction from medical text."""
    primary_diagnosis:   str   = Field(..., description='Main diagnosis in plain English')
    icd10_code:          str   = Field(..., description='ICD-10 code, e.g. I21.0')
    confidence_score:    float = Field(..., ge=0.0, le=1.0, description='Confidence 0-1')
    severity:            Literal['mild', 'moderate', 'severe', 'critical']
    key_findings:        list[str] = Field(..., description='Main clinical findings')
    urgent_referral:     bool  = Field(..., description='Does this need urgent referral?')
    first_line_treatment:str   = Field(..., description='Evidence-based initial treatment')

# ── Convert Pydantic model to OpenAI function spec ─────────────
def pydantic_to_function_spec(model_class: type, func_name: str, description: str) -> dict:
    """Convert a Pydantic model to an OpenAI function/tool specification."""
    schema = model_class.model_json_schema()
    return {
        'type': 'function',
        'function': {
            'name': func_name,
            'description': description,
            'parameters': schema,
        }
    }


# ── Call with function forcing ─────────────────────────────────
def extract_clinical(text: str) -> ClinicalExtraction:
    """Extract clinical entities — returns validated Pydantic object."""
    tool_spec = pydantic_to_function_spec(
        ClinicalExtraction,
        'extract_clinical_entities',
        'Extract structured clinical information from medical text',
    )
    response = client.chat.completions.create(
        model='gpt-4o-mini',
        messages=[
            {'role': 'system', 'content': 'You are a clinical NLP system. Extract entities from medical text.'},
            {'role': 'user',   'content': text}
        ],
        tools=[tool_spec],
        tool_choice={'type': 'function', 'function': {'name': 'extract_clinical_entities'}},
        temperature=0.0,
    )
    # Extract the function call arguments (always valid JSON since model was forced)
    raw_args = response.choices[0].message.tool_calls[0].function.arguments
    parsed   = json.loads(raw_args)
    # Validate with Pydantic — raises ValidationError if model returned wrong types
    return ClinicalExtraction(**parsed)

# ── Test ───────────────────────────────────────────────────────
result = extract_clinical(
    'ECG shows inferior STEMI. Troponin I 2.4 ng/mL. BP 90/60. Urgent cath lab activation.'
)

print(f'Diagnosis:   {result.primary_diagnosis}')
print(f'ICD-10:      {result.icd10_code}')
print(f'Severity:    {result.severity}')
print(f'Urgent:      {result.urgent_referral}')
print(f'Confidence:  {result.confidence_score:.0%}')
print(f'Treatment:   {result.first_line_treatment}')

# This is now a TYPED Pydantic object — not a string, not a dict
# IDE autocomplete works, type checking works, Pydantic validation works


def extract_with_retry(
    text: str,
    max_retries: int = 3,
) -> ClinicalExtraction:
    """Extract with automatic retry on validation failure."""
    error_context = ''
    for attempt in range(max_retries):
        try:
            # Add error context to help the model correct itself
            full_text = text + error_context
            return extract_clinical(full_text)
        except ValidationError as e:
            # Feed the validation error BACK to the model in the next attempt
            error_context = f'''

CORRECTION NEEDED: Your previous output failed validation with these errors:
{e}

Please ensure: - confidence_score is a float between 0.0 and 1.0 - severity is exactly one of: mild, moderate, severe, critical - urgent_referral is a boolean (true/false)
'''
            print(f'Attempt {attempt+1} failed. Retrying with error feedback...')
            time.sleep(0.5)   # Brief pause before retry

    raise RuntimeError(f'Failed to extract clinical entities after {max_retries} attempts')


