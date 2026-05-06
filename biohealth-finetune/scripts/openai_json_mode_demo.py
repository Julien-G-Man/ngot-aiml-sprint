# JSON mode forces the model to ALWAYS return valid JSON
# (But does NOT enforce a specific schema — just valid JSON syntax)

import os
from openai import OpenAI
from dotenv import load_dotenv
import json

load_dotenv()
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

def extract_with_json_mode(clinical_text: str) -> dict:
    """Extract medical entities — guaranteed valid JSON output."""
    response = client.chat.completions.create(
        model='gpt-4o-mini',
        messages=[
            {'role': 'system',
             'content': '''You are a clinical NLP system.
Extract the following from medical text and return as JSON:
{
  "diagnosis": "primary diagnosis",
  "confidence": 0.0-1.0,
  "key_findings": ["finding1", "finding2"],
  "recommended_tests": ["test1", "test2"]
}'''},
            {'role': 'user', 'content': clinical_text}
        ],
        response_format={'type': 'json_object'},   # ← ENABLES JSON MODE
        temperature=0.0,
    )
    raw = response.choices[0].message.content
    return json.loads(raw)  # Always safe — JSON mode guarantees valid JSON

result = extract_with_json_mode(
    'Patient has ST-elevation MI, troponin 2.4, referred for urgent PCI'
)

print(result)
# always returns a dict - never crashes on JSON parse
