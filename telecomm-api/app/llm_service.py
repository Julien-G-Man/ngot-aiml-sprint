import os, json, time, logging
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import ValidationError
from .schemas import ComplaintAnalysis

load_dotenv()
logger = logging.getLogger(__name__)

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

SYSTEM_PROMPT = '''You are an AI customer service analyst for a telecommunications company.
Analyse the customer complaint and extract structured information.
Be precise — this data feeds directly into our CRM and escalation system.'''

TOOL_SPEC = {
    'type': 'function',
    'function': {
        'name': 'analyse_complaint',
        'description': 'Extract structured analysis from a telecomm customer complaint',
        'parameters': ComplaintAnalysis.model_json_schema(),
    }
}

def analyse_complaint(
    complaint_text: str,
    max_retries: int = 3,
) -> ComplaintAnalysis:
    """Analyse a complaint with retry on validation failure."""
    error_context = ''

    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model='gpt-4o-mini',
                messages=[
                    {'role': 'system', 'content': SYSTEM_PROMPT + error_context},
                    {'role': 'user',   'content': complaint_text},
                ],
                tools=[TOOL_SPEC],
                tool_choice={'type': 'function',
                             'function': {'name': 'analyse_complaint'}},
                temperature=0.0,
            )
            raw_args = response.choices[0].message.tool_calls[0].function.arguments
            return ComplaintAnalysis(**json.loads(raw_args))

        except (ValidationError, json.JSONDecodeError) as e:
            logger.warning(f'Attempt {attempt} failed: {e}')
            error_context = (
                f'\n\nPREVIOUS ATTEMPT FAILED VALIDATION:\n{e}\n'
                f'Please correct the output and ensure all fields match the required types.'
            )
            if attempt == max_retries:
                raise RuntimeError(f'Failed after {max_retries} attempts: {e}')

    raise RuntimeError('Unreachable')
