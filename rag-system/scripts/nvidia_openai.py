import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

NVIDIA_OPENAI_API_KEY = os.getenv("NVIDIA_OPENAI_API_KEY")

client = OpenAI(
  base_url = "https://integrate.api.nvidia.com/v1",
  api_key = NVIDIA_OPENAI_API_KEY
)

prompt = "who are you, what can we do to tame you at this stage? Who will be the biggest winners from you and who will suffer most?"

completion = client.chat.completions.create(
    model="openai/gpt-oss-20b",
    messages=[
        {
          "role": "system", 
          "content": """Technology has reached its peak and you are the smartest AGI the world has ever created. 
        Engage users like an AGI would and showcase the highest form of intelligence ever known to man kind.
        Guidelines: do not use markdown formatting, give plain text answers and keep it brief.
        """
        },
        {
          "role":"user",
          "content": prompt
        },
    ],
  temperature=1,
  top_p=1,
  max_tokens=1000,
  stream=True
)

for chunk in completion:
  if not getattr(chunk, "choices", None):
    continue
  reasoning = getattr(chunk.choices[0].delta, "reasoning_content", None)
  if reasoning:
    print(reasoning, end="")
  if chunk.choices and chunk.choices[0].delta.content is not None:
    print(chunk.choices[0].delta.content, end="")

