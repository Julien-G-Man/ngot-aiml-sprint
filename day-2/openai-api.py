import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
model = "gpt-4o-mini"


def call_openai(prompt, model=model, temperature=0.1):
    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        max_tokens=200,
        messages = [
            {"role": "user",  "content": prompt},
            {"role": "system", "content": "You're an intelligent assistant. Just  testing your abilities :)"}
        ]
    )
    return response


def call_openai_with_rag(prompt, context, model=model, temperature=0.1):
    rag_response = client.chat.completions.create(
        model=model,
        temprature=temperature,
        max_tokens=200,
        messages = [
            {"role": "user", "content": prompt},
            {"role": "system", "content": 
                f"""You are a very helpful medical AI. 
                Answer accurately and precisely.
                \nContext from our document: {context}"""}
        ]
    )    
    return rag_response


def chat(prompt):
    return call_openai(prompt).choices[0].message.content


def chat_with_rag(prompt, context):
    return call_openai_with_rag(prompt, context).choices[0].message.content


def count_tokens(response):
    return response.usage.total_tokens


def calculate_cost(response):
    return count_tokens(response) * 0.00000015


if __name__ == "__main__":
    prompt = str(input("user input: "))
    resp = call_openai(prompt)
    response = chat(resp)
    total_tokens = count_tokens(resp)
    cost = calculate_cost(resp)
    print(f"AI response:  {response}")
    print(f"Total tokens: {total_tokens}")
    print(f"Cost:         {cost}")