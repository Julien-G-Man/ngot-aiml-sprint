import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def chat(prompt):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.1,
        max_tokens=200,
        messages = [
            {"role": "user", "content": prompt},
            {"role": "system", "content": "You are an intelligent assistant designed to help people with medical issues"}
        ]
    )    
    return response.choices[0].message.content

if __name__ == "__main__":
    prompt = str(input("user message: "))
    response = chat(prompt)
    print(response)