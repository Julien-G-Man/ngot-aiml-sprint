import os 
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))


def embed(text: str) -> list[float]:
    """Convert text to a vector using OpenAI text-embedding-3-small.
    text-embedding-3-small: 1536 dimensions, $0.02 per 1M tokens 
    text-embedding-3-large: 3072 dimensions, more accurate but 5x more expensive
    """ 
    
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding  # A list of 1536 floats


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    a = np.array(vec_a)
    b = np.array(vec_b)
    return float(np.dot(a, b) / np.linalg.norm(a) * (np.linalg.norm(b)))

 
# Test with medical sentences 
sentences = [ 
    'The patient presents with high fever and chills',      # [0] query 
    'Elevated temperature and rigors are present',          # [1] similar meaning 
    'Malaria symptoms include fever, sweating, and nausea', # [2] related 
    'Administer 500mg paracetamol every 6 hours',           # [3] treatment 
    'The stock market fell 3% today',                       # [4] completely unrelated 
] 

if __name__ == "__main__":
    print("Generating embeddings...")
    vectors = [embed(s) for s in sentences]
    print(f"Each vector has {len(vectors[0])} dimensions")
    
    print("\n====SIMILARITY TO QUERY: sentences[0] ====")
    query = sentences[0]
    for i, (text, vec) in enumerate(zip(sentences, vectors)):
        sim = cosine_similarity(vectors[0], vec)
        print(f"  [{i}] sim={sim:.3f} '{text[:50]}...' " if len(text) > 50 
              else f" [{i}] sim={sim:.3f} '{text}' ")
        # Expected output shows [1] and [2] are most similar to [0] 
        # and [4] (stock market) has very low similarity