# Implement hybrid search combining BM25 and dense vector search 

import os
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from rank_bm25 import BM25Okapi

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

print("\n====================================================================================================")
print("Hybrid Search script running...")
print("====================================================================================================")

# Sample medical knowledge base (in production, these would be document chunks) 
KNOWLEDGE_BASE = [ 
    {'id': 0, 'text': 'Artemether-Lumefantrine is first-line treatment for P. falciparum malaria'}, 
    {'id': 1, 'text': 'AL therapy should be taken with fatty food for better absorption'}, 
    {'id': 2, 'text': 'Malaria prevention includes insecticide-treated nets and indoor residual spraying'}, 
    {'id': 3, 'text': 'Severe malaria requires IV Artesunate and hospital admission'}, 
    {'id': 4, 'text': 'Hypertension treatment: Amlodipine 5mg daily as first-line calcium channel blocker'}, 
    {'id': 5, 'text': 'Blood pressure target below 130/80 mmHg for most adult patients'}, 
    {'id': 6, 'text': 'Metformin 500mg with meals is the first-line drug for type 2 diabetes'}, 
    {'id': 7, 'text': 'HbA1c should be checked every 3 months for newly diagnosed diabetics'}, 
]


# ── BM25 Setup ──────────────────────────────────────────────────── 
# BM25 needs tokenised text (list of words per document) 
tokenised_corpus = [doc['text'].lower().split() for doc in KNOWLEDGE_BASE] 
bm25 = BM25Okapi(tokenised_corpus) 
  
  
def embed_batch(texts: list[str]) -> list[list[float]]:
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )
    return [item.embedding for item in response.data]

def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

 
# Pre-compute document embeddings 
print("Embedding knowledge base...")
doc_embeddings = embed_batch([d['text'] for d in KNOWLEDGE_BASE])

def dense_search(query: str, top_k: int=5) -> list[tuple[int, float]]:
    """Return (doc_id, score) sorted by cosine similarity.""" 
    q_emb = embed_batch([query])[0]
    scores = [(i, cosine_similarity(q_emb, emb)) for i, emb in enumerate(doc_embeddings)]
    return sorted(scores, key=lambda x: x[1], reverse=True)[:top_k]

def bm25_search(query: str, top_k: int = 5) -> list[tuple[list, float]]:
    """Return (doc_id, score) sorted by BM25 score.""" 
    scores = bm25.get_scores(query.lower().split())
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
    return ranked

def reciprocal_rank_fusion(
    dense_results: list[tuple[int ,float]],
    bm25_results: list[tuple[int ,float]],
    k: int = 60,
    weight_dense: float = 0.7,
    weight_bm25: float = 0.3
) -> list[tuple[int, float]]: 
    """ 
    Merge two ranked lists using Reciprocal Rank Fusion. 
    Weights allow tuning how much each retriever contributes. 
    Typical: dense 0.7, bm25 0.3 (semantic search is usually better for text) 
    """ 
    rrf_scores: dict[int, float] = {}
    
    for rank, (doc_id, _) in enumerate(dense_results):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + weight_dense * (1 / (k + rank + 1))
        
    for rank, (doc_id, _) in enumerate(bm25_results):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + weight_bm25 * (1 / (k + rank + 1))
        
    return sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

# ── Test hybrid search ───────────────────────────────────────────── 
queries = [ 
    'What is AL therapy?',             # 'AL' not in dense training → BM25 helps 
    'first treatment for fever disease', # No 'malaria' keyword → dense helps 
    'Artemether-Lumefantrine dosing',   # Both should find this 
] 

for query in queries:
    print(f'\n=== QUERY: "{query}" ===')
    dense = dense_search(query, top_k=5) 
    bm25_r = bm25_search(query, top_k=5) 
    hybrid = reciprocal_rank_fusion(dense, bm25_r)[:3] 
  
    print(f'  Dense top-3:  {[(i, f"{s:.3f}") for i, s in dense[:3]]}') 
    print(f'  BM25  top-3:  {[(i, f"{s:.3f}") for i, s in bm25_r[:3]]}') 
    print(f'  Hybrid top-3: {[(i, f"{s:.4f}") for i, s in hybrid[:3]]}') 
    print(f'  Best result:  "{KNOWLEDGE_BASE[hybrid[0][0]]["text"]}"') 