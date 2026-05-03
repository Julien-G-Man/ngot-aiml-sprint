# Use Cohere's re-ranking API to improve retrieval precision 

import os
import cohere 
from dotenv import load_dotenv 

load_dotenv()
co = cohere.Client(api_key=os.getenv('COHERE_API_KEY')) 
  
# Our knowledge base
DOCUMENTS = [ 
    'Artemether-Lumefantrine is first-line for P. falciparum malaria. Take with fatty food.', 
    'AL therapy absorption improves significantly with co-administration of foodcontaining fat.', 
    'Malaria prevention: ITNs (insecticide-treated nets) reduce transmission by 50%.', 
    'Severe malaria: immediate IV Artesunate, hospital admission, monitor closely.', 
    'Hypertension: Amlodipine 5mg daily, target BP below 130/80 mmHg.', 
    'Metformin 500mg twice daily is first-line pharmacotherapy for type 2 diabetes.', 
    'HbA1c monitoring every 3 months for newly diagnosed diabetes patients.', 
    'Primaquine required for P. vivax and P. ovale to eliminate liver hypnozoites.', 
]  


query = 'How do I improve the effectiveness of malaria medication?' 
  
# ── Without re-ranking: just return all documents ──────────────── 
print('Without re-ranking (raw retrieval order):') 
for i, doc in enumerate(DOCUMENTS[:5]): 
    print(f'  [{i}] {doc[:80]}...') 
 
# ── With Cohere re-ranking ──────────────────────────────────────── 
print(f'\nRe-ranking for query: "{query}"') 

try:
    results = co.rerank( 
        query=query, 
        documents=DOCUMENTS, 
        model='rerank-english-v3.0', 
        top_n=5,   # Return top 5 after re-ranking 
    ) 
except Exception as e:
    print(e)

  
print('\nAfter Cohere re-ranking (top 5):') 
for i, r in enumerate(results.results): 
    print(f'  [{i}] score={r.relevance_score:.4f}  doc_idx={r.index}') 
    print(f'       "{DOCUMENTS[r.index][:90]}"') 
  
  
# Expected: docs about 'fatty food' and 'absorption' should rank high 
# because they are most relevant to 'improve effectiveness'