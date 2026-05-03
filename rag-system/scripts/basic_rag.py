import os
from dotenv import load_dotenv
from pathlib import Path
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Settings,
    StorageContext
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI as LlamaOpenAI

load_dotenv()

# ── Step 1: Configure global LlamaIndex settings ────────────────── 
# This sets the default LLM and embedding model for all operations 
Settings.llm = LlamaOpenAI(
    model='gpt-4o-mini',
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0.1,
)

Settings.embed_model = OpenAIEmbedding(
    model='text-embedding-3-small',
    api_key=os.getenv("OPENAI_API_KEY"),
)

Settings.node_parser = SentenceSplitter(
    chunk_size=256, 
    chunk_overlap=50
)  

# ── Step 2: Create sample medical documents ───────────────────────
# In a real system, we would load PDFs, databases, or web pages. 
# For this demo, we create text files. 
Path('data/medical').mkdir(parents=True, exist_ok=True)

  
docs_content = { 
    'malaria.txt': """ 
Malaria Diagnosis and Treatment Guidelines — Ghana Health Service 
Malaria is caused by Plasmodium parasites. Symptoms: fever, chills, sweating, 
headache. 
Diagnosis: Blood smear microscopy (gold standard) or Rapid Diagnostic Test (RDT). 
First-line treatment: Artemether-Lumefantrine (AL) for uncomplicated P. falciparum. 
Severe malaria: IV Artesunate immediately. Admit to hospital. 
Prevention: Insecticide-treated mosquito nets, indoor residual spraying. 
"""   , 

    'hypertension.txt': """ 
Hypertension Management Protocol — Primary Care 
Hypertension is defined as BP >= 140/90 mmHg on two separate occasions. 
Stage 1 (140-159/90-99): lifestyle modification first. If no improvement in 3 
months, start medication. 
Stage 2 (>=160/100): immediate pharmacological treatment required. 
First-line drugs: ACE inhibitors (Lisinopril), Calcium channel blockers 
(Amlodipine), 
Thiazide diuretics (Hydrochlorothiazide). 
Target BP: < 130/80 mmHg for most adults. < 140/90 for elderly > 65 years. 
"""   , 

    'diabetes.txt': """ 
Type 2 Diabetes Management Guidelines 
Diagnosis: Fasting glucose >= 7.0 mmol/L or HbA1c >= 48 mmol/mol (6.5%). 
First-line treatment: Metformin 500mg twice daily with meals, titrate to 2000mg/day. 
If HbA1c not controlled after 3 months: add SGLT2 inhibitor or GLP-1 agonist. 
Monitoring: HbA1c every 3 months initially, then every 6 months when stable. 
Target: HbA1c < 7% (53 mmol/mol) for most patients. 
Foot care, eye exams, and kidney function tests annually. 
"""   , 
} 

for filename, content in docs_content.items():
    Path(f'data/medical/{filename}').write_text(content.strip())
print(f"Created {len(docs_content)} sample documents")

  
# ── Step 3: Load documents ──────────────────────────────────────── 
reader = SimpleDirectoryReader('data/medical') 
documents = reader.load_data() 
print(f'Loaded {len(documents)} documents') 

  
# ── Step 4: Build index (chunks → embed → store in memory) ──────── 
print('Building index (embedding documents)...') 
# VectorStoreIndex uses an in-memory vector store by default. 
# This is fine for small datasets and demos. 
# For production: we'll use Pinecone  
index = VectorStoreIndex.from_documents(documents) 
print('Index built!') 
  
# ── Step 5: Create query engine and ask questions ───────────────── 
# The query engine handles: embed query → retrieve chunks → generate answer 
query_engine = index.as_query_engine( 
    similarity_top_k=3,       # Retrieve the 3 most similar chunks 
    response_mode='compact',  # Compact mode: fit all context in one LLM call 
) 
  
# Test queries 
test_queries = [ 
    'What is the first-line treatment for malaria in Ghana?', 
    'What BP level requires immediate medication?', 
    'How often should HbA1c be monitored in a stable diabetic patient?', 
    'What is the target blood pressure for elderly patients?', 
] 
  
for query in test_queries: 
    print(f'\nQ: {query}') 
    response = query_engine.query(query) 
    print(f'A: {response}') 
    # Show which source documents were used 
    for node in response.source_nodes: 
        print(f'  Source: {node.metadata.get("file_name", "unknown")} (score: {node.score:.3f})') 