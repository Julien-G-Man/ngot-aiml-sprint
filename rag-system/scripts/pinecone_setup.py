import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

pc = Pinecone(api_key=os.getenv("PINECONE_API_KE"))


# ── Create index for Project 3 (Medical Literature) ───────────── 
MEDICAL_INDEX = "medical-literature"
EMBED_DIM = 1536   # text-embedding-3-small produces 1536-dimensional vectors 

# Check if index already exists before creating
existing = [idx.name for idx in pc.list_indexes()]

if MEDICAL_INDEX not in existing: 
    print(f'Creating index: {MEDICAL_INDEX}')
    pc.create_index(
        name=MEDICAL_INDEX,
        dimension=EMBED_DIM,
        metric='cosine',      # cosine similarity — standard for text embeddings
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1',
        ),
    )
    print(f"Index created: {MEDICAL_INDEX}")
else:
    print(f"Index already exists: {MEDICAL_INDEX}")
    
    
# ── Create index for Project 4 (Product Recommender) ───────────── 
PRODUCT_INDEX = 'product-catalog'

if PRODUCT_INDEX not in existing:
    print(f"Creeating index: {PRODUCT_INDEX}...")
    pc.create_index(
        name=PRODUCT_INDEX,
        dimension=EMBED_DIM,
        metric='cosine',
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'
        )
    )
    print(f"Index created: {PRODUCT_INDEX}")
else: 
    print(f"Index already exists: {PRODUCT_INDEX}")
    
  
# ── Verify indexes are ready ────────────────────────────────────── 
for name in [MEDICAL_INDEX, PRODUCT_INDEX]:
    index = pc.Index(name)
    stats = index.describe_index_stats()
    print(f"{name}: {stats.total_vector_count} vectors, dimension={stats.dimension}")
    