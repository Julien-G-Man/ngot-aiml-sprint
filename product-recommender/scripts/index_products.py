# Embed product descriptions and upsert into Pinecone 
  
import os, json 
from pathlib import Path 
from dotenv import load_dotenv 
from openai import OpenAI 
from pinecone import Pinecone, ServerlessSpec
  
load_dotenv() 
  
openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY')) 
pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY')) 
index_name = os.getenv("PINECONE_INDEX_NAME", 'product-catalog')

pc.create_index(
    name=index_name,
    dimension=1536,
    metric="cosine",
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    )
)

index = pc.Index(index_name) 
  
# Load products 
with open('data/products.json') as f: 
    products = json.load(f) 
print(f'Loaded {len(products)} products') 
  
# Build text to embed: combine name + description for richer semantics 
# More text = richer embedding = better retrieval 
texts_to_embed = [ 
    f"{p['name']}. {p['description']} Category: {p['category']}." 
    for p in products 
] 
  
# Embed all products in a single batch call (cheaper + faster than one by one) 
print('Generating embeddings for all products...') 
response = openai_client.embeddings.create( 
    model='text-embedding-3-small', 
    input=texts_to_embed, 
) 
embeddings = [item.embedding for item in response.data] 
print(f'Generated {len(embeddings)} embeddings, each of dimension {len(embeddings[0])}') 
  
# Build vectors for Pinecone upsert 
# Each vector: (id, embedding_list, metadata_dict) 
vectors = [] 
for product, embedding in zip(products, embeddings): 
    vectors.append({ 
        'id':     product['id'], 
        'values': embedding, 
        'metadata': { 
            'name':        product['name'], 
            'description': product['description'][:200],  # Pinecone metadata limit 
            'category':    product['category'], 
            'price_ghs':   product['price_ghs'], 
        } 
    }) 
  
# Upsert in batches of 100 (Pinecone recommends batching) 
batch_size = 100 
for i in range(0, len(vectors), batch_size):
    batch = vectors[i:i + batch_size] 
    index.upsert(vectors=batch) 
    print(f'Upserted batch {i//batch_size + 1}: {len(batch)} vectors') 
  
# Verify 
stats = index.describe_index_stats() 
print(f'\nIndex now contains {stats.total_vector_count} vectors') 
print('Product indexing complete!') 