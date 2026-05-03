# Demonstrate different chunking strategies and their effects 

from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Document


sample_text = """
Malaria is a life-threatening disease caused by Plasmodium parasites transmitted 
through the bites of infected female Anopheles mosquitoes. 
  
Clinical Presentation: 
Malaria typically presents with cyclical fever, chills, and sweating. 
The fever pattern varies by Plasmodium species: P. falciparum causes continuous 
or irregular fever, while P. vivax causes 48-hour cycles. 
Other symptoms include headache, nausea, vomiting, and muscle pain. 
Severe malaria may present with cerebral involvement, severe anaemia, or respiratory 
distress. 
  
Diagnosis: 
The gold standard for malaria diagnosis is microscopic examination of a thick and 
thin blood smear. 
Rapid Diagnostic Tests (RDTs) detecting HRP-2 antigen are widely used in resource
limited settings. 
PCR-based tests offer highest sensitivity but are rarely available in field 
conditions. 
  
Treatment: 
First-line treatment for uncomplicated P. falciparum malaria in Ghana is 
Artemisinin-Based Combination Therapy (ACT), specifically Artemether-Lumefantrine. 
Severe malaria requires intravenous Artesunate as first-line therapy. 
P. vivax and
"""

# Create a LlamaIndex Document from the text
doc = Document(text=sample_text)

# ── Strategy 1: Small chunks (aggressive splitting) ────────────── 
small_splitter = SentenceSplitter(
    chunk_size=100,    # Max tokens per chunk
    chunk_overlap=20   # Overlap between consecutive chunks 
)

small_chunks = small_splitter.get_nodes_from_documents([doc])
print(f'Small chunks (100 tokens): {len(small_chunks)} chunks')
print(f"    First chunks: '{small_chunks[0].text[:100]}...' ")


 
# ── Strategy 2: Medium chunks (recommended for most use cases) ──── 
medium_splitter = SentenceSplitter(
    chunk_size=256,
    chunk_overlap=50
)

medium_chunks = medium_splitter.get_nodes_from_documents([doc])
print(f"\nMedium chunks (256 tokens): {len(medium_chunks)} chunks")
print(f"    First chunk: '{medium_chunks[0].text[:200]}...' ")

 
# ── Strategy 3: Large chunks (for dense technical documents) ────── 
large_splitter = SentenceSplitter(
    chunk_size=512, 
    chunk_overlap=100
)
large_chunks = large_splitter.get_nodes_from_documents([doc])
print(f"\nLarge chunks (512 tokens): {len(large_chunks)} chunks")
print(f"    First chunk: '{large_chunks[0].text[:400]}...' ")

print("\nConclusion:")
print("    Small: More chunks, higher precision, less context per token")
print("    Large: Fewer chunks, more context, harder to rank precisely")
print("    For medical text: 256-512 is usually ideal ")