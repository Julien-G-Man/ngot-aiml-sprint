# Project 4: Personalised Product Recommender

This project leverages vector similarity to move beyond traditional keyword search, allowing for semantic product discovery in an eCommerce context.

---

| Category | Details |
| :--- | :--- |
| **Project Title** | Personalised Product Recommender with Vector Similarity |
| **Sector** | eCommerce / Retail |
| **Problem** | Traditional keyword search fails on descriptive, intent-based queries (e.g., *"something cozy to wear in the rain"*) because it relies on exact word matches. |
| **Solution** | Utilizes **OpenAI embeddings** to represent products and queries in a high-dimensional vector space. It then uses **Pinecone** to perform cosine similarity searches, finding products based on semantic meaning rather than just vocabulary. |
| **Tech Stack** | OpenAI `text-embedding-3-small`, Pinecone, FastAPI, Pydantic V2, Poetry |
| **Duration** | 45 minutes (15:45 – 16:30) |
| **Deliverable** | **FastAPI Recommender API**<br>Endpoint: `http://localhost:8002`<br>Method: `POST /recommend`<br>Output: Top-N products with similarity scores |

---

### Key Features
*   **Semantic Search:** Finds products based on the context of a query.
*   **Vector Database:** Managed retrieval via Pinecone for low-latency similarity matching.
*   **Modern Python Tooling:** Built with Pydantic V2 for robust data validation and Poetry for dependency management.