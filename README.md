# NGOT Giants 10-Day AI/ML Sprint

This repository contains the hands-on projects built during a 10-day AI/ML and agentic systems sprint at NGOT Giants. The focus is practical, project-based learning across FastAPI apps, LLM workflows, RAG pipelines, model training, and deployment-ready project structure.

## Projects

- `telecomm-api`: telecom complaint analysis API with functional-calling, structured LLM output and validation
- `biohealth-finetune`: QLoRA fine-tuning and inference demos for biomedical NER
- `eta-predictor`: logistics ETA service with validated inputs and model-backed predictions
- `medical-literature-rag`: medical question-answering API grounded in retrieved guideline sources
- `product-recommender`: semantic product recommendation API using embeddings and Pinecone
- `rag-system`: standalone RAG demos for chunking, embeddings, retrieval, and OpenAI calls
- `day-0` to `day-2`: sprint exercises and starter tasks for Python and Pydantic

The sprint is designed to move participants from foundational Python and ML skills toward building and shipping modern AI applications.

## Agents (agent-systems)

This workspace contains two agent projects under the `agent-systems` folder that demonstrate LangChain-based agents with real tool integrations and FastAPI wrappers:

- **SpaceTech Satellite Data Summariser** — A ReAct-style agent that combines a mock satellite data store, orbital calculations, Tavily web searches, and a report formatter to produce structured intelligence reports for a given satellite. Run the service:
	- `cd agent-systems`
	- `poetry run python run.py` (starts the Satellite Intelligence API on port 8000)

- **eCommerce Smart Shopper Agent** — An agent that uses Tavily to fetch live product listings and reviews, ranks candidates with a structured tool, and returns top recommendations with reasoning. Run the shopper service:
	- `cd agent-systems`
	- `poetry run python run_shopper.py` (starts the Shopper API on port 8005)

Both agents use the modern LangChain `create_agent(...)` flow with structured tool-calling; see `agent-systems/README.md` for more details, tooling, and examples.
