# Prompt-Augmented LLM Demo

This repository contains a small demo that shows how to use a public prompt
database to enrich user questions before sending them to a local LLM.  The goal
is to provide additional context and reduce hallucination.  The system is split
into three services:

1. **Backend** – exposes an API that receives a user prompt, queries the prompt database for similar prompts and then calls the LLM. Implemented with FastAPI.
2. **Public context database** – a [Qdrant](https://qdrant.tech/) vector database populated with user prompts from the open dataset [RyokoAI/ShareGPT52K](https://huggingface.co/datasets/RyokoAI/ShareGPT52K).
3. **LLM** – provided by [Ollama](https://github.com/ollama/ollama) running the
   `llama3:8b-instruct` model.

When a prompt arrives the backend retrieves the five most similar prompts from
Qdrant, prepends them as additional context and sends the combined prompt to the
LLM.  For comparison the LLM is also queried with the original user prompt.  The
`client` script displays the retrieved prompts and both answers so you can see
the effect of the augmentation.

## Quick start

1. Install [Docker](https://docs.docker.com/get-docker/) and
   [docker compose](https://docs.docker.com/compose/).
2. Start all services:

```bash
docker compose up --build
```

The first start downloads the model and tries to ingest the ShareGPT dataset into
Qdrant (about 20k prompts for a fast demo). If the dataset cannot be downloaded
because of network restrictions, the backend falls back to a small set of sample
prompts stored locally so the demo still works offline.

3. In another terminal, send a query:

```bash
python client/client.py "How do I bake bread?"
```

Or test the API directly:

```bash
curl -X POST http://localhost:8000/chat \
     -H "Content-Type: application/json" \
     -d '{"prompt": "Hello, how are you?"}'
```

The script shows the top‑5 similar prompts found in Qdrant and prints both the
"original" answer (using only the user prompt) and the "augmented" answer (using
the merged prompt).

## Development and Troubleshooting

### Rebuilding after code changes

Always rebuild the backend container after making changes to the code:

```bash
# Stop all services
docker compose down

# Rebuild backend container (forces fresh build)
docker compose build --no-cache backend

# Start services
docker compose up -d
```

Or in one command:

```bash
docker compose up --build
```

### Testing the fix

If you encounter Pydantic validation errors related to Qdrant search (like "Input should be a valid number" for limit/score_threshold), the fix involves ensuring proper parameter separation in the search call:

```python
# Correct: parameters passed as separate kwargs
hits = client.search(
    collection_name=COLLECTION,
    query_vector=vector,
    limit=k,
    with_payload=True
)

# Incorrect: would cause parameters to be included in vector
hits = client.search(vector, limit=k, score_threshold=None, ...)
```

### First-time data setup

The demo includes an automatic ingestion process that runs on first startup. If you need to manually trigger data ingestion:

```bash
# Run ingestion script inside backend container
docker exec demodb-backend-1 python ingest_prompts.py
```

## Files

- `backend/` – FastAPI app and ingestion script.
- `client/` – simple CLI client.
- `docker-compose.yml` – launches Qdrant, Ollama and the backend service.


