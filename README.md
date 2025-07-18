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
docker compose up
```

The first start downloads the model and ingests the ShareGPT dataset into Qdrant (about 20k prompts for a fast demo).

3. In another terminal, send a query:

```bash
python client/client.py "How do I bake bread?"
```

The script shows the top‑5 similar prompts found in Qdrant and prints both the
"original" answer (using only the user prompt) and the "augmented" answer (using
the merged prompt).

## Files

- `backend/` – FastAPI app and ingestion script.
- `client/` – simple CLI client.
- `docker-compose.yml` – launches Qdrant, Ollama and the backend service.

