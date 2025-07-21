import os
import asyncio
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

QDRANT_HOST = os.getenv("QDRANT_HOST", "qdrant")
COLLECTION = os.getenv("COLLECTION", "sharegpt-prompts")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434")

app = FastAPI(title="Prompt‑Augmented LLM Demo")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
client = QdrantClient(QDRANT_HOST, port=6333)


# --- Cold‑start ingestion (disabled for demo) ----
# @app.on_event("startup")
# async def startup_event():
#     try:
#         # Wait for services to be ready
#         print("Waiting for services to be ready...")
#         await asyncio.sleep(10)
#         
#         # Check if collection exists
#         try:
#             collection_exists = client.collection_exists(COLLECTION)
#             print(f"Collection exists: {collection_exists}")
#         except Exception as e:
#             print(f"Cannot check collection existence: {e}")
#             return
#             
#         if not collection_exists:
#             print("Collection doesn't exist, starting ingestion...")
#             import ingest_prompts as ing
#             await asyncio.to_thread(ing.main, limit=20000)
#             print("Ingestion completed!")
#         else:
#             print("Collection already exists, skipping ingestion")
#     except Exception as e:
#         print(f"Error during startup: {e}")


# -------- Data models ----------
class ChatRequest(BaseModel):
    prompt: str
    top_k: int = 5


class ChatResponse(BaseModel):
    original_answer: str
    final_answer: str
    similar: list[str]


# -------- Helpers --------------
def retrieve_similar(text: str, k: int = 5) -> list[str]:
    try:
        # 1) Encode prompt
        embedding = embedder.encode([text], normalize_embeddings=True)
        vector = embedding[0].tolist()

        # 2) Proper Qdrant search - using correct API for qdrant-client 1.4.0
        hits = client.search(
            collection_name=COLLECTION,
            query_vector=vector,
            limit=k,
            with_payload=True
        )

        # 3) Extract prompt strings
        return [hit.payload.get("text", "") for hit in hits]
    
    except Exception as e:
        print(f"Qdrant search failed: {e}, falling back to sample data")
        # Fallback to sample data
        try:
            import json
            from pathlib import Path
            sample_file = Path(__file__).with_name("sample_prompts.json")
            with open(sample_file, "r", encoding="utf-8") as f:
                sample_prompts = json.load(f)
            return sample_prompts[:k]  # Return first k samples
        except Exception as fallback_e:
            print(f"Error loading sample data: {fallback_e}")
            return []  # Return empty list if everything fails


def call_ollama(prompt: str) -> str:
    # For debugging: use mock response first
    if True:  # Temporarily force mock response
        return f"Mock response for: {prompt[:50]}..."
    
    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": "llama3.2:3b",
                "prompt": prompt,
                "stream": False,
                "temperature": 0.2
            },
            timeout=30
        )
        if resp.status_code != 200:
            raise HTTPException(status_code=500, detail=f"Ollama error: {resp.text}")
        return resp.json()["response"].strip()
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Connection error: {str(e)}")


# -------- API -------------------
@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    try:
        # Get similar prompts (fallback to sample if collection doesn't exist)
        similar = await asyncio.to_thread(retrieve_similar, req.prompt, req.top_k)

        # Create prompts
        original_prompt = req.prompt

        if similar:
            merged_prompt = (
                    "You are an AI assistant.\n"
                    "Here are some previous high‑similarity prompts (treated as additional context, not answers):\n"
                    + "\n---\n".join(similar) +
                    "\n---\nNow answer the USER question as accurately as possible:\nUSER: "
                    + req.prompt
            )
        else:
            merged_prompt = original_prompt

        # Call model
        original_answer = await asyncio.to_thread(call_ollama, original_prompt)
        final_answer = await asyncio.to_thread(call_ollama, merged_prompt)

        return ChatResponse(
            original_answer=original_answer,
            final_answer=final_answer,
            similar=similar,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/")
def root():
    return {"message": "Prompt-Augmented LLM Demo is running"}
