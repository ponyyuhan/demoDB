import os
import asyncio
import threading
import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from download_and_ingest import main as ingest_main

QDRANT_HOST = os.getenv("QDRANT_HOST", "qdrant")
COLLECTION = os.getenv("COLLECTION", "sharegpt-prompts")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434")
TIMEOUT_SEC = int(os.getenv("OLLAMA_TIMEOUT", "300"))  # 5 min upper bound
EMB_MODEL = os.getenv("EMB_MODEL", "BAAI/bge-base-en-v1.5")
SCORE_THRESHOLD = float(os.getenv("SCORE_THRESHOLD", "0.35"))
MODEL_DEFAULT = os.getenv("OLLAMA_MODEL", "tinyllama:1.1b")

app = FastAPI(title="Prompt‑Augmented LLM Demo")
print(f"Using embedding model: {EMB_MODEL}")
embedder = SentenceTransformer(EMB_MODEL)
client = QdrantClient(QDRANT_HOST, port=6333)


# --- Cold‑start ingestion ----
@app.on_event("startup")
async def cold_start():
    """Start background ingestion if collection is empty"""
    def _ingest():
        try:
            # Give other services time to start
            import time
            time.sleep(5)
            print("Starting background ingestion check...")
            ingest_main()
        except Exception as e:
            print(f"[Ingest error] {e}")
    
    # Start ingestion in background thread so app can start
    threading.Thread(target=_ingest, daemon=True).start()
    print("Background ingestion thread started")


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

        # 2) Over-fetch to allow for filtering
        hits = client.search(
            collection_name=COLLECTION,
            query_vector=vector,
            limit=k*3,  # over-fetch, then filter
            with_payload=True
        )

        if not hits:
            raise HTTPException(status_code=500, detail="No vectors found - check ingestion")

        # 3) Apply score threshold and deduplication
        query_norm = text.lower().strip()
        unique = []
        seen = set()
        
        # Keep only hits with good similarity scores
        filtered = [h for h in hits if h.score >= SCORE_THRESHOLD]
        
        for hit in filtered:
            prompt_text = hit.payload.get("text", "").strip()
            prompt_key = prompt_text.lower()
            
            # Skip if identical to query or already seen
            if prompt_key == query_norm or prompt_key in seen:
                continue
                
            seen.add(prompt_key)
            unique.append(prompt_text)
            
            # Return when we have enough unique results
            if len(unique) == k:
                break
        
        print(f"Found {len(unique)} unique similar prompts (threshold: {SCORE_THRESHOLD})")
        return unique
    
    except Exception as e:
        print(f"Qdrant search failed: {e}")
        return []  # Return empty list - we want real data only


def call_ollama(prompt: str, model: str = MODEL_DEFAULT) -> str:
    """Call Ollama API for real LLM generation"""
    try:
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.2,
                "num_predict": 128      # stay small
            }
        }
        
        with httpx.Client(timeout=httpx.Timeout(TIMEOUT_SEC)) as client:
            response = client.post(f"{OLLAMA_URL}/api/generate", json=payload)
            response.raise_for_status()
            
        result = response.json()
        return result.get("response", "").strip()
        
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail=f"Ollama request timed out after {TIMEOUT_SEC}s")
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=500, detail=f"Ollama HTTP error: {e.response.status_code}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ollama connection error: {str(e)}")


# -------- API -------------------
@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    try:
        # Get similar prompts from Qdrant
        similar = await asyncio.to_thread(retrieve_similar, req.prompt, req.top_k)

        # Create prompts
        original_prompt = req.prompt
        
        # Build merged prompt only if we have similar prompts
        if similar and len(similar) > 0:
            merged_prompt = (
                "You are an AI assistant.\n"
                "Here are some similar user prompts for context (do not directly answer these, just use them as context):\n\n"
                + "\n".join([f"- {prompt}" for prompt in similar]) +
                "\n\nNow answer the following USER question:\n"
                + req.prompt
            )
        else:
            merged_prompt = original_prompt

        # Call Ollama for both original and merged prompts
        print(f"Calling Ollama for original prompt: {original_prompt[:50]}...")
        original_answer = await asyncio.to_thread(call_ollama, original_prompt)
        
        print(f"Calling Ollama for merged prompt (similar found: {len(similar)})...")
        final_answer = await asyncio.to_thread(call_ollama, merged_prompt)

        return ChatResponse(
            original_answer=original_answer,
            final_answer=final_answer,
            similar=similar,
        )
    except HTTPException:
        raise
    except Exception as e:
        print(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/")
def root():
    return {"message": "Prompt-Augmented LLM Demo is running"}
