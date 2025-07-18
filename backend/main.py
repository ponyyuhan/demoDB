import os, asyncio, requests
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


# --- Cold‑start ingestion (只跑一次) ----
@app.on_event("startup")
async def startup_event():
    try:
        if not client.collection_exists(COLLECTION):
            print("Collection doesn't exist, starting ingestion...")
            import ingest_prompts as ing
            await asyncio.to_thread(ing.main, limit=20000)
            print("Ingestion completed!")
        else:
            print("Collection already exists, skipping ingestion")
    except Exception as e:
        print(f"Error during startup: {e}")


# -------- Data models ----------
class ChatRequest(BaseModel):
    prompt: str
    top_k: int = 5


class ChatResponse(BaseModel):
    original_answer: str
    final_answer: str
    similar: list[str]


# -------- Helpers --------------
def retrieve_similar(text: str, k: int):
    try:
        vec = embedder.encode(text, normalize_embeddings=True).tolist()
        hits = client.search(
            collection_name=COLLECTION,
            query_vector=vec,
            limit=k,
            with_payload=True
        )
        return [h.payload["text"] for h in hits]
    except Exception as e:
        print(f"Error in retrieve_similar: {e}")
        return []  # Return empty list if search fails


def call_ollama(prompt: str) -> str:
    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": "llama3.2:3b",
                "prompt": prompt,
                "stream": False,
                "temperature": 0.2
            },
            timeout=120
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
        # Check if collection exists
        if not client.collection_exists(COLLECTION):
            raise HTTPException(status_code=503,
                                detail="Collection not ready yet, please wait for ingestion to complete")

        # Get similar prompts
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