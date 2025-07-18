import os, asyncio, requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models as qmodels

QDRANT_HOST = os.getenv("QDRANT_HOST", "qdrant")
COLLECTION  = os.getenv("COLLECTION", "sharegpt-prompts")
OLLAMA_URL  = os.getenv("OLLAMA_URL", "http://ollama:11434")

app = FastAPI(title="Prompt‑Augmented LLM Demo")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
client   = QdrantClient(QDRANT_HOST, port=6333)

# --- Cold‑start ingestion (只跑一次) ----
@app.on_event("startup")
async def _():
    if not client.collection_exists(COLLECTION):
        import ingest_prompts as ing
        await asyncio.to_thread(ing.main, limit=20000)  # demo 2 万条更快

# -------- Data models ----------
class ChatRequest(BaseModel):
    prompt: str
    top_k: int | None = 5

class ChatResponse(BaseModel):
    original_answer: str
    final_answer: str
    similar: list[str]

# -------- Helpers --------------
def retrieve_similar(text: str, k: int):
    vec = embedder.encode(text, normalize_embeddings=True).tolist()
    hits = client.search(
        COLLECTION,
        qmodels.SearchRequest(vector=vec, limit=k, with_payload=True)
    )
    return [h.payload["text"] for h in hits]

def call_ollama(prompt: str) -> str:
    resp = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": "llama3:8b-instruct",
            "prompt": prompt,
            "stream": False,
            "temperature": 0.2
        }, timeout=120
    )
    if resp.status_code != 200:
        raise HTTPException(status_code=500, detail=resp.text)
    return resp.json()["response"].strip()

# -------- API -------------------
@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    similar = await asyncio.to_thread(retrieve_similar, req.prompt, req.top_k)
    merged_prompt = (
        "You are an AI assistant.\n"
        "Here are some previous high‑similarity prompts (treated as additional context, not answers):\n"
        + "\n---\n".join(similar) +
        "\n---\nNow answer the USER question as accurately as possible:\nUSER: "
        + req.prompt
    )
    # call model on original and merged prompt
    original_answer = await asyncio.to_thread(call_ollama, req.prompt)
    final_answer = await asyncio.to_thread(call_ollama, merged_prompt)
    return ChatResponse(
        original_answer=original_answer,
        final_answer=final_answer,
        similar=similar,
    )

@app.get("/health")
def health(): return {"status": "ok"}
