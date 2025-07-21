"""
Download ShareGPT dataset, extract user prompts, embed them with sentence-transformers,
and upsert into the Qdrant collection `sharegpt-prompts`.
Idempotent: skips if collection already contains points.
"""

import os
import itertools
import json
import uuid
from typing import Generator, List

import tqdm
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models
from qdrant_client.http import models as rest

DATASET_ID = "RyokoAI/ShareGPT52K"  # Working dataset with 52k prompts
COLLECTION = "sharegpt-prompts"
QDRANT_HOST = os.getenv("QDRANT_HOST", "qdrant")
PORT = int(os.getenv("QDRANT_PORT", "6333"))
PROMPT_LIMIT = int(os.getenv("PROMPT_LIMIT", "20000"))  # ingest at most N prompts
EMB_MODEL = os.getenv("EMB_MODEL", "BAAI/bge-base-en-v1.5")
FORCE_RECREATE = os.getenv("FORCE_RECREATE", "false").lower() == "true"

def get_embedder():
    """Initialize embedder model"""
    print(f"Using embedding model: {EMB_MODEL} (dim 768)")
    return SentenceTransformer(EMB_MODEL)

def get_client():
    """Initialize Qdrant client"""
    return QdrantClient(QDRANT_HOST, port=PORT)

def ensure_collection(client: QdrantClient, dim: int = 768):
    """Create collection if it doesn't exist"""
    try:
        collections = [c.name for c in client.get_collections().collections]
        
        # Force recreate if switching models or explicitly requested
        if COLLECTION in collections and FORCE_RECREATE:
            print(f"Re-creating {COLLECTION} for new dimension")
            client.delete_collection(COLLECTION)
            collections.remove(COLLECTION)
        
        if COLLECTION not in collections:
            print(f"Creating collection: {COLLECTION} (dim {dim})")
            client.recreate_collection(
                COLLECTION,
                vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE)
            )
        else:
            print(f"Collection {COLLECTION} already exists")
    except Exception as e:
        print(f"Error ensuring collection: {e}")
        raise

def stream_prompts() -> Generator[str, None, None]:
    """Stream user prompts from ShareGPT dataset"""
    print(f"Loading dataset {DATASET_ID} ...")
    try:
        ds = load_dataset(DATASET_ID, split="train", streaming=True)
        count = 0
        for row in ds:
            if count >= PROMPT_LIMIT:
                break
            
            # Extract first message from conversation (always a user prompt)
            conversations = row.get("conversations", [])
            if conversations:
                text = conversations[0].get("value", "").strip()
                if text:  # Only yield non-empty prompts
                    yield text
                    count += 1
                    
    except Exception as e:
        print(f"Error loading dataset {DATASET_ID}: {e}")
        # Fallback to local sample data if dataset fails
        print("Falling back to local sample data...")
        try:
            from pathlib import Path
            sample_file = Path(__file__).with_name("sample_prompts.json")
            with open(sample_file, "r", encoding="utf-8") as f:
                sample_prompts = json.load(f)
            for prompt in sample_prompts:
                yield prompt
        except Exception as fallback_e:
            print(f"Error loading fallback data: {fallback_e}")
            # Return some default prompts if everything fails
            default_prompts = [
                "How do I bake bread?",
                "What is the capital of France?",
                "Explain the theory of relativity in simple terms.",
                "Give me some tips for learning Python.",
                "What are good strategies for time management?"
            ]
            for prompt in default_prompts:
                yield prompt

def main():
    """Main ingestion function"""
    print("Starting ShareGPT dataset ingestion...")
    
    # Initialize clients
    client = get_client()
    embedder = get_embedder()
    
    # Ensure collection exists
    ensure_collection(client)
    
    # Check if collection already has data
    try:
        current_count = client.count(COLLECTION).count
        if current_count > 0:
            print(f"Collection already populated with {current_count} prompts. Skipping ingest.")
            return
    except Exception as e:
        print(f"Warning: Could not check collection count: {e}")
    
    # Process prompts in batches
    batch_size = 256
    total_ingested = 0
    
    # Collect prompts into batches
    prompt_generator = stream_prompts()
    
    with tqdm.tqdm(total=PROMPT_LIMIT, desc="Ingesting prompts") as pbar:
        while True:
            # Collect a batch of prompts
            batch_prompts = list(itertools.islice(prompt_generator, batch_size))
            if not batch_prompts:
                break  # No more prompts
            
            try:
                # Generate embeddings for the batch
                print(f"Generating embeddings for batch of {len(batch_prompts)} prompts...")
                embeddings = embedder.encode(batch_prompts, normalize_embeddings=True, show_progress_bar=False)
                
                # Prepare batch data for Qdrant
                ids = [str(uuid.uuid4()) for _ in batch_prompts]
                vectors = embeddings.tolist()
                payloads = [{"text": prompt} for prompt in batch_prompts]
                
                # Upsert to Qdrant
                client.upsert(
                    COLLECTION,
                    rest.Batch(ids=ids, vectors=vectors, payloads=payloads)
                )
                
                total_ingested += len(batch_prompts)
                pbar.update(len(batch_prompts))
                
                # Break if we've reached the limit
                if total_ingested >= PROMPT_LIMIT:
                    break
                    
            except Exception as e:
                print(f"Error processing batch: {e}")
                # Continue with next batch
                continue
    
    print(f"Ingestion complete. Total prompts ingested: {total_ingested}")
    
    # Verify final count
    try:
        final_count = client.count(COLLECTION).count
        print(f"Final collection count: {final_count}")
    except Exception as e:
        print(f"Warning: Could not verify final count: {e}")

if __name__ == "__main__":
    main()