"""
首次容器启动时自动执行：
1. 下载 RyokoAI/ShareGPT52K
2. 抽取所有 user prompts
3. 生成嵌入，批量写入 Qdrant
总计约 90k 条，
"""
import os, itertools, uuid, tqdm
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from datasets import load_dataset

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
COLLECTION   = os.getenv("COLLECTION", "sharegpt-prompts")

def main(limit=None):
    client = QdrantClient(QDRANT_HOST, port=6333)
    if client.collection_exists(COLLECTION):
        print("Collection already exists, skip ingestion")
        return

    # 1. 创建 collection
    client.create_collection(
        COLLECTION,
        vectors_config={"size": 384, "distance": "Cosine"}
    )

    # 2. 加载数据
    ds = load_dataset("RyokoAI/ShareGPT52K", split="train")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    batch_size = 512
    points = []

    for i, record in tqdm.tqdm(enumerate(ds), total=(len(ds) if limit is None else limit)):
        if limit and i >= limit: break
        for turn in record["conversations"]:
            if turn["from"] == "human":
                prompt = turn["value"].strip()
                points.append(prompt)
    print(f"Collected {len(points)} user prompts")

    # 3. 批量写入
    for batch_idx in range(0, len(points), batch_size):
        sub = points[batch_idx : batch_idx+batch_size]
        embeds = model.encode(sub, show_progress_bar=False, batch_size=64, normalize_embeddings=True).tolist()
        client.upload_collection(
            collection_name=COLLECTION,
            vectors=embeds,
            payload=[{"text": p} for p in sub],
            ids=[uuid.uuid4().hex for _ in sub],
            batch_size=batch_size
        )
    print("Ingestion finished ✔")

if __name__ == "__main__":
    main(limit=None)  # 可修改 limit=20000 进行快速演示
