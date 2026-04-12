"""
rag/vector_store.py - FAISS Vector Store (dynamic dimension)
=============================================================
Detects embedding dimension at index-build time so it works
with any model (768-dim text-embedding-004 or 3072-dim gemini-embedding-001).
"""

import os, pickle
import numpy as np
import faiss
from typing import List, Dict

INDEX_PATH    = "data/faiss_index.bin"
METADATA_PATH = "data/faiss_metadata.pkl"


class VectorStore:
    def __init__(self):
        self.index     = None
        self.metadata  = []
        self.dimension = None   # set dynamically from first embedding

    def build_index(self, documents: List[Dict], embedder):
        if os.path.exists(INDEX_PATH) and os.path.exists(METADATA_PATH):
            print("💾 Loading cached FAISS index...")
            self._load_index()
            return

        print("🔨 Building new FAISS index from hospital data...")
        texts = [doc["text"] for doc in documents]
        embeddings = embedder.embed_documents(texts)
        embeddings_np = np.array(embeddings, dtype=np.float32)

        # Detect dimension from first embedding
        self.dimension = embeddings_np.shape[1]
        print(f"  Detected embedding dimension: {self.dimension}")

        faiss.normalize_L2(embeddings_np)
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings_np)
        self.metadata = documents

        os.makedirs("data", exist_ok=True)
        self._save_index()
        print(f"✅ FAISS index built with {self.index.ntotal} vectors (dim={self.dimension})")

    def search(self, query: str, embedder, top_k: int = 4) -> List[Dict]:
        if self.index is None or self.index.ntotal == 0:
            return []
        query_np = np.array([embedder.embed(query)], dtype=np.float32)
        faiss.normalize_L2(query_np)
        scores, indices = self.index.search(query_np, min(top_k, self.index.ntotal))
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            chunk = dict(self.metadata[idx])
            chunk["score"] = float(score)
            results.append(chunk)
        return results

    def _save_index(self):
        faiss.write_index(self.index, INDEX_PATH)
        with open(METADATA_PATH, "wb") as f:
            pickle.dump({"metadata": self.metadata, "dimension": self.dimension}, f)

    def _load_index(self):
        self.index = faiss.read_index(INDEX_PATH)
        with open(METADATA_PATH, "rb") as f:
            saved = pickle.load(f)
            # Support both old format (list) and new format (dict)
            if isinstance(saved, list):
                self.metadata  = saved
                self.dimension = self.index.d
            else:
                self.metadata  = saved["metadata"]
                self.dimension = saved["dimension"]
        print(f"✅ Loaded FAISS index: {self.index.ntotal} vectors (dim={self.dimension})")