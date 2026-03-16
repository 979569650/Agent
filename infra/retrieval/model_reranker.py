import os
from functools import lru_cache
from typing import List

import numpy as np


class _NoopReranker:
    enabled = False

    def rerank(self, _: str, candidates: List[str]) -> List[str]:
        return candidates


class _EmbeddingReranker:
    enabled = True

    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)

    def rerank(self, query: str, candidates: List[str]) -> List[str]:
        if not candidates:
            return candidates
        qv = self.model.encode(query or "", normalize_embeddings=True)
        dvs = self.model.encode(candidates, normalize_embeddings=True)
        scores = np.dot(dvs, qv)
        ranked = sorted(zip(scores.tolist(), candidates), key=lambda x: x[0], reverse=True)
        return [x[1] for x in ranked]


@lru_cache(maxsize=1)
def _get_reranker():
    enabled = (os.getenv("RERANKER_ENABLED", "false") or "false").strip().lower() == "true"
    if not enabled:
        return _NoopReranker()
    model_name = (os.getenv("RERANKER_MODEL") or "BAAI/bge-reranker-base").strip()
    try:
        return _EmbeddingReranker(model_name)
    except Exception:
        return _NoopReranker()


def rerank_with_model(query: str, candidates: List[str]) -> List[str]:
    return _get_reranker().rerank(query, candidates)
