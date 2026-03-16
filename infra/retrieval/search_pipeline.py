from typing import List, Tuple

from core.domain.retrieval_planner import (
    rewrite_query,
    infer_retrieval_filters,
    rerank_hits,
)
from infra.retrieval.model_reranker import rerank_with_model


def build_ranked_context(
    query: str,
    vlm_only: bool,
    text_hits: List[str],
    image_note_hits: List[str],
    image_hits: List[str],
    k: int,
) -> str:
    """统一做 query rewrite、过滤策略和 rerank，输出最终上下文。"""
    rewritten = rewrite_query(query)
    filters = infer_retrieval_filters(rewritten, vlm_only=vlm_only)

    candidates: List[Tuple[str, str]] = []
    if text_hits and not filters["vlm_only"]:
        candidates.extend([("text", h) for h in text_hits])
    if image_note_hits:
        candidates.extend([("image_note", h) for h in image_note_hits])
    if image_hits:
        candidates.extend([("image_hit", h) for h in image_hits])

    ranked = rerank_hits(rewritten, candidates, prefer_image=filters["prefer_image"])
    if filters["prefer_image"] and not filters["vlm_only"]:
        ranked = ranked[:k]
    else:
        ranked = ranked[: max(k, 4)]

    # 可选：模型级 reranker（二次排序，失败自动降级为原顺序）
    ranked = rerank_with_model(rewritten, ranked)

    if filters["vlm_only"] and not image_note_hits:
        ranked.insert(
            0,
            "[系统提示]\n用户要求‘不要OCR，仅用图片语义理解’。当前未检索到可用的图片语义理解文本（VLM）。",
        )

    return "\n\n==========\n\n".join(ranked).strip()
