import re
from typing import Dict, List, Tuple


def rewrite_query(query: str) -> str:
    """轻量 query rewrite：标准化输入并补充常见同义词。"""
    q = (query or "").strip()
    q = re.sub(r"\s+", " ", q)
    q_lower = q.lower()

    extras: List[str] = []
    if "身份证" in q and "idcard" not in q_lower:
        extras.append("idcard")
    if "图片" in q and "图像" not in q:
        extras.append("图像")
    if "图像" in q and "图片" not in q:
        extras.append("图片")

    if extras:
        return f"{q} {' '.join(extras)}".strip()
    return q


def infer_retrieval_filters(query: str, vlm_only: bool) -> Dict[str, bool]:
    q = (query or "").lower()
    image_intent = any(x in q for x in ["idcard", "身份证", "图片", "图里", "图像", "照片"])
    text_intent = any(x in q for x in ["笔记", "文档", "记录", "总结", "提炼", "回顾"])
    return {
        "prefer_image": image_intent,
        "prefer_text": (not image_intent) and text_intent,
        "vlm_only": bool(vlm_only),
    }


def rerank_hits(query: str, hits: List[Tuple[str, str]], prefer_image: bool = False) -> List[str]:
    """轻量 reranker：关键词重叠 + 来源类型优先级。"""
    q = (query or "").lower()
    tokens = [t for t in re.split(r"[\s,，。！？!?.:：;；/\\()\[\]{}]+", q) if len(t) >= 2]

    def score(item: Tuple[str, str]) -> int:
        source_type, content = item
        c = (content or "").lower()
        s = 0

        if prefer_image:
            if source_type == "image_note":
                s += 6
            elif source_type == "image_hit":
                s += 4
            elif source_type == "text":
                s += 1
        else:
            if source_type == "text":
                s += 4
            elif source_type == "image_note":
                s += 3
            elif source_type == "image_hit":
                s += 2

        for t in tokens:
            if t in c:
                s += 1
        if q and q in c:
            s += 2
        return s

    ranked = sorted(hits, key=score, reverse=True)
    return [content for _, content in ranked]
