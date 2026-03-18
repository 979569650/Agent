from typing import List


NO_OCR_MARKERS = ["不要ocr", "不用ocr", "别用ocr", "不使用ocr", "不要 ocr", "不用 ocr"]
LIST_NOTE_KEYWORDS = ["本地有什么笔记", "有哪些笔记", "列出笔记", "笔记列表", "本地笔记"]


def should_force_vlm_only(query: str) -> bool:
    q = (query or "").lower()
    return bool(q) and any(marker in q for marker in NO_OCR_MARKERS)


def should_list_notes_directly(query: str) -> bool:
    q = query or ""
    return bool(q) and any(keyword in q for keyword in LIST_NOTE_KEYWORDS)


def contains_restricted_keywords(query: str) -> bool:
    q = (query or "").lower()
    password_keywords: List[str] = ["密码", "password", "口令", "key", "账号", "account", "登录", "login"]
    return any(keyword in q for keyword in password_keywords)


def should_use_retrieval(query: str) -> bool:
    """判断是否应触发本地检索，避免闲聊/寒暄被强行检索污染回答。"""
    q = (query or "").strip().lower()
    if not q:
        return False

    # 常见寒暄/闲聊，直接走对话
    smalltalk_exact = {
        "你好", "您好", "嗨", "hi", "hello", "在吗", "在么", "早上好", "晚上好", "谢谢", "好的"
    }
    if q in smalltalk_exact:
        return False

    # 明确检索意图关键词
    retrieval_keywords = [
        "笔记", "本地", "查", "检索", "搜索", "总结", "提炼", "回顾", "记录", "文档", "资料",
        "idcard", "身份证", "图片", "图像", "照片", "ocr", "vlm"
    ]
    if any(k in q for k in retrieval_keywords):
        return True

    # 包含问句语气，且长度较长时，默认允许检索辅助
    question_markers = ["?", "？", "吗", "呢", "么", "什么", "怎么", "如何", "为啥", "为什么"]
    if len(q) >= 12 and any(m in q for m in question_markers):
        return True

    # 长句里出现“写过/记录/方案/找出来”等回溯意图，也默认走检索
    retrospective_markers = ["写过", "记录", "方案", "找出来", "之前", "有没有"]
    if len(q) >= 14 and any(m in q for m in retrospective_markers):
        return True

    return False

