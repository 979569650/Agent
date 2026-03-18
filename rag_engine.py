import os
import shutil
import json
import base64
import mimetypes
import time
from pathlib import Path

# 必须在导入 transformers/huggingface 之前设置镜像源
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from typing import List, Dict, Any
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import numpy as np
import faiss
from PIL import Image
from sentence_transformers import SentenceTransformer
from rapidocr_onnxruntime import RapidOCR
import requests
from infra.retrieval.search_pipeline import build_ranked_context
from infra.retrieval.storage_utils import (
    atomic_write_json,
    collect_hashes,
    create_db_snapshot,
    index_lock,
    read_recovery_marker,
    restore_db_snapshot,
    write_recovery_marker,
)


class RAGEngine:
    """
    负责本地知识库的管理：
    1. 加载 (Load): 读取 data 目录下的笔记
    2. 索引 (Index): 切分并向量化
    3. 检索 (Retrieve): 根据语义查找相关片段
    4. 自动更新 (Auto-Update): 基于文件哈希检测变化
    """

    TEXT_EXTS = {".md", ".txt"}
    IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

    def __init__(self, data_dir: str = "data", db_path: str = "faiss_index"):
        self.data_dir = data_dir
        self.db_path = db_path
        self.hash_file = os.path.join(db_path, "file_hashes.json")  # 哈希记录文件
        self.image_index_file = os.path.join(db_path, "image_clip.faiss")
        self.image_meta_file = os.path.join(db_path, "image_clip_meta.json")
        self.image_text_note_file = os.path.join(db_path, "image_text_notes.json")
        self.lock_file = os.path.join(db_path, ".index.lock")
        self.recovery_marker_file = os.path.join(db_path, ".recovery.json")
        self.snapshot_dir = os.path.join(db_path, ".snapshot")
        self.embedding_model = os.getenv(
            "EMBEDDING_MODEL", "shibing624/text2vec-base-chinese"
        )
        self.clip_model_name = os.getenv("IMAGE_EMBEDDING_MODEL", "clip-ViT-B-32")
        self.vision_model_name = os.getenv("VISION_MODEL", os.getenv("MODEL", "gpt-4o-mini"))
        self.enable_image_ocr = os.getenv("ENABLE_IMAGE_OCR", "true").lower() == "true"
        self.enable_image_vlm = os.getenv("ENABLE_IMAGE_VLM", "true").lower() == "true"
        self.request_timeout = int(os.getenv("REQUEST_TIMEOUT", "60"))
        self.embeddings = None
        self.clip_model = None
        self.ocr_engine = None
        self.vector_store = None
        self.image_index = None
        self.image_metadata: List[Dict[str, Any]] = []
        self.image_text_notes: Dict[str, Dict[str, str]] = {}
        self.rag_ready = False
        self.image_ready = False
        
        # 启动时尝试初始化向量模型；失败时降级，不阻塞主程序
        self._init_embeddings()
        self._init_image_embeddings()
        self._init_ocr_engine()
        self._recover_if_needed()

    def _recover_if_needed(self) -> None:
        marker = read_recovery_marker(self.recovery_marker_file)
        if not marker:
            return

        if marker.get("status") != "in_progress":
            return

        print("⚠️ [RAG] 检测到上次索引构建可能异常中断，开始恢复快照...")
        restored = restore_db_snapshot(
            self.snapshot_dir,
            self.db_path,
            preserve_names={".index.lock", ".recovery.json", ".snapshot"},
        )
        if restored:
            write_recovery_marker(
                self.recovery_marker_file,
                {"status": "recovered", "from": marker, "recovered_at": int(time.time())},
            )
            print("✅ [RAG] 索引快照恢复完成。")
        else:
            write_recovery_marker(
                self.recovery_marker_file,
                {"status": "recovery_failed", "from": marker, "recovered_at": int(time.time())},
            )
            print("❌ [RAG] 索引快照恢复失败，请手动执行 update 触发重建。")

    def _init_embeddings(self) -> bool:
        """初始化 Embedding 模型。失败时返回 False，不抛出异常。"""
        if self.embeddings is not None:
            self.rag_ready = True
            return True

        print(f"📥 [RAG] 正在初始化 Embedding 模型 ({self.embedding_model})...")
        try:
            self.embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model)
            self.rag_ready = True
            return True
        except Exception as e:
            self.rag_ready = False
            self.embeddings = None
            print("⚠️ [RAG] Embedding 模型初始化失败，已自动降级为无 RAG 模式。")
            print(f"   失败原因: {type(e).__name__}: {e}")
            print("   可能原因: 无法连接 HuggingFace / 镜像源超时 / 网络受限。")
            print("   你仍可继续对话，但 search_notes 将暂时不可用。")
            return False

    def _init_image_embeddings(self) -> bool:
        """初始化 CLIP 图文向量模型。失败时返回 False，不抛异常。"""
        if self.clip_model is not None:
            self.image_ready = True
            return True

        print(f"🖼️ [RAG] 正在初始化多模态模型 ({self.clip_model_name})...")
        try:
            from sentence_transformers import SentenceTransformer
            from transformers import CLIPProcessor

            original_from_pretrained = CLIPProcessor.from_pretrained

            def _load_with_mode(use_fast: bool):
                def _patched_from_pretrained(*args, **kwargs):
                    if ("use_fast" not in kwargs) or (kwargs.get("use_fast") is None):
                        kwargs["use_fast"] = use_fast
                    return original_from_pretrained(*args, **kwargs)

                CLIPProcessor.from_pretrained = _patched_from_pretrained
                try:
                    return SentenceTransformer(self.clip_model_name)
                finally:
                    CLIPProcessor.from_pretrained = original_from_pretrained

            try:
                self.clip_model = _load_with_mode(use_fast=True)
            except ImportError as e:
                if "Torchvision" not in str(e):
                    raise
                print("⚠️ [RAG] 未检测到 torchvision，已回退慢速图片处理器。")
                self.clip_model = _load_with_mode(use_fast=False)

            self.image_ready = True
            return True
        except Exception as e:
            self.image_ready = False
            self.clip_model = None
            print("⚠️ [RAG] 多模态模型初始化失败，图片检索功能已降级关闭。")
            print(f"   失败原因: {type(e).__name__}: {e}")
            return False

    def _init_ocr_engine(self) -> bool:
        """初始化 OCR 引擎，用于把图片内容转为可检索文本。"""
        if not self.enable_image_ocr:
            return False
        if self.ocr_engine is not None:
            return True
        try:
            self.ocr_engine = RapidOCR()
            return True
        except Exception as e:
            print(f"⚠️ [RAG] OCR 引擎初始化失败，图片文本抽取已关闭: {type(e).__name__}: {e}")
            self.ocr_engine = None
            return False

    def _extract_image_text(self, image_path: str) -> str:
        """OCR 抽取单张图片文字。失败返回空字符串。"""
        if not self._init_ocr_engine():
            return ""
        try:
            result, _ = self.ocr_engine(image_path)
            if not result:
                return ""
            lines = []
            for item in result:
                if not item or len(item) < 2:
                    continue
                text = item[1]
                if text and isinstance(text, str):
                    lines.append(text.strip())
            return "\n".join([x for x in lines if x])
        except Exception as e:
            print(f"⚠️ [RAG] OCR 失败: {os.path.basename(image_path)} - {e}")
            return ""

    def _describe_image_with_vlm(self, image_path: str) -> str:
        """调用多模态大模型理解图片内容，返回可检索文本。"""
        if not self.enable_image_vlm:
            return ""

        base_url = (os.getenv("BASE_URL") or "").rstrip("/")
        api_key = os.getenv("API_KEY")
        if not base_url or not api_key:
            return ""

        try:
            with open(image_path, "rb") as f:
                image_bytes = f.read()
            mime = mimetypes.guess_type(image_path)[0] or "image/jpeg"
            b64 = base64.b64encode(image_bytes).decode("utf-8")
            data_url = f"data:{mime};base64,{b64}"

            payload = {
                "model": self.vision_model_name,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "你是图像理解助手。请输出中文结构化摘要，包含："
                            "1) 图片类型；2) 关键可见元素；3) 可读文字（若有）；"
                            "4) 可用于笔记检索的关键词。避免编造。"
                        ),
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "请详细理解这张图片并生成可检索摘要。"},
                            {"type": "image_url", "image_url": {"url": data_url}},
                        ],
                    },
                ],
                "temperature": 0.2,
            }

            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }

            resp = requests.post(
                f"{base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=self.request_timeout,
            )
            if resp.status_code >= 400:
                print(f"⚠️ [RAG] 图片语义理解失败: HTTP {resp.status_code}")
                return ""

            data = resp.json()
            choices = data.get("choices", []) or []
            if not choices:
                return ""
            content = choices[0].get("message", {}).get("content", "")
            return content.strip() if isinstance(content, str) else str(content)
        except Exception as e:
            print(f"⚠️ [RAG] 图片语义理解异常: {os.path.basename(image_path)} - {e}")
            return ""

    def _get_current_hashes(self) -> Dict[str, str]:
        """扫描目录，获取当前所有文件的哈希值"""
        return collect_hashes(self.data_dir, self.TEXT_EXTS | self.IMAGE_EXTS)

    def check_for_updates(self) -> bool:
        """检查是否有文件变动（新增、修改、删除）"""
        print("🔍 [RAG] 正在检查文档变更...")
        
        # 1. 获取当前磁盘状态
        current_hashes = self._get_current_hashes()
        
        # 2. 获取上次记录的状态
        saved_hashes = {}
        if os.path.exists(self.hash_file):
            try:
                with open(self.hash_file, 'r', encoding='utf-8') as f:
                    saved_hashes = json.load(f)
            except Exception as e:
                print(f"⚠️ 读取哈希记录失败，视为需全量更新: {e}")
        
        # 3. 比较差异
        if current_hashes == saved_hashes:
            print("✅ 文档无变更，无需更新索引。")
            return False
            
        # 打印具体变更（可选）
        added = set(current_hashes.keys()) - set(saved_hashes.keys())
        deleted = set(saved_hashes.keys()) - set(current_hashes.keys())
        modified = {k for k in current_hashes if k in saved_hashes and current_hashes[k] != saved_hashes[k]}
        
        if added: print(f"  ➕ 新增: {len(added)} 个文件")
        if deleted: print(f"  ➖ 删除: {len(deleted)} 个文件")
        if modified: print(f"  ✏️ 修改: {len(modified)} 个文件")
        
        return True

    def build_index(self):
        """重建索引：文本向量索引 + 图片向量索引"""
        with index_lock(self.db_path, self.lock_file):
            os.makedirs(self.db_path, exist_ok=True)
            create_db_snapshot(
                self.db_path,
                self.snapshot_dir,
                exclude_names={".index.lock", ".recovery.json", ".snapshot"},
            )
            write_recovery_marker(
                self.recovery_marker_file,
                {"status": "in_progress", "started_at": int(time.time())},
            )

            success = False
            text_ok = self._init_embeddings()
            image_ok = self._init_image_embeddings()
            if not text_ok and not image_ok:
                print("⚠️ [RAG] 跳过索引构建：文本与图片向量模型均不可用。")
                write_recovery_marker(
                    self.recovery_marker_file,
                    {"status": "skipped", "reason": "both_models_unavailable", "finished_at": int(time.time())},
                )
                return

            print("📚 [RAG] 正在构建/重建索引...")

            try:
                docs = []
                image_files: List[str] = []
                image_text_notes: Dict[str, Dict[str, str]] = {}
                if not os.path.exists(self.data_dir):
                    os.makedirs(self.data_dir)

                for root, dirs, files in os.walk(self.data_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        ext = Path(file).suffix.lower()

                        if ext in self.TEXT_EXTS and text_ok:
                            try:
                                loader = TextLoader(file_path, encoding="utf-8")
                                docs.extend(loader.load())
                                print(f"  ✅ 文本加载成功: {file}")
                            except UnicodeDecodeError:
                                try:
                                    loader = TextLoader(file_path, encoding="gbk")
                                    docs.extend(loader.load())
                                    print(f"  ✅ 文本加载成功 (GBK): {file}")
                                except Exception as e:
                                    print(f"  ❌ 文本加载失败 (编码问题): {file} - {e}")
                            except Exception as e:
                                print(f"  ❌ 文本加载失败: {file} - {e}")

                        if ext in self.IMAGE_EXTS and image_ok:
                            image_files.append(file_path)
                            rel_path = os.path.relpath(file_path, self.data_dir)
                            image_text_notes.setdefault(rel_path, {"vlm": "", "ocr": ""})

                            if text_ok and self.enable_image_vlm:
                                vlm_text = self._describe_image_with_vlm(file_path)
                                if vlm_text:
                                    docs.append(
                                        Document(
                                            page_content=f"[图片语义理解]\n{vlm_text}",
                                            metadata={"source": rel_path, "type": "image_vlm"},
                                        )
                                    )
                                    image_text_notes[rel_path]["vlm"] = vlm_text
                                    print(f"  ✅ 图片语义理解成功并入库: {rel_path}")
                                else:
                                    print(f"  ⚠️ 图片语义理解无结果: {rel_path}")

                            if text_ok and self.enable_image_ocr:
                                ocr_text = self._extract_image_text(file_path)
                                if ocr_text:
                                    docs.append(
                                        Document(
                                            page_content=f"[图片OCR内容]\n{ocr_text}",
                                            metadata={"source": rel_path, "type": "image_ocr"},
                                        )
                                    )
                                    image_text_notes[rel_path]["ocr"] = ocr_text
                                    print(f"  ✅ 图片OCR成功并入库: {rel_path}")
                                else:
                                    print(f"  ⚠️ 图片OCR无文本: {rel_path}")

                if docs and text_ok:
                    print(f"📄 加载了 {len(docs)} 个文档。正在切分...")
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                    splits = text_splitter.split_documents(docs)

                    print(f"🔢 切分出 {len(splits)} 个文本片段。正在向量化...")
                    self.vector_store = FAISS.from_documents(splits, self.embeddings)
                    self.vector_store.save_local(self.db_path)
                elif text_ok:
                    print("⚠️ 未找到可索引文本（.md/.txt），跳过文本索引。")

                self._build_image_index(image_files)
                self._save_image_text_notes(image_text_notes)

                current_hashes = self._get_current_hashes()
                atomic_write_json(self.hash_file, current_hashes)
                print("✅ 索引构建完成并已保存！(Hash Updated)")
                success = True
            except Exception as e:
                print(f"❌ 索引构建失败，尝试回滚: {e}")
                restored = restore_db_snapshot(
                    self.snapshot_dir,
                    self.db_path,
                    preserve_names={".index.lock", ".recovery.json", ".snapshot"},
                )
                write_recovery_marker(
                    self.recovery_marker_file,
                    {
                        "status": "rollback_done" if restored else "rollback_failed",
                        "error": f"{type(e).__name__}: {e}",
                        "finished_at": int(time.time()),
                    },
                )
                raise
            finally:
                if success:
                    write_recovery_marker(
                        self.recovery_marker_file,
                        {"status": "success", "finished_at": int(time.time())},
                    )
                    try:
                        if os.path.exists(self.recovery_marker_file):
                            os.remove(self.recovery_marker_file)
                        if os.path.exists(self.snapshot_dir):
                            shutil.rmtree(self.snapshot_dir, ignore_errors=True)
                    except Exception:
                        pass

    def _build_image_index(self, image_files: List[str]):
        """构建 CLIP 图片向量索引。"""
        if not self._init_image_embeddings():
            print("⚠️ [RAG] 跳过图片索引构建：多模态模型不可用。")
            return

        if not image_files:
            print("⚠️ 未找到可索引图片（png/jpg/jpeg/webp/bmp），跳过图片索引。")
            self.image_index = None
            self.image_metadata = []
            return

        vectors = []
        metadata = []
        print(f"🖼️ 正在向量化 {len(image_files)} 张图片...")
        for path in image_files:
            rel_path = os.path.relpath(path, self.data_dir)
            try:
                img = Image.open(path).convert("RGB")
                vec = self.clip_model.encode(img, normalize_embeddings=True)
                vec = np.array(vec, dtype="float32")
                vectors.append(vec)
                metadata.append({"source": rel_path})
                print(f"  ✅ 图片向量化成功: {rel_path}")
            except Exception as e:
                print(f"  ❌ 图片向量化失败: {rel_path} - {e}")

        if not vectors:
            print("⚠️ 图片向量化全部失败，未生成图片索引。")
            self.image_index = None
            self.image_metadata = []
            return

        matrix = np.vstack(vectors).astype("float32")
        dim = matrix.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(matrix)

        os.makedirs(self.db_path, exist_ok=True)
        faiss.write_index(index, self.image_index_file)
        atomic_write_json(self.image_meta_file, metadata)

        self.image_index = index
        self.image_metadata = metadata
        print(f"✅ 图片索引构建完成：{len(metadata)} 张")

    def _save_image_text_notes(self, notes: Dict[str, Dict[str, str]]):
        """保存图片文本资源（VLM/OCR）到本地，便于检索兜底。"""
        self.image_text_notes = notes or {}
        try:
            atomic_write_json(self.image_text_note_file, self.image_text_notes)
        except Exception as e:
            print(f"⚠️ 图片文本资源保存失败: {e}")

    def _load_image_text_notes(self):
        """加载图片文本资源（VLM/OCR）。"""
        if not os.path.exists(self.image_text_note_file):
            self.image_text_notes = {}
            return
        try:
            with open(self.image_text_note_file, "r", encoding="utf-8") as f:
                self.image_text_notes = json.load(f)
        except Exception as e:
            print(f"⚠️ 图片文本资源加载失败: {e}")
            self.image_text_notes = {}

    def _load_image_index(self) -> bool:
        """加载图片向量索引与元数据。"""
        if not self._init_image_embeddings():
            return False

        if not (os.path.exists(self.image_index_file) and os.path.exists(self.image_meta_file)):
            return False

        try:
            self.image_index = faiss.read_index(self.image_index_file)
            with open(self.image_meta_file, "r", encoding="utf-8") as f:
                self.image_metadata = json.load(f)
            self._load_image_text_notes()
            return True
        except Exception as e:
            print(f"⚠️ 图片索引加载失败: {e}")
            self.image_index = None
            self.image_metadata = []
            return False

    def load_index(self) -> bool:
        """加载已保存的索引，并自动检查增量更新"""
        text_ok = self._init_embeddings()
        image_ok = self._init_image_embeddings()
        if not text_ok and not image_ok:
            return False

        # 1. 检查是否需要更新 (增量检测)
        if self.check_for_updates():
            self.build_index()
            text_loaded = False
            image_loaded = False
            if text_ok and os.path.exists(self.db_path):
                try:
                    self.vector_store = FAISS.load_local(
                        self.db_path, self.embeddings, allow_dangerous_deserialization=True
                    )
                    text_loaded = True
                except Exception as e:
                    print(f"⚠️ 重建后文本索引加载失败: {e}")
            if image_ok:
                image_loaded = self._load_image_index()
            return text_loaded or image_loaded

        # 2. 正常加载
        text_loaded = False
        image_loaded = False

        if text_ok and os.path.exists(self.db_path):
            try:
                self.vector_store = FAISS.load_local(
                    self.db_path, self.embeddings, allow_dangerous_deserialization=True
                )
                text_loaded = True
            except Exception as e:
                print(f"⚠️ 索引加载失败: {e}")
                # 加载失败可能是索引损坏，尝试重建
                self.build_index()
                if text_ok:
                    try:
                        self.vector_store = FAISS.load_local(
                            self.db_path, self.embeddings, allow_dangerous_deserialization=True
                        )
                        text_loaded = True
                    except Exception:
                        text_loaded = False
        elif text_ok:
            self.build_index()
            try:
                self.vector_store = FAISS.load_local(
                    self.db_path, self.embeddings, allow_dangerous_deserialization=True
                )
                text_loaded = True
            except Exception:
                text_loaded = False

        if image_ok:
            image_loaded = self._load_image_index()
            if not image_loaded:
                self.build_index()
                image_loaded = self._load_image_index()

        return text_loaded or image_loaded

    def _search_text(self, query: str, k: int = 6) -> List[str]:
        """文本检索。"""
        if not self._init_embeddings():
            return []
        if not self.vector_store and not self.load_index():
            return []
        if not self.vector_store:
            return []

        try:
            results = self.vector_store.similarity_search(query, k=k)
            return [
                f"[文本来源: {os.path.basename(doc.metadata.get('source', '未知'))}]\n{doc.page_content}"
                for doc in results
            ]
        except Exception as e:
            print(f"文本检索出错: {e}")
            return []

    def _search_images(self, query: str, k: int = 4, vlm_only: bool = False) -> List[str]:
        """图片语义检索：文本 query -> CLIP 向量 -> 图片近邻。"""
        if not self._init_image_embeddings():
            return []
        if self.image_index is None and not self.load_index():
            return []
        if self.image_index is None:
            return []

        try:
            q = self.clip_model.encode(query, normalize_embeddings=True)
            q = np.array([q], dtype="float32")
            top_k = min(k, self.image_index.ntotal)
            if top_k <= 0:
                return []

            scores, indices = self.image_index.search(q, top_k)
            hits = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < 0 or idx >= len(self.image_metadata):
                    continue
                src = self.image_metadata[idx].get("source", "未知")
                ocr_hint = ""
                if self.enable_image_ocr and not vlm_only:
                    ocr_hint = "（若启用OCR，可继续追问图片中的文字内容）"
                hits.append(f"[图片来源: {src}]\n语义相似度: {float(score):.4f} {ocr_hint}")
            return hits
        except Exception as e:
            print(f"图片检索出错: {e}")
            return []

    def _search_image_text_notes(self, query: str, k: int = 2, vlm_only: bool = False) -> List[str]:
        """基于图片文件名与文本内容做轻量关键词匹配，兜底返回图片文本资源。"""
        if not self.image_text_notes:
            self._load_image_text_notes()
        if not self.image_text_notes:
            return []

        q = (query or "").lower().strip()
        if not q:
            return []

        scored = []
        for src, payload in self.image_text_notes.items():
            vlm = (payload.get("vlm") or "").strip()
            ocr = (payload.get("ocr") or "").strip()
            if vlm_only and not vlm:
                continue
            merged = f"{vlm}\n{ocr}".strip()
            if not merged:
                continue

            score = 0
            src_l = src.lower()
            if q in src_l:
                score += 3
            for token in q.replace("，", " ").replace("。", " ").split():
                if token and token in src_l:
                    score += 1
                if token and token in merged.lower():
                    score += 1
            # 针对“身份证/idcard/图片内容”等意图给图片文本更高优先级
            if any(x in q for x in ["idcard", "身份证", "图片", "图里", "图像"]):
                score += 2

            if score > 0:
                scored.append((score, src, vlm, ocr))

        scored.sort(key=lambda x: x[0], reverse=True)
        hits = []
        for _, src, vlm, ocr in scored[:k]:
            parts = [f"[图片文本来源: {src}]"]
            if vlm:
                parts.append(f"[图片语义理解]\n{vlm}")
            if ocr and not vlm_only:
                parts.append(f"[图片OCR内容]\n{ocr}")
            hits.append("\n".join(parts))
        return hits

    def search(self, query: str, k: int = 8, vlm_only: bool = False) -> str:
        """多模态检索接口（文本 + 图片双索引融合）。"""
        # 召回阶段
        text_hits = self._search_text(query, k=max(2, k - 2))
        image_note_hits = self._search_image_text_notes(query, k=2, vlm_only=vlm_only)
        image_hits = self._search_images(query, k=min(4, k), vlm_only=vlm_only)

        # 规划/排序阶段（已抽离到 infra.retrieval）
        full_context = build_ranked_context(
            query=query,
            vlm_only=vlm_only,
            text_hits=text_hits,
            image_note_hits=image_note_hits,
            image_hits=image_hits,
            k=k,
        )
        if not full_context:
            return ""

        # 安全阀：限制返回字符，防止 Token 溢出
        if len(full_context) > 4500:
            print(f"⚠️ 检索内容过长 ({len(full_context)} 字符)，已截断至 4500 字符。")
            return full_context[:4500] + "\n...(内容已截断)"
        return full_context

    def list_note_files(self) -> List[str]:
        """返回当前 data 目录可见的笔记/图片文件列表（相对路径）。"""
        try:
            hashes = self._get_current_hashes()
            return sorted(hashes.keys())
        except Exception as e:
            print(f"列出笔记文件失败: {e}")
            return []
