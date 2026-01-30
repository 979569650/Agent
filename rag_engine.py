import os
import hashlib
import json

# 必须在导入 transformers/huggingface 之前设置镜像源
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from typing import List, Dict
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document


class RAGEngine:
    """
    负责本地知识库的管理：
    1. 加载 (Load): 读取 data 目录下的笔记
    2. 索引 (Index): 切分并向量化
    3. 检索 (Retrieve): 根据语义查找相关片段
    4. 自动更新 (Auto-Update): 基于文件哈希检测变化
    """

    def __init__(self, data_dir: str = "data", db_path: str = "faiss_index"):
        self.data_dir = data_dir
        self.db_path = db_path
        self.hash_file = os.path.join(db_path, "file_hashes.json")  # 哈希记录文件
        
        # 更换为中文专用 Embedding 模型，解决中文语义匹配不准的问题
        # shibing624/text2vec-base-chinese 是目前效果最好的开源中文模型之一
        print(
            "📥 [RAG] 正在初始化 Embedding 模型 (shibing624/text2vec-base-chinese)..."
        )
        self.embeddings = HuggingFaceEmbeddings(
            model_name="shibing624/text2vec-base-chinese"
        )
        self.vector_store = None

    def _calculate_file_hash(self, filepath: str) -> str:
        """计算单个文件的 MD5 哈希值"""
        hasher = hashlib.md5()
        with open(filepath, 'rb') as f:
            buf = f.read()
            hasher.update(buf)
        return hasher.hexdigest()

    def _get_current_hashes(self) -> Dict[str, str]:
        """扫描目录，获取当前所有文件的哈希值"""
        current_hashes = {}
        if not os.path.exists(self.data_dir):
            return current_hashes
            
        for root, dirs, files in os.walk(self.data_dir):
            for file in files:
                if file.endswith(".md") or file.endswith(".txt"):
                    file_path = os.path.join(root, file)
                    # 使用相对路径作为 Key，避免绝对路径变化导致的问题
                    rel_path = os.path.relpath(file_path, self.data_dir)
                    try:
                        current_hashes[rel_path] = self._calculate_file_hash(file_path)
                    except Exception as e:
                        print(f"⚠️ 无法读取文件哈希: {file} - {e}")
        return current_hashes

    def check_for_updates(self) -> bool:
        """检查是否有文件变动（新增、修改、删除）"""
        print("� [RAG] 正在检查文档变更...")
        
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
        """重建索引：遍历目录 -> 读取 -> 切分 -> 向量化 -> 保存"""
        print(f"📚 [RAG] 正在重建索引 (检测到变更)...")

        docs = []
        # 手动遍历目录，确保编码控制 (Windows下 UTF-8 兼容性)
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

        for root, dirs, files in os.walk(self.data_dir):
            for file in files:
                if file.endswith(".md") or file.endswith(".txt"):
                    file_path = os.path.join(root, file)
                    try:
                        # 尝试 UTF-8 加载
                        loader = TextLoader(file_path, encoding="utf-8")
                        docs.extend(loader.load())
                        print(f"  ✅ 加载成功: {file}")
                    except UnicodeDecodeError:
                        try:
                            # 失败则尝试 GBK (兼容 Windows 旧文件)
                            loader = TextLoader(file_path, encoding="gbk")
                            docs.extend(loader.load())
                            print(f"  ✅ 加载成功 (GBK): {file}")
                        except Exception as e:
                            print(f"  ❌ 加载失败 (编码问题): {file} - {e}")
                    except Exception as e:
                        print(f"  ❌ 加载失败: {file} - {e}")

        if not docs:
            print("⚠️ 未找到文档，跳过索引构建。请在 data/ 目录下添加 .md 笔记。")
            return

        print(f"📄 加载了 {len(docs)} 个文档。正在切分...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        splits = text_splitter.split_documents(docs)

        print(f"🔢 切分出 {len(splits)} 个片段。正在向量化...")
        self.vector_store = FAISS.from_documents(splits, self.embeddings)

        # 本地持久化
        self.vector_store.save_local(self.db_path)
        
        # 保存新的文件哈希记录
        current_hashes = self._get_current_hashes()
        try:
            with open(self.hash_file, 'w', encoding='utf-8') as f:
                json.dump(current_hashes, f, indent=2, ensure_ascii=False)
            print("✅ 索引构建完成并已保存！(Hash Updated)")
        except Exception as e:
            print(f"⚠️ 索引已保存，但哈希记录写入失败: {e}")

    def load_index(self) -> bool:
        """加载已保存的索引，并自动检查增量更新"""
        # 1. 检查是否需要更新 (增量检测)
        if self.check_for_updates():
            self.build_index()
            # 重建后重新加载
            try:
                self.vector_store = FAISS.load_local(
                    self.db_path, self.embeddings, allow_dangerous_deserialization=True
                )
                return True
            except Exception as e:
                print(f"⚠️ 重建后加载失败: {e}")
                return False

        # 2. 正常加载
        if os.path.exists(self.db_path):
            try:
                self.vector_store = FAISS.load_local(
                    self.db_path, self.embeddings, allow_dangerous_deserialization=True
                )
                return True
            except Exception as e:
                print(f"⚠️ 索引加载失败: {e}")
                # 加载失败可能是索引损坏，尝试重建
                self.build_index()
                return True
        else:
            # 索引不存在，构建
            self.build_index()
            return True

    def search(self, query: str, k: int = 8) -> str:
        """检索接口"""
        if not self.vector_store:
            if not self.load_index():
                return ""

        try:
            # 增加检索数量 k=8，提高召回率，防止关键信息漏掉
            results = self.vector_store.similarity_search(query, k=k)

            # 拼接内容，并添加来源信息
            context_list = [
                f"[来源: {os.path.basename(doc.metadata.get('source', '未知'))}]\n{doc.page_content}"
                for doc in results
            ]
            full_context = "\n---\n".join(context_list)

            # 安全阀：限制返回的最大字符数，防止 Token 溢出导致 LLM 报错或变慢
            # 4000 字符大约对应 2000-3000 tokens，对大多数 LLM 都是安全的
            if len(full_context) > 4000:
                print(
                    f"⚠️ 检索内容过长 ({len(full_context)} 字符)，已截断至 4000 字符。"
                )
                return full_context[:4000] + "\n...(内容已截断)"

            return full_context
        except Exception as e:
            print(f"检索出错: {e}")
            return ""
