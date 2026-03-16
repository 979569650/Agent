import os
import operator
import hashlib
import traceback
from typing import Annotated, TypedDict, List
import requests
from dotenv import load_dotenv

# LangChain / LangGraph 组件
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
)
from langgraph.graph import StateGraph, END

# 引入我们封装好的 RAG 引擎
from rag_engine import RAGEngine

# ==========================================
# 1. 配置与初始化
# ==========================================
load_dotenv()

# 初始化 OpenAI Responses 客户端（兼容中转站 OpenAI API）
MODEL_NAME = os.getenv("MODEL", "deepseek-chat")
API_MODE = os.getenv("API_MODE", "responses").lower()  # responses / chat
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "45"))

# 初始化 RAG 引擎
rag = RAGEngine()
# 启动时尝试加载索引，如果不存在则自动构建
if not rag.load_index():
    rag.build_index()

# ==========================================
# 2. Responses API 辅助函数
# ==========================================

def verify_security_key(user_input: str) -> bool:
    """
    验证用户输入的安全密钥
    使用SHA256哈希比对，避免明文存储和传输
    """
    expected_hash = os.getenv("SECURITY_KEY_HASH")
    if not expected_hash:
        print("⚠️ 安全密钥哈希未配置")
        return False
    
    user_hash = hashlib.sha256(user_input.encode()).hexdigest()
    return user_hash == expected_hash


def search_notes(query: str) -> str:
    """
    【查笔记】搜索用户的本地笔记知识库。
    当用户问及个人记录、之前的想法、项目构思、已有知识或**账号密码**等隐私信息时，必须优先使用此工具。
    """
    print(f"\n🔍 [Tool] 正在检索本地笔记: {query}")
    
    # 检测密码相关查询的关键词
    password_keywords = ["密码", "password", "密钥", "key", "账号", "account", "登录", "login"]
    query_lower = query.lower()
    is_password_query = any(keyword in query_lower for keyword in password_keywords)
    
    # 如果是密码相关查询，需要安全验证
    if is_password_query:
        print("⚠️ 检测到密码相关查询，需要安全验证")
        print("请输入安全密钥（输入'cancel'取消查询）：")
        
        try:
            user_key = input("安全密钥: ").strip()
            
            # 允许用户取消查询
            if user_key.lower() == "cancel":
                return "查询已取消。"
            
            # 验证安全密钥
            if not verify_security_key(user_key):
                return "❌ 没有获取密码的权限。安全密钥验证失败。"
            
            print("✅ 安全密钥验证通过")
        except KeyboardInterrupt:
            return "查询被用户中断。"
        except Exception as e:
            return f"❌ 安全验证过程中发生错误: {str(e)}"
    
    # 执行实际搜索
    context = rag.search(query)
    if not context:
        return "本地笔记中没有找到相关内容。"
    return f"找到以下相关笔记片段：\n{context}"


def _to_responses_input(messages: List[BaseMessage]):
    """将 LangChain message 历史转换为 OpenAI Responses API 输入格式。"""
    role_map = {
        "system": "system",
        "human": "user",
        "ai": "assistant",
    }

    items = []
    for m in messages:
        msg_type = getattr(m, "type", "")
        role = role_map.get(msg_type)
        if not role:
            continue

        content = m.content if isinstance(m.content, str) else str(m.content)
        # Responses API 中 assistant 历史消息应使用 output_text；
        # user/system 输入消息使用 input_text。
        content_type = "output_text" if role == "assistant" else "input_text"
        items.append(
            {
                "role": role,
                "content": [{"type": content_type, "text": content}],
            }
        )
    return items


def _to_chat_messages(messages: List[BaseMessage]):
    """将 LangChain message 历史转换为 Chat Completions 格式。"""
    role_map = {
        "system": "system",
        "human": "user",
        "ai": "assistant",
    }

    items = []
    for m in messages:
        msg_type = getattr(m, "type", "")
        role = role_map.get(msg_type)
        if not role:
            continue

        content = m.content if isinstance(m.content, str) else str(m.content)
        items.append({"role": role, "content": content})
    return items


def _post_openai_compatible(path: str, payload: dict) -> dict:
    """使用最小兼容请求格式调用中转站 OpenAI 兼容接口。"""
    base_url = (os.getenv("BASE_URL") or "").rstrip("/")
    api_key = os.getenv("API_KEY")
    if not base_url or not api_key:
        raise RuntimeError("缺少 BASE_URL 或 API_KEY 配置")

    url = f"{base_url}{path}"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    resp = requests.post(url, headers=headers, json=payload, timeout=REQUEST_TIMEOUT)
    if resp.status_code >= 400:
        raise RuntimeError(
            f"HTTP {resp.status_code} @ {path}: {resp.text[:400]}"
        )

    try:
        return resp.json()
    except Exception:
        raise RuntimeError(f"接口返回非 JSON: {resp.text[:400]}")


def call_responses_api(messages: List[BaseMessage], model_name: str) -> str:
    """调用 OpenAI Responses API，返回文本。"""
    response = _post_openai_compatible(
        "/responses",
        {
            "model": model_name,
            "input": _to_responses_input(messages),
        },
    )

    text = response.get("output_text", "")
    if text:
        return text

    # 兜底：当 output_text 为空时尝试从 output 结构提取
    output = response.get("output", []) or []
    chunks = []
    for item in output:
        for c in (item.get("content", []) if isinstance(item, dict) else []):
            if c.get("type", "") in {"output_text", "text"}:
                t = c.get("text", "")
                if t:
                    chunks.append(t)
    return "\n".join(chunks).strip()


def call_chat_completions_api(messages: List[BaseMessage], model_name: str) -> str:
    """调用 OpenAI Chat Completions API，返回文本。"""
    response = _post_openai_compatible(
        "/chat/completions",
        {
            "model": model_name,
            "messages": _to_chat_messages(messages),
        },
    )

    choices = response.get("choices", [])
    if not choices:
        return ""
    msg = choices[0].get("message", {})
    return msg.get("content", "") or ""


def call_model_api(messages: List[BaseMessage]) -> str:
    """
    中转站兼容层：
    - API_MODE=responses: 仅走 /responses
    - API_MODE=chat: 仅走 /chat/completions
    默认 responses，失败时明确提示。
    """
    print(f"\n🔁 [API] 使用模型: {MODEL_NAME}")

    if API_MODE == "responses":
        try:
            return call_responses_api(messages, MODEL_NAME)
        except Exception as e:
            raise RuntimeError(
                f"/responses 调用失败 | model={MODEL_NAME} | {type(e).__name__}: {e}"
            )

    if API_MODE == "chat":
        try:
            return call_chat_completions_api(messages, MODEL_NAME)
        except Exception as e:
            raise RuntimeError(
                f"/chat/completions 调用失败 | model={MODEL_NAME} | {type(e).__name__}: {e}"
            )

    raise RuntimeError("API_MODE 仅支持 responses 或 chat")

# ==========================================
# 3. 定义图状态 (Graph State) - Agent 的“脑回路”
# ==========================================


class AgentState(TypedDict):
    # 消息历史：使用 operator.add 实现追加模式
    messages: Annotated[List[BaseMessage], operator.add]


# ==========================================
# 4. 定义节点 (Nodes)
# ==========================================


def agent_node(state: AgentState):
    """思考节点：先做本地检索增强，再调用模型生成回复"""
    messages = list(state["messages"])

    # 取最后一条用户输入作为检索查询
    last_user_query = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            last_user_query = msg.content if isinstance(msg.content, str) else str(msg.content)
            break

    # 用户可强制“不要 OCR，只看图片语义理解”
    no_ocr_markers = ["不要ocr", "不用ocr", "别用ocr", "不使用ocr", "不要 ocr", "不用 ocr"]
    force_vlm_only = bool(last_user_query) and any(m in last_user_query.lower() for m in no_ocr_markers)

    # 对“本地有什么笔记/列出笔记”这类问题，直接返回本地文件列表，避免模型幻觉
    list_keywords = ["本地有什么笔记", "有哪些笔记", "列出笔记", "笔记列表", "本地笔记"]
    if last_user_query and any(k in last_user_query for k in list_keywords):
        files = rag.list_note_files()
        if files:
            file_lines = "\n".join([f"- {f}" for f in files])
            return {
                "messages": [
                    AIMessage(
                        content=(
                            "我当前可访问到以下本地笔记文件：\n"
                            f"{file_lines}\n\n"
                            "你可以继续说：‘找一下某篇的重点’或‘总结这篇内容’。"
                        )
                    )
                ]
            }
        return {"messages": [AIMessage(content="当前 data/ 目录下未发现可用笔记文件（.md/.txt）。")]}

    # 将本地检索结果注入上下文，避免模型“看不到本地笔记”
    if last_user_query:
        local_context = rag.search(last_user_query, vlm_only=force_vlm_only)
        print(f"\n🧠 [RAG] 本轮检索长度: {len(local_context)} 字符")
        if local_context:
            # 用“系统约束 + 用户补充上下文”双重注入，提升模型遵循度
            policy = (
                "你必须优先基于【本地检索结果】回答，"
                "不要说自己无法访问本地笔记；若检索结果中有来源，请明确引用。"
            )
            if force_vlm_only:
                policy += " 用户明确要求不使用OCR，请仅依据【图片语义理解】与图片向量信息回答，不要引用OCR字段。"
            messages.append(
                SystemMessage(
                    content=policy
                )
            )
            messages.append(
                HumanMessage(
                    content=(
                        "【本地检索结果】\n"
                        f"{local_context}\n\n"
                        f"【用户问题】{last_user_query}\n"
                        "请根据以上检索结果直接回答。"
                    )
                )
            )
        else:
            messages.append(
                SystemMessage(
                    content=(
                        "本轮本地笔记检索未命中，请明确告知用户未找到相关笔记，"
                        "不要假装已读取到本地内容。"
                    )
                )
            )

    reply = call_model_api(messages)
    return {"messages": [AIMessage(content=reply)]}

# ==========================================
# 5. 定义边 (Edges) - 流程控制
# ==========================================


def should_continue(state: AgentState):
    """Responses API 版本：单次生成后直接结束"""
    return END


# ==========================================
# 6. 构建与编译图
# ==========================================

workflow = StateGraph(AgentState)

# 添加节点
workflow.add_node("agent", agent_node)

# 设置入口
workflow.set_entry_point("agent")

# 添加边
workflow.add_conditional_edges("agent", should_continue, {END: END})

# 编译应用
app = workflow.compile()

# ==========================================
# 7. 主程序入口
# ==========================================


def main():
    print("==================================================")
    print("🤖 个人智能笔记助理 (OpenAI Responses + LangGraph + RAG)")
    print("==================================================")
    print(f"当前模型: {os.getenv('MODEL', 'deepseek-chat')}")
    print(f"当前 Base URL: {os.getenv('BASE_URL')}")
    print(f"当前 API 模式: {API_MODE} (默认 responses)")
    print("输入 'q' 或 'quit' 退出")
    print("输入 'update' 可强制重建笔记索引")

    # 初始化系统提示词
    sys_msg = SystemMessage(
        content="""
    你是一个智能个人助理。
    1. 你的首要任务是帮助用户管理和检索【本地笔记】。
    2. 你会收到系统注入的本地检索上下文，请优先基于该上下文回答。
    3. 若本地检索无结果，请直接说明“未找到相关本地笔记”。
    4. 回答要简洁明了，引用笔记内容时请说明。
    """
    )

    # 会话历史 (在内存中保持，重启丢失)
    chat_history = [sys_msg]

    while True:
        try:
            user_input = input("\n👉 你: ").strip()
            if not user_input:
                continue

            if user_input.lower() in ["q", "quit", "exit"]:
                print("👋 再见！")
                break

            if user_input.lower() == "update":
                rag.build_index()
                continue

            # 构造本次输入
            chat_history.append(HumanMessage(content=user_input))

            # 运行图
            # stream_mode="values" 会返回每一步的状态更新
            print("⏳ 思考中...", end="", flush=True)

            final_response = None
            for event in app.stream({"messages": chat_history}, stream_mode="values"):
                last_msg = event["messages"][-1]

                final_response = last_msg

            # 输出最终回答
            print(f"\n\n🤖 Agent: {final_response.content}")

            # 更新历史 (LangGraph 每次运行是无状态的，需要手动维护外层历史，或者把 state 传回去)
            # 这里为了简单，我们直接把 LangGraph 产生的新消息追加到 chat_history
            # 注意：app.stream 返回的是完整状态，我们只需要取新增的部分
            # 但最简单的做法是：把 final_response 加进去，如果是多轮对话，其实应该把中间的 tool_messages 也加进去
            # 为了严谨，我们直接用最后的状态更新 chat_history
            chat_history = event["messages"]

        except KeyboardInterrupt:
            print("\n👋 用户终止")
            break
        except Exception as e:
            print(f"\n❌ 发生错误: {type(e).__name__}: {e}")
            print("\n[Debug] 完整异常堆栈如下：")
            traceback.print_exc()
            print("\n[Hint] 如果你使用的是中转站/OpenAI 兼容网关，请重点检查：")
            print("1. BASE_URL 是否是网关要求的完整 OpenAI 兼容地址")
            print("2. MODEL 是否是该网关实际支持的模型名")
            print("3. 当前默认走 /responses；若网关不支持，可临时设置 API_MODE=chat")
            print("4. 该网关是否允许当前 Key 访问所选模型（避免 blocked）")


if __name__ == "__main__":
    main()
