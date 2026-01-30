import os
import operator
import hashlib
from typing import Annotated, TypedDict, List
from dotenv import load_dotenv

# LangChain / LangGraph 组件
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    ToolMessage,
    SystemMessage,
)
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

# 引入我们封装好的 RAG 引擎
from rag_engine import RAGEngine

# ==========================================
# 1. 配置与初始化
# ==========================================
load_dotenv()

# 初始化 LLM (DeepSeek)
llm = ChatOpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    model="deepseek-chat",
    temperature=0,  # 保持冷静，利于工具调用
    streaming=False,
)

# 初始化 RAG 引擎
rag = RAGEngine()
# 启动时尝试加载索引，如果不存在则自动构建
if not rag.load_index():
    rag.build_index()

# ==========================================
# 2. 定义工具 (Tools) - Agent 的"手"
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


@tool
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


@tool
def web_search(query: str) -> str:
    """
    【查网络】搜索互联网。
    当本地笔记中没有相关信息，或者用户明确询问实时信息、外部通用知识（如技术定义、新闻）时使用。
    """
    print(f"\n🌐 [Tool] 正在搜索网络: {query}")
    # 这里是模拟数据，实际项目中可接入 Google Search API (如 SerpApi)
    return f"""【模拟搜索结果】关于 '{query}' 的信息：
    1. LangGraph 是 LangChain 官方推出的 Agent 编排框架。
    2. DeepSeek 是中国领先的开源大模型提供商。
    3. RAG (Retrieval-Augmented Generation) 是目前解决 LLM 幻觉的主流方案。
    (注：这是一个模拟结果，用于演示 Agent 的联网意图)"""


# 工具列表
tools = [search_notes, web_search]
llm_with_tools = llm.bind_tools(tools)

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
    """思考节点：LLM 观察历史消息，决定下一步行动"""
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


# 工具执行节点 (直接使用 LangGraph 预置实现)
tool_node = ToolNode(tools)

# ==========================================
# 5. 定义边 (Edges) - 流程控制
# ==========================================


def should_continue(state: AgentState):
    """条件边：决定是继续调用工具，还是结束并回复用户"""
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return END


# ==========================================
# 6. 构建与编译图
# ==========================================

workflow = StateGraph(AgentState)

# 添加节点
workflow.add_node("agent", agent_node)
workflow.add_node("tools", tool_node)

# 设置入口
workflow.set_entry_point("agent")

# 添加边
workflow.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
workflow.add_edge("tools", "agent")  # 工具用完后，必须回 agent 思考

# 编译应用
app = workflow.compile()

# ==========================================
# 7. 主程序入口
# ==========================================


def main():
    print("==================================================")
    print("🤖 个人智能笔记助理 (DeepSeek + LangGraph + RAG)")
    print("==================================================")
    print("输入 'q' 或 'quit' 退出")
    print("输入 'update' 可强制重建笔记索引")

    # 初始化系统提示词
    sys_msg = SystemMessage(
        content="""
    你是一个智能个人助理。
    1. 你的首要任务是帮助用户管理和检索【本地笔记】。
    2. 遇到问题时，先尝试调用 search_notes 工具。
    3. 如果本地笔记没有，再尝试调用 web_search 工具获取外部信息。
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

                # 打印中间过程 (Debug用)
                if isinstance(last_msg, AIMessage) and last_msg.tool_calls:
                    print(f"\n🤖 [Plan] 决定调用工具: {last_msg.tool_calls[0]['name']}")
                elif isinstance(last_msg, ToolMessage):
                    print(f"🛠️ [Exec] 工具执行完毕")

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
            print(f"\n❌ 发生错误: {e}")


if __name__ == "__main__":
    main()
