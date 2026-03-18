import traceback
from typing import List
from dotenv import load_dotenv

# LangChain / LangGraph 组件
from langchain_core.messages import HumanMessage, SystemMessage

from config.settings import Settings
from core.observability.logger import log_event
from core.observability.metrics import runtime_metrics
from core.security.frequency_guard import SlidingWindowFrequencyGuard
from infra.llm.openai_compatible import OpenAICompatibleClient
from rag_engine import RAGEngine
from workflow.graph import build_app
from workflow.nodes.agent_node import create_agent_node

# ==========================================
# 1. 配置与初始化
# ==========================================
load_dotenv()

settings = Settings.from_env()
llm_client = OpenAICompatibleClient(settings)
restricted_query_limiter = SlidingWindowFrequencyGuard(
    limit=settings.restricted_query_limit_per_minute,
    window_seconds=60,
)

rag = RAGEngine()
agent_node = create_agent_node(settings, llm_client, rag, restricted_query_limiter)
app = build_app(agent_node)

# ==========================================
# 7. 主程序入口
# ==========================================


def main():
    def print_help() -> None:
        print("\n🧭 可用功能速览")
        print("- 直接提问：如“帮我总结我写过的 LangGraph 笔记”")
        print("- 列出本地文件：输入 `files` 或自然语言“列出笔记/本地有什么笔记”")
        print("- 图片问答禁用 OCR：在问题中加“不要ocr/不用ocr/别用ocr”")
        print("- 强制重建索引：输入 `update`")
        print("- 查看会话指标快照：输入 `metrics`")
        print("- 查看当前关键配置：输入 `config`")
        print("- 退出：`q` / `quit` / `exit`\n")

    def print_config() -> None:
        print("\n⚙️ 当前关键配置")
        print(f"- model: {settings.model_name}")
        print(f"- base_url: {settings.base_url}")
        print(f"- api_mode: {settings.api_mode}")
        print(f"- reranker_enabled: {settings.reranker_enabled}")
        print(f"- image_vlm_enabled: {rag.enable_image_vlm}")
        print(f"- image_ocr_enabled: {rag.enable_image_ocr}")
        print(f"- allowed_tools: {', '.join(settings.allowed_tools) if settings.allowed_tools else '(none)'}")
        print()

    def print_metrics() -> None:
        m = runtime_metrics.snapshot()
        print("\n📊 会话指标快照（进程内）")
        print(f"- requests_total: {m.get('requests_total', 0)}")
        print(f"- retrieval_hit_total: {m.get('retrieval_hit_total', 0)}")
        print(f"- retrieval_miss_total: {m.get('retrieval_miss_total', 0)}")
        print(f"- error_total: {m.get('error_total', 0)}")
        print(f"- avg_latency_ms: {m.get('avg_latency_ms', 0)}")
        print(f"- retrieval_hit_rate: {m.get('retrieval_hit_rate', 0)}")
        print(f"- error_rate: {m.get('error_rate', 0)}")
        print()

    print("==================================================")
    print("🤖 个人智能笔记助理 (OpenAI Responses + LangGraph + RAG)")
    print("==================================================")
    try:
        settings.validate_startup()
    except Exception as e:
        print(f"❌ 启动配置校验失败: {e}")
        return

    if not rag.load_index():
        rag.build_index()

    print(f"当前模型: {settings.model_name}")
    print(f"当前 Base URL: {settings.base_url}")
    print(f"当前 API 模式: {settings.api_mode} (默认 responses)")
    print("输入 'help' 查看功能速览")
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

            if user_input.lower() in ["help", "h", "?"]:
                print_help()
                continue

            if user_input.lower() == "config":
                print_config()
                continue

            if user_input.lower() == "metrics":
                print_metrics()
                continue

            if user_input.lower() == "files":
                files = rag.list_note_files()
                if files:
                    print("\n📁 当前可见本地笔记文件：")
                    for f in files:
                        print(f"- {f}")
                    print()
                else:
                    print("\n📁 当前 data/ 目录下未发现可用笔记文件（.md/.txt/.图片）。\n")
                continue

            if user_input.lower() == "update":
                rag.build_index()
                log_event("manual_reindex", "console", result="triggered")
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


