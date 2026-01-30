# AgentLearn: 从零构建商业级智能笔记助理

本项目演示了如何使用 DeepSeek LLM + LangGraph + RAG 构建一个具备“自主思考”能力的智能体。

## 🏗️ 架构概览

我们的 Agent 模仿了人类的思考-行动循环：

```mermaid
graph TD
    A[用户输入] --> B[DeepSeek (Brain)]
    B -->|思考| C{是否需要工具?}
    C -->|是| D[Tools (Hands)]
    D -->|RAG| E[本地笔记库]
    D -->|Web| F[互联网搜索]
    D -->|结果反馈| B
    C -->|否| G[生成回答]
```

### 核心组件

1.  **LLM (大脑)**: 使用 `DeepSeek-V3` (兼容 OpenAI 接口)。
    *   *作用*: 理解意图、逻辑推理、生成回复。
2.  **RAG (海马体/长期记忆)**: 基于 `LangChain` + `FAISS`。
    *   *作用*: 将你的私有笔记 (`data/*.md`) 向量化，解决 LLM 不知道你私人数据的问题。
3.  **Tools (手眼)**:
    *   `search_notes`: 查阅本地知识库。
    *   `web_search`: 模拟联网获取外部实时信息。
4.  **LangGraph (神经系统)**:
    *   *作用*: 编排 "思考 -> 工具 -> 观察 -> 再思考" 的多步推理循环。

## 📂 项目结构

```text
d:\code\AgentLearn\
├── data/               # 你的笔记存放处 (.md, .txt)
├── faiss_index/        # 自动生成的向量索引文件
├── .env                # 配置文件 (API Key)
├── main.py             # 程序入口 (Agent 主逻辑)
├── rag_engine.py       # RAG 核心模块
└── requirements.txt    # 依赖清单
```

## 🚀 快速开始

### 1. 环境准备
确保已安装 Python 3.10+。

```bash
pip install -r requirements.txt
```

### 2. 配置密钥
在 `.env` 文件中填入你的 DeepSeek API Key：
```properties
DEEPSEEK_API_KEY=sk-xxxx
DEEPSEEK_BASE_URL=https://api.deepseek.com
```

### 3. 准备数据
在 `data/` 目录下随便写几个 `.md` 文件作为你的笔记。
例如 `data/ideas.md`:
```markdown
# 我的创业点子
我想做一个 AI 驱动的咖啡机...
```

### 4. 运行
```bash
python main.py
```

## 🛠️ 实战指南

### 如何添加新笔记？
只需把 `.md` 文件丢进 `data/` 目录，然后在程序中输入 `update` 指令，或者重启程序，Agent 就会自动索引新内容。

### 如何让它更聪明？
目前的 `web_search` 是模拟返回固定数据的。如果你想让它真的联网：
1. 去注册 [SerpApi](https://serpapi.com/) 或 [Tavily](https://tavily.com/)。
2. 修改 `main.py` 中的 `web_search` 函数，调用真实的搜索 API。

## 📚 理论巩固

- **为什么不用单纯的 Prompt Engineering?**
    - Prompt 只能处理一次性任务，无法处理“先查A，根据A的结果再去查B”的复杂流程。LangGraph 解决了这个问题。
- **为什么需要 RAG?**
    - LLM 的训练数据截止到过去某个时间点，且不知道你的私有数据。RAG 是外挂知识库。
- **Agent vs Chatbot?**
    - Chatbot (如 ChatGPT 网页版) 主要靠对话。
    - Agent (智能体) = LLM + Memory + Planning + Tools。它能主动**做事**。
