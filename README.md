# 🤖 AgentLearn: 个人智能笔记助理（OpenAI Responses + 多模态 RAG）

> 一个面向本地笔记管理的智能助理项目。  
> 项目聚焦：**OpenAI Responses 接口优先调用** + **本地多模态 RAG（文本+图片）可降级运行**。

## 📖 项目简介

本项目用于构建一个可在本地运行的笔记助理，核心能力包括：

- 使用 **OpenAI Responses API** 进行对话（支持中转站）
- 使用 **RAG** 管理并检索本地 Markdown / TXT / 图片 笔记
- 提供 **CLI + Web** 两种使用方式（`main.py` 与 `app_web.py`）
- 在 Embedding 模型网络不可用时，自动降级，避免程序启动失败

## 🏗️ 技术架构

技术栈：

* **模型调用**: OpenAI Responses API（默认）+ Chat Completions（可切换）
* **编排**: [LangGraph](https://langchain-ai.github.io/langgraph/)
* **框架**: [LangChain](https://www.langchain.com/)
* **RAG**: [FAISS](https://github.com/facebookresearch/faiss) + HuggingFace Embeddings（文本）+ CLIP（图片）
* **文件增量检测**: 基于 MD5 哈希
* **可观测**: 结构化日志 + OpenTelemetry（可选）+ 本地指标看板脚本 + Web 系统状态/后台诊断页
* **防护**: 受限查询校验、频控、防注入、工具白名单、审计事件

### 调用流程

```mermaid
graph LR
    Start([用户输入]) --> Agent[🤖 LangGraph 节点]
    Agent --> API[OpenAI-Compatible API]
    API --> End([回复用户])
```

## ✨ 核心特性

1. **🔌 中转 API 兼容增强**
   - 默认走 `/responses`
   - 可切换 `/chat/completions`
   - 失败时输出清晰错误提示，便于定位网关策略/权限问题
   - 内置网络重试与退避（针对连接抖动、429、5xx）

2. **🧠 本地多模态 RAG 索引**
   - 递归扫描 `data/` 下 `.md/.txt/.png/.jpg/.jpeg/.webp/.bmp`
   - 自动增量检测，按需重建索引
   - 文本与图片双索引融合检索（文本 query 可召回相关图片）
   - 默认自动融合：**图片语义理解（VLM）+ OCR + CLIP 召回**
   - 可选“视觉大模型理解”：直接让多模态 LLM 生成图片语义摘要入库（不仅是 OCR）
   - 支持图片 OCR：图片文字会转为“可切分、可向量化”的文本资源
   - 支持对话级“禁用 OCR”：用户输入“不要ocr/不用ocr/别用ocr”时，切换到 VLM-only 回答
   - 新增轻量检索质量链路：**query rewrite + 意图过滤 + 轻量 reranker**
   - 支持**模型级 reranker（可开关）**，不可用时自动降级到规则 reranker
   - 索引构建支持**快照回滚与崩溃恢复标记**，降低中断/异常导致的损坏风险

3. **🛡️ 启动容错降级**
   - HuggingFace 模型连通失败时，不阻塞主程序启动

4. **📊 可观测增强（最小闭环）**
   - 每轮请求生成 `trace_id`
   - 结构化 JSON 日志：`request_received / retrieval_done / response_generated`
   - 内置基础指标快照：请求数、检索命中/未命中、错误率、平均延迟
   - 可选 OpenTelemetry span（`OTEL_ENABLED=true`）
   - 支持日志落盘为 JSONL，并通过脚本查看延迟/命中率/错误率

5. **🔐 受限查询防护键校验（可选）**
   - 账号/密码类查询可要求输入访问口令（支持 bcrypt，兼容 SHA256）
   - 内置受限查询频控（默认每分钟 3 次）
   - 输入防注入规则（提示词劫持/越权探测）
   - 工具白名单策略（限制可执行工具）
   - 防护审计事件日志（便于事后排查）

## 🚀 快速开始

### 1. 环境准备
*   Windows / macOS / Linux
*   Python 3.10+
* OpenAI 兼容 API Key（可来自官方或中转站）

### 2. 安装依赖
```bash
# 推荐创建虚拟环境
python -m venv .venv
# Windows 激活
.venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt
```

### 3. 配置口令
复制 `.env.example` 为 `.env`，按你的中转站配置填写：
```ini
API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxx
BASE_URL=
MODEL=gpt-5.3-codex
API_MODE=responses
REQUEST_TIMEOUT=45
SECURITY_KEY_HASH=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

> 注意：仓库中的 `.env` 属于本地受限信息文件，请勿提交到 git。当前项目已在 `.gitignore` 中忽略它。

### 4. 准备数据
在 `data/` 目录下放入你的 Markdown (`.md`) / Text (`.txt`) / 图片 (`.png/.jpg/.jpeg/.webp/.bmp`) 文件。
*   支持子目录递归扫描。
*   系统会自动处理编码（UTF-8/GBK）。

### 5. 运行助理
```bash
python main.py
```

### 6. 运行 Web UI

```bash
python -m uvicorn app_web:app --host 127.0.0.1 --port 8000 --reload
```

启动后访问：`http://127.0.0.1:8000`

可用页面与接口：

- `/api/chat`
- `/api/notes*`（列表/读取/新增/修改/删除/重命名/上传）
- `/api/index/*`（重建与状态）
- `/api/config*`（读取/更新/测试/重置）
- `/api/system/status`
- `/api/health`

## ✅ 使用指南（建议按这个流程）

### A. 第一次使用（初始化一次）

1. **创建虚拟环境 & 安装依赖**
   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **配置环境变量（OpenAI 兼容 API）**
   把 `.env.example` 复制为 `.env` 并填写：
   ```ini
   API_KEY=sk-你的key
   BASE_URL=https://你的中转站域名/v1
   MODEL=gpt-5.3-codex
   API_MODE=responses
   # SECURITY_KEY_HASH=（可选，见下方“访问口令”）
   ```

3. **准备笔记数据**
   - 在项目根目录创建 `data/` 文件夹（没有就新建）。
   - 将你的 `.md` / `.txt` 笔记放入其中（支持子目录）。

4. **启动**
   ```bash
   python main.py
   ```
   首次启动会下载中文 Embedding 模型并建立索引，耗时取决于网络与机器性能。

### B. 日常使用（像聊天一样提问）

启动后出现 `👉 你:`，直接输入问题即可。例如：

- “帮我找一下我之前写的 LangGraph 总结”
- “我记录过 XXX 项目的 TODO 吗？”
- “把我关于 XX 的笔记要点汇总一下”

默认按“对话优先”工作流返回答案；RAG 检索能力可按需接入工具调用链。

#### 图片问答建议写法

- 默认融合模式（语义理解 + OCR）：
  - `看一下我的身份证背面`
- 强制不使用 OCR（仅图片语义理解）：
  - `不要ocr，看一下这个idcard的内容`

> 说明：若你要求“不要 OCR”，但当前网关未返回图片语义理解结果，系统会明确提示未命中 VLM 文本，而不是偷偷回退 OCR 文本。

### C. 常用指令

- 功能导航：`help` / `h` / `?`
- 退出：`q` / `quit` / `exit`
- 强制重建索引：`update`
- 列出当前可见本地笔记文件：`files`
- 查看当前关键配置：`config`
- 查看当前会话指标快照：`metrics`

> 提示：如果你是新用户，建议启动后先输入 `help`，再输入 `files` 快速确认本地数据是否已被识别。

### D. 常见提示与排错

- 如果看到：`⚠️ 未找到可索引文本` 或 `⚠️ 未找到可索引图片`：说明 `data/` 中缺少对应类型文件。
  放入文件后输入 `update` 或重启程序。
- 若看到 `/responses 调用失败`：通常是中转站风控、模型权限或接口策略限制。
  - 可临时改 `.env`：`API_MODE=chat` 再测。
- 若出现 `ConnectionResetError(10054)` / `远程主机强迫关闭连接`：通常是网关连接被重置。
  - 先重试同一问题（网络抖动常见）
  - 将 `.env` 中 `API_MODE=chat` 后重启再测
  - 检查 `BASE_URL`、`MODEL`、Key 权限是否匹配当前网关策略
  - 可调大 `REQUEST_MAX_RETRIES` 与 `REQUEST_RETRY_BACKOFF_MS`（例如 3 / 800）
  - 必要时提高 `REQUEST_TIMEOUT`（如 60）并执行一次 `update` 重建索引
- 第一次运行慢：通常是 embedding 下载 + 索引构建导致，属于正常现象。
- 若多实例同时触发 `update`：当前版本已加入基础文件锁与 JSON 原子写，避免索引脏写。

## 🔐 访问口令（可选）

当你询问“账号/密码/口令”等受限信息时，程序会要求输入访问口令；它会将你输入的口令做 **SHA256** 后与 `SECURITY_KEY_HASH` 比对。

你可以用下面命令生成哈希：

```bash
python -c "import hashlib;print(hashlib.sha256('your-key'.encode()).hexdigest())"
```

## ⚙️ 关键环境变量

- `API_KEY`: OpenAI 兼容 API Key
- `BASE_URL`: OpenAI 兼容服务地址（一般以 `/v1` 结尾）
- `MODEL`: 单模型名（例如 `gpt-5.3-codex`）
- `API_MODE`: `responses` 或 `chat`（默认 `responses`）
- `REQUEST_TIMEOUT`: HTTP 超时时间（秒）
- `REQUEST_MAX_RETRIES`: 网络重试次数（默认 `2`）
- `REQUEST_RETRY_BACKOFF_MS`: 重试退避基准毫秒（默认 `500`）
- `OTEL_ENABLED`: 是否启用 OpenTelemetry（默认 `false`，启用后输出 span）
- `LOG_FILE`: 结构化日志落盘路径（默认 `observability/events.jsonl`）
- `ALLOWED_TOOLS`: 工具白名单（逗号分隔，默认 `search_notes,list_note_files`）
- `RERANKER_ENABLED`: 是否启用模型级 reranker（默认 `false`）
- `RERANKER_MODEL`: 模型级 reranker 名称（默认 `BAAI/bge-reranker-base`）
- `IMAGE_EMBEDDING_MODEL`: 图片向量模型（默认 `clip-ViT-B-32`）
- `VISION_MODEL`: 图片理解用多模态模型（默认跟随 `MODEL`）
- `ENABLE_IMAGE_VLM`: 是否启用“视觉大模型理解入库”（默认 `true`）
- `ENABLE_IMAGE_OCR`: 是否启用图片 OCR 入库（默认 `true`）
- `SECURITY_KEY_HASH`: 受限查询防护键哈希（可选）
- `RESTRICTED_QUERY_LIMIT_PER_MINUTE`: 受限查询频控阈值（默认 `3`）

> 备注：`ENABLE_IMAGE_OCR=false` 可全局关闭 OCR；若只想单轮禁用 OCR，可在提问中加入“不要ocr”。

## 🧪 最小测试与评测

- 单元测试：
  ```bash
  python -m unittest tests/test_policies.py tests/test_security.py
  ```
- 检索触发策略评测：
  ```bash
  python eval/run_eval.py
  ```

- 生成评测报告 + 回归对比：
  ```bash
  python eval/run_eval.py --report eval/reports/current.json
  python eval/compare_reports.py --baseline eval/reports/baseline.json --current eval/reports/current.json --out eval/reports/compare.md
  ```

- 查看本地指标看板（延迟/命中率/错误率）：
  ```bash
  python scripts/show_metrics_dashboard.py
  ```

- 查看 E2E 回归矩阵：
  - `eval/e2e_regression_matrix.md`

## 📂 项目结构

```
AgentLearn/
├── data/                 # 存放你的个人笔记 (Markdown/Txt/Images)
├── faiss_index/          # 自动生成的向量索引数据库
├── .env                  # 环境变量 (API Key)
├── config/settings.py    # ⚙️ 配置中心（含启动校验）
├── core/domain/policies.py
├── core/domain/retrieval_planner.py   # 🧭 检索规划（rewrite/filter/rerank）
├── core/security/         # 🔐 鉴权、频控、注入防护、审计、工具白名单
├── core/observability/   # 📊 日志与基础指标
├── infra/llm/            # 🔌 OpenAI-compatible 客户端封装
├── infra/retrieval/      # 🔎 检索流水线、模型 reranker、存储恢复工具
├── workflow/             # 🕸️ LangGraph state/graph/nodes 编排层
├── main.py               # 🚀 主程序入口
├── rag_engine.py         # 🧠 RAG 引擎（向量化、检索、增量更新、快照回滚与恢复）
├── scripts/quality_gate.py           # ✅ 一键质量门禁
├── scripts/show_metrics_dashboard.py # 📈 本地指标看板
├── eval/run_eval.py                  # 🧪 评测执行
├── eval/compare_reports.py           # 📊 评测回归对比
├── eval/e2e_regression_matrix.md     # 🧾 E2E 回归矩阵
├── requirements.txt      # 依赖列表
└── README.md             # 说明文档
```

## 🗺️ 路线图 (Roadmap)

- [x] **基础 RAG**: 实现本地文档问答
- [x] **OpenAI-Compatible 兼容**: 支持 `/responses` 与 `/chat/completions`
- [x] **中转站稳定性改造**: 原生 HTTP 调用 + 明确报错
- [x] **RAG 启动降级**: embedding 不可用时主流程不崩
- [ ] **RAG 与工具链深度整合**: 将本地检索接入更完整 Agent 工具流程
- [ ] **多格式支持**: 支持 PDF, Word, Excel 文档
- [ ] **Web 界面**: 开发 Streamlit/Gradio 网页版 UI
- [ ] **长期记忆**: 引入 SQLite 存储历史对话摘要
