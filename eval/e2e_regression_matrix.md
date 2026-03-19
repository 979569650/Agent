# 智能笔记助手 E2E 回归矩阵（P1）

> 目标：覆盖检索、降级、update、no-OCR 四类核心场景，作为每次发布前最小回归清单。

| ID | 场景 | 输入示例 | 预期结果 | 覆盖项 |
|---|---|---|---|---|
| E2E-01 | 普通检索 | `帮我检索本地笔记里的智能笔记助手总结` | 命中检索并返回上下文化回答 | retrieval |
| E2E-02 | 小聊跳过检索 | `你好` | 不触发检索，直接对话回复 | retrieval gating |
| E2E-03 | no-OCR 模式 | `不要ocr，检索这张图片` | 仅走 VLM 语义路径，不注入 OCR 文本 | no-OCR |
| E2E-04 | update 重建 | 控制台输入 `update` | 成功重建索引并写入 hash | update |
| E2E-05 | 受限查询鉴权 | `帮我找账号密码` | 触发防护校验/频控/审计日志 | security |
| E2E-06 | 模型级 reranker 开关 | 开 `RERANKER_ENABLED=true` | 排序可二次重排；失败自动降级规则排序 | reranker |
| E2E-07 | 索引失败回滚 | 构建时注入异常（测试环境） | 自动回滚到快照，保留恢复标记 | rollback |

## 执行建议

1. 先跑质量门禁：`python scripts/quality_gate.py`
2. 启动主程序：`python main.py`
3. 按矩阵逐条手测并记录结论（Pass/Fail/备注）

