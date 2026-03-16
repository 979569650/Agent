from typing import List
import time
import requests
from langchain_core.messages import BaseMessage

from config.settings import Settings


class OpenAICompatibleClient:
    def __init__(self, settings: Settings):
        self.settings = settings

    def _to_responses_input(self, messages: List[BaseMessage]):
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
            content_type = "output_text" if role == "assistant" else "input_text"
            items.append(
                {
                    "role": role,
                    "content": [{"type": content_type, "text": content}],
                }
            )
        return items

    def _to_chat_messages(self, messages: List[BaseMessage]):
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

    def _post(self, path: str, payload: dict) -> dict:
        if not self.settings.base_url or not self.settings.api_key:
            raise RuntimeError("缺少 BASE_URL 或 API_KEY 配置")

        url = f"{self.settings.base_url}{path}"
        headers = {
            "Authorization": f"Bearer {self.settings.api_key}",
            "Content-Type": "application/json",
        }
        max_attempts = self.settings.request_max_retries + 1
        backoff = self.settings.request_retry_backoff_ms / 1000.0
        last_error: Exception | None = None

        for attempt in range(1, max_attempts + 1):
            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=self.settings.request_timeout)
                status = resp.status_code

                # 429 / 5xx 作为可重试错误
                if status == 429 or status >= 500:
                    raise RuntimeError(f"HTTP {status} @ {path}: {resp.text[:400]}")

                if status >= 400:
                    # 4xx（除429）视为非重试错误，直接返回
                    raise ValueError(f"HTTP {status} @ {path}: {resp.text[:400]}")

                try:
                    return resp.json()
                except Exception:
                    raise ValueError(f"接口返回非 JSON: {resp.text[:400]}")
            except ValueError as e:
                # 非重试错误，立即失败
                raise RuntimeError(str(e))
            except (requests.Timeout, requests.ConnectionError, requests.RequestException, RuntimeError) as e:
                last_error = e
                if attempt >= max_attempts:
                    break
                time.sleep(backoff * attempt)

        raise RuntimeError(f"请求失败(重试{max_attempts}次后仍失败): {type(last_error).__name__}: {last_error}")

    def call_responses(self, messages: List[BaseMessage]) -> str:
        response = self._post(
            "/responses",
            {
                "model": self.settings.model_name,
                "input": self._to_responses_input(messages),
            },
        )

        text = response.get("output_text", "")
        if text:
            return text

        output = response.get("output", []) or []
        chunks = []
        for item in output:
            for c in (item.get("content", []) if isinstance(item, dict) else []):
                if c.get("type", "") in {"output_text", "text"}:
                    t = c.get("text", "")
                    if t:
                        chunks.append(t)
        return "\n".join(chunks).strip()

    def call_chat_completions(self, messages: List[BaseMessage]) -> str:
        response = self._post(
            "/chat/completions",
            {
                "model": self.settings.model_name,
                "messages": self._to_chat_messages(messages),
            },
        )
        choices = response.get("choices", [])
        if not choices:
            return ""
        msg = choices[0].get("message", {})
        return msg.get("content", "") or ""

    def call_model(self, messages: List[BaseMessage]) -> str:
        if self.settings.api_mode == "responses":
            try:
                return self.call_responses(messages)
            except Exception as e:
                raise RuntimeError(f"/responses 调用失败 | model={self.settings.model_name} | {type(e).__name__}: {e}")

        if self.settings.api_mode == "chat":
            try:
                return self.call_chat_completions(messages)
            except Exception as e:
                raise RuntimeError(f"/chat/completions 调用失败 | model={self.settings.model_name} | {type(e).__name__}: {e}")

        raise RuntimeError("API_MODE 仅支持 responses 或 chat")
