import json
import unittest
from unittest import mock

from langchain_core.messages import HumanMessage

from config.settings import Settings
from infra.llm.openai_compatible import OpenAICompatibleClient


class _FakeStreamResponse:
    def __init__(self, lines, status_code=200):
        self._lines = lines
        self.status_code = status_code
        self.text = ""
        self.closed = False

    def iter_lines(self, decode_unicode=True):
        for line in self._lines:
            yield line

    def close(self):
        self.closed = True


class TestOpenAICompatibleStream(unittest.TestCase):
    def _build_settings(self, api_mode: str) -> Settings:
        return Settings(
            model_name="test-model",
            api_mode=api_mode,
            request_timeout=5,
            request_max_retries=0,
            request_retry_backoff_ms=100,
            base_url="https://example.com/v1",
            api_key="k",
        )

    def test_chat_stream_parses_delta_and_concatenates(self):
        client = OpenAICompatibleClient(self._build_settings("chat"))
        stream_resp = _FakeStreamResponse(
            [
                'data: {"choices":[{"delta":{"content":"你"}}]}',
                'data: {"choices":[{"delta":{"content":"好"}}]}',
                "data: [DONE]",
            ]
        )

        with mock.patch("infra.llm.openai_compatible.requests.post", return_value=stream_resp) as post_mock:
            chunks = []
            out = client.call_model([HumanMessage(content="hi")], on_delta=chunks.append)

        self.assertEqual(out, "你好")
        self.assertEqual(chunks, ["你", "好"])
        self.assertTrue(stream_resp.closed)

        payload = post_mock.call_args.kwargs["json"]
        self.assertTrue(payload.get("stream"))

    def test_responses_stream_parses_output_text_delta(self):
        client = OpenAICompatibleClient(self._build_settings("responses"))
        stream_resp = _FakeStreamResponse(
            [
                'data: {"type":"response.output_text.delta","delta":"今"}',
                'data: {"type":"response.output_text.delta","delta":"天"}',
                "data: [DONE]",
            ]
        )

        with mock.patch("infra.llm.openai_compatible.requests.post", return_value=stream_resp) as post_mock:
            chunks = []
            out = client.call_model([HumanMessage(content="hi")], on_delta=chunks.append)

        self.assertEqual(out, "今天")
        self.assertEqual(chunks, ["今", "天"])
        self.assertTrue(stream_resp.closed)

        payload = post_mock.call_args.kwargs["json"]
        self.assertTrue(payload.get("stream"))

    def test_chat_stream_decodes_utf8_bytes(self):
        client = OpenAICompatibleClient(self._build_settings("chat"))
        stream_resp = _FakeStreamResponse(
            [
                b'data: {"choices":[{"delta":{"content":"\xe4\xbd\xa0"}}]}',
                b'data: {"choices":[{"delta":{"content":"\xe5\xa5\xbd"}}]}',
                b"data: [DONE]",
            ]
        )

        with mock.patch("infra.llm.openai_compatible.requests.post", return_value=stream_resp):
            chunks = []
            out = client.call_model([HumanMessage(content="hi")], on_delta=chunks.append)

        self.assertEqual(out, "你好")
        self.assertEqual(chunks, ["你", "好"])

    def test_stream_failure_falls_back_to_non_stream(self):
        client = OpenAICompatibleClient(self._build_settings("chat"))

        with mock.patch.object(client, "call_chat_completions_stream", side_effect=RuntimeError("stream failed")):
            with mock.patch.object(client, "call_chat_completions", return_value="fallback") as fallback_mock:
                out = client.call_model([HumanMessage(content="hi")], on_delta=lambda _: None)

        self.assertEqual(out, "fallback")
        fallback_mock.assert_called_once()


if __name__ == "__main__":
    unittest.main()
