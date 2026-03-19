import contextlib
import io
import unittest
from unittest import mock

import main


class _FakeApp:
    def __init__(self, event):
        self._event = event
        self.last_state = None

    def stream(self, state, stream_mode="values"):
        self.last_state = state
        yield self._event


class TestMainCliStreaming(unittest.TestCase):
    def setUp(self):
        self._orig_quiet = main.FLAGS.quiet
        self._orig_verbose = main.FLAGS.verbose

    def tearDown(self):
        main.FLAGS.quiet = self._orig_quiet
        main.FLAGS.verbose = self._orig_verbose

    def test_run_agent_once_passes_stream_sink_into_state(self):
        main.FLAGS.quiet = False
        main.FLAGS.verbose = True

        final_msg = mock.Mock()
        final_msg.content = "ok"
        app = _FakeApp({"messages": [final_msg]})
        sink = lambda chunk: None

        out, history = main.run_agent_once(app, ["m1"], stream_sink=sink)

        self.assertEqual(out.content, "ok")
        self.assertEqual(history[-1].content, "ok")
        self.assertIn("stream_sink", app.last_state)
        self.assertIs(app.last_state["stream_sink"], sink)

    def test_run_chat_command_forwards_on_assistant_chunk(self):
        main.FLAGS.quiet = False
        main.FLAGS.verbose = True

        fake_answer = mock.Mock()
        fake_answer.content = "最终答案"
        fake_history = ["h1", "h2"]

        with mock.patch.object(main, "run_agent_once", return_value=(fake_answer, fake_history)) as run_mock:
            chunks = []
            result = main._run_chat_command(
                "你好",
                settings=mock.Mock(),
                rag=mock.Mock(),
                app=mock.Mock(),
                chat_history=[],
                on_assistant_chunk=chunks.append,
            )

        self.assertEqual(result["kind"], "assistant")
        self.assertEqual(result["answer"], "最终答案")
        kwargs = run_mock.call_args.kwargs
        self.assertIn("stream_sink", kwargs)
        self.assertTrue(callable(kwargs["stream_sink"]))

    def test_run_chat_once_json_does_not_stream_chunks(self):
        main.FLAGS.quiet = False
        main.FLAGS.verbose = True

        settings = mock.Mock()
        rag = mock.Mock()
        app = mock.Mock()

        final_msg = mock.Mock()
        final_msg.content = "完整答案"

        with mock.patch.object(main, "ensure_startup", return_value=True):
            with mock.patch.object(main, "run_agent_once", return_value=(final_msg, [])) as run_mock:
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    code = main.run_chat_once(settings, rag, app, "你好", output="json")

        self.assertEqual(code, main.EXIT_OK)
        out = buf.getvalue()
        self.assertIn('"answer": "完整答案"', out)
        kwargs = run_mock.call_args.kwargs
        self.assertNotIn("stream_sink", kwargs)


if __name__ == "__main__":
    unittest.main()
