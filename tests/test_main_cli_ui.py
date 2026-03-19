import contextlib
import io
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import main


class TestMainCliUi(unittest.TestCase):
    def setUp(self):
        self._orig_quiet = main.FLAGS.quiet

    def tearDown(self):
        main.FLAGS.quiet = self._orig_quiet

    def test_build_status_lines_returns_empty_when_quiet(self):
        main.FLAGS.quiet = True
        settings = SimpleNamespace(model_name="test-model", api_mode="chat", base_url="https://example.com/v1")

        lines = main._build_status_lines(settings, no_banner=False)

        self.assertEqual(lines, [])

    def test_build_status_lines_returns_empty_when_no_banner(self):
        main.FLAGS.quiet = False
        settings = SimpleNamespace(model_name="test-model", api_mode="chat", base_url="https://example.com/v1")

        lines = main._build_status_lines(settings, no_banner=True)

        self.assertEqual(lines, [])

    def test_build_status_lines_contains_title_and_key_fields(self):
        main.FLAGS.quiet = False
        settings = SimpleNamespace(model_name="test-model", api_mode="chat", base_url="https://example.com/v1")

        lines = main._build_status_lines(settings, no_banner=False)
        text = "\n".join(lines)

        self.assertIn("智能笔记助手 CLI", text)
        self.assertIn("智能笔记助手", text)
        self.assertIn("model test-model", text)
        self.assertIn("mode chat", text)
        self.assertIn("endpoint", text)

    def test_shorten_middle_keeps_both_ends(self):
        src = "https://very.long.example.com/path/to/resource/with/query/and/more/segments"
        max_len = 36

        shortened = main._shorten_middle(src, max_len=max_len)

        self.assertLessEqual(len(shortened), max_len)
        self.assertIn("...", shortened)

        remain = max_len - len("...")
        left = remain // 2
        right = remain - left
        self.assertTrue(shortened.startswith(src[:left]))
        self.assertTrue(shortened.endswith(src[-right:]))

    def test_doctor_json_output_has_no_banner_text(self):
        main.FLAGS.quiet = False

        stdout_buf = io.StringIO()
        with contextlib.redirect_stdout(stdout_buf):
            code = main.run_doctor(None, None, output="json", runtime_err=RuntimeError("boom"))

        self.assertEqual(code, main.EXIT_GENERIC_ERROR)
        out_text = stdout_buf.getvalue()
        self.assertIn('"config_ok": false', out_text)
        self.assertNotIn("智能笔记助手 CLI", out_text)

    def test_resolve_agent_home_prefers_env_var(self):
        with tempfile.TemporaryDirectory() as td:
            expected = Path(td).resolve()
            with patch.dict(os.environ, {"AGENT_HOME": str(expected)}, clear=False):
                with patch("main.Path.cwd", return_value=Path("/tmp/should-not-use")):
                    actual = main.resolve_agent_home()
        self.assertEqual(actual, expected)

    def test_resolve_agent_home_uses_cwd_when_env_or_data_exists(self):
        with tempfile.TemporaryDirectory() as td:
            cwd = Path(td).resolve()
            (cwd / ".env").write_text("API_KEY=\n", encoding="utf-8")
            with patch.dict(os.environ, {}, clear=True):
                with patch("main.Path.cwd", return_value=cwd):
                    actual = main.resolve_agent_home()
        self.assertEqual(actual, cwd)

    def test_resolve_agent_home_falls_back_to_user_home(self):
        with tempfile.TemporaryDirectory() as td:
            cwd = Path(td).resolve()
            with tempfile.TemporaryDirectory() as home_dir:
                home = Path(home_dir).resolve()
                with patch.dict(os.environ, {}, clear=True):
                    with patch("main.Path.cwd", return_value=cwd):
                        with patch("main.Path.home", return_value=home):
                            actual = main.resolve_agent_home()
        self.assertEqual(actual, home / ".noteai")

    def test_run_init_dry_run_works_without_repo_template(self):
        with tempfile.TemporaryDirectory() as td:
            agent_home = Path(td).resolve()
            runtime_paths = {
                "agent_home": agent_home,
                "env_file": agent_home / ".env",
                "env_example_fallback": agent_home / "missing.example",
            }
            with patch("main._read_env_example_text", return_value=("API_KEY=\n", "package:cli/.env.example")):
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    code = main.run_init(force=False, dry_run=True, output="json", runtime_paths=runtime_paths)

        self.assertEqual(code, main.EXIT_OK)
        out_text = buf.getvalue()
        self.assertIn('"dry_run": true', out_text)
        self.assertIn('"agent_home"', out_text)
        self.assertIn("package:cli/.env.example", out_text)

    def test_build_runtime_uses_explicit_runtime_paths(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td).resolve()
            env_path = root / ".env"
            data_dir = root / "data"
            faiss_dir = root / "faiss_index"
            runtime_paths = {
                "agent_home": root,
                "env_file": env_path,
                "data_dir": data_dir,
                "faiss_index_dir": faiss_dir,
                "env_example_fallback": root / ".env.example",
                "base_dir": root,
            }
            fake_settings = SimpleNamespace(
                restricted_query_limit_per_minute=5,
                log_file="observability/events.jsonl",
            )

            with patch("main.load_dotenv") as mock_load_dotenv, \
                 patch("main.Settings.from_env", return_value=fake_settings), \
                 patch("main.OpenAICompatibleClient", return_value="llm"), \
                 patch("main.SlidingWindowFrequencyGuard", return_value="guard"), \
                 patch("main._get_rag_engine_class", return_value=lambda **kwargs: "rag") as mock_rag_class, \
                 patch("main.create_agent_node", return_value="node"), \
                 patch("main.build_app", return_value="graph"):
                settings, rag, app = main.build_runtime(runtime_paths=runtime_paths)

        self.assertEqual(settings, fake_settings)
        self.assertEqual(rag, "rag")
        self.assertEqual(app, "graph")
        mock_load_dotenv.assert_called_once_with(env_path)
        mock_rag_class.assert_called_once_with()
        self.assertEqual(rag, "rag")
        self.assertEqual(app, "graph")
        self.assertTrue(Path(settings.log_file).is_absolute())

    def test_version_flag_does_not_require_rag_dependencies(self):
        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()
        with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
            with self.assertRaises(SystemExit) as exc:
                main.main(["--version"])

        self.assertEqual(exc.exception.code, 0)
        self.assertIn("noteai 0.3.0", stdout_buf.getvalue())


if __name__ == "__main__":
    unittest.main()
