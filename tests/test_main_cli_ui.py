import contextlib
import io
import unittest
from types import SimpleNamespace

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

        self.assertIn("Agent CLI", text)
        self.assertIn("本地智能笔记助理", text)
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
        self.assertNotIn("Agent CLI", out_text)


if __name__ == "__main__":
    unittest.main()
