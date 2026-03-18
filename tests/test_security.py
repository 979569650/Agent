import unittest

from core.observability.logger import _sanitize_fields
from core.security.auth import verify_access_code
from core.security.frequency_guard import SlidingWindowFrequencyGuard


class TestSecurity(unittest.TestCase):
    def test_sha256_compat_verify(self):
        expected = "2bb80d537b1da3e38bd30361aa855686bde0eacd7162fef6a25fe97bf527a25b"
        self.assertTrue(verify_access_code("secret", expected))
        self.assertFalse(verify_access_code("wrong", expected))

    def test_frequency_guarder(self):
        limiter = SlidingWindowFrequencyGuard(limit=2, window_seconds=60)
        self.assertTrue(limiter.allow())
        self.assertTrue(limiter.allow())
        self.assertFalse(limiter.allow())

    def test_log_sanitize(self):
        payload = _sanitize_fields({"api_key": "sk-xxx", "normal": "ok", "token_value": "abc"})
        self.assertEqual(payload["api_key"], "***REDACTED***")
        self.assertEqual(payload["token_value"], "***REDACTED***")
        self.assertEqual(payload["normal"], "ok")


if __name__ == "__main__":
    unittest.main()


