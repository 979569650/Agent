import io
import unittest
from unittest import mock

from cli.fullscreen_terminal import (
    ANSI_ENTER_ALT_SCREEN,
    ANSI_EXIT_ALT_SCREEN,
    AlternateScreenSession,
    FullscreenSupport,
    enable_windows_vt_mode_if_needed,
    supports_fullscreen,
)


class _FakeTTY:
    def __init__(self, is_tty=True):
        self._is_tty = is_tty

    def isatty(self):
        return self._is_tty


class _FakeStdout(io.StringIO):
    def __init__(self, fail_on_exit=False):
        super().__init__()
        self.fail_on_exit = fail_on_exit

    def write(self, s):
        if self.fail_on_exit and s == ANSI_EXIT_ALT_SCREEN:
            raise RuntimeError("exit write failed")
        return super().write(s)


class TestFullscreenTerminal(unittest.TestCase):
    def test_supports_fullscreen_rejects_non_tty(self):
        support = supports_fullscreen(stdin=_FakeTTY(False), stdout=_FakeTTY(True))
        self.assertIsInstance(support, FullscreenSupport)
        self.assertFalse(support.supported)
        self.assertIn("TTY", support.reason)

    @mock.patch("cli.fullscreen_terminal.os.name", "posix")
    def test_supports_fullscreen_posix_term_empty(self):
        with mock.patch.dict("cli.fullscreen_terminal.os.environ", {"TERM": ""}, clear=False):
            support = supports_fullscreen(stdin=_FakeTTY(True), stdout=_FakeTTY(True))
        self.assertFalse(support.supported)
        self.assertIn("TERM", support.reason)

    @mock.patch("cli.fullscreen_terminal.os.name", "posix")
    def test_supports_fullscreen_posix_term_dumb(self):
        with mock.patch.dict("cli.fullscreen_terminal.os.environ", {"TERM": "dumb"}, clear=False):
            support = supports_fullscreen(stdin=_FakeTTY(True), stdout=_FakeTTY(True))
        self.assertFalse(support.supported)

    @mock.patch("cli.fullscreen_terminal.os.name", "posix")
    def test_supports_fullscreen_posix_ok(self):
        with mock.patch.dict("cli.fullscreen_terminal.os.environ", {"TERM": "xterm-256color"}, clear=False):
            support = supports_fullscreen(stdin=_FakeTTY(True), stdout=_FakeTTY(True))
        self.assertTrue(support.supported)

    @mock.patch("cli.fullscreen_terminal.os.name", "nt")
    @mock.patch("cli.fullscreen_terminal.enable_windows_vt_mode_if_needed", return_value=True)
    def test_supports_fullscreen_windows_ok(self, _):
        support = supports_fullscreen(stdin=_FakeTTY(True), stdout=_FakeTTY(True))
        self.assertTrue(support.supported)

    @mock.patch("cli.fullscreen_terminal.os.name", "nt")
    @mock.patch("cli.fullscreen_terminal.enable_windows_vt_mode_if_needed", return_value=False)
    def test_supports_fullscreen_windows_fail(self, _):
        support = supports_fullscreen(stdin=_FakeTTY(True), stdout=_FakeTTY(True))
        self.assertFalse(support.supported)
        self.assertIn("Windows VT", support.reason)

    @mock.patch("cli.fullscreen_terminal.os.name", "nt")
    def test_enable_windows_vt_mode_success(self):
        kernel32 = mock.MagicMock()
        kernel32.GetStdHandle.return_value = 1
        kernel32.GetConsoleMode.return_value = 1
        kernel32.SetConsoleMode.return_value = 1

        class _UInt32:
            def __init__(self):
                self.value = 0

        fake_ctypes = mock.MagicMock()
        fake_ctypes.windll = mock.MagicMock(kernel32=kernel32)
        fake_ctypes.c_uint32 = _UInt32
        fake_ctypes.byref = lambda x: x

        with mock.patch.dict("sys.modules", {"ctypes": fake_ctypes}):
            ok = enable_windows_vt_mode_if_needed()
        self.assertTrue(ok)

    @mock.patch("cli.fullscreen_terminal.os.name", "nt")
    def test_enable_windows_vt_mode_failure(self):
        kernel32 = mock.MagicMock()
        kernel32.GetStdHandle.return_value = -1
        fake_ctypes = mock.MagicMock()
        fake_ctypes.windll = mock.MagicMock(kernel32=kernel32)

        with mock.patch.dict("sys.modules", {"ctypes": fake_ctypes}):
            ok = enable_windows_vt_mode_if_needed()
        self.assertFalse(ok)

    def test_alternate_screen_session_exits_on_exception(self):
        stdout = _FakeStdout()

        with self.assertRaises(RuntimeError):
            with AlternateScreenSession(stdout=stdout):
                raise RuntimeError("boom")

        output = stdout.getvalue()
        self.assertIn(ANSI_ENTER_ALT_SCREEN, output)
        self.assertIn(ANSI_EXIT_ALT_SCREEN, output)

    def test_alternate_screen_session_exit_is_best_effort(self):
        stdout = _FakeStdout(fail_on_exit=True)
        session = AlternateScreenSession(stdout=stdout)
        session.enter()
        session.exit()
        self.assertFalse(session._entered)


if __name__ == "__main__":
    unittest.main()
