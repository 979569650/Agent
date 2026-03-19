import os
import sys
from dataclasses import dataclass

ANSI_ENTER_ALT_SCREEN = "\x1b[?1049h\x1b[2J\x1b[H"
ANSI_EXIT_ALT_SCREEN = "\x1b[?1049l"


@dataclass
class FullscreenSupport:
    supported: bool
    reason: str = ""


def _stream_is_tty(stream) -> bool:
    return bool(stream and hasattr(stream, "isatty") and stream.isatty())


def enable_windows_vt_mode_if_needed() -> bool:
    if os.name != "nt":
        return True

    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetStdHandle(-11)
        if handle in (0, -1):
            return False

        mode = ctypes.c_uint32()
        if kernel32.GetConsoleMode(handle, ctypes.byref(mode)) == 0:
            return False

        ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004
        new_mode = mode.value | ENABLE_VIRTUAL_TERMINAL_PROCESSING
        if kernel32.SetConsoleMode(handle, new_mode) == 0:
            return False

        return True
    except Exception:
        return False


def supports_fullscreen(stdin=None, stdout=None) -> FullscreenSupport:
    stdin = sys.stdin if stdin is None else stdin
    stdout = sys.stdout if stdout is None else stdout

    if not _stream_is_tty(stdin) or not _stream_is_tty(stdout):
        return FullscreenSupport(False, "stdin/stdout 不是 TTY")

    if os.name != "nt":
        term = os.getenv("TERM", "").strip()
        if not term:
            return FullscreenSupport(False, "TERM 未设置")
        if term.lower() == "dumb":
            return FullscreenSupport(False, "TERM=dumb")
        return FullscreenSupport(True, "")

    if enable_windows_vt_mode_if_needed():
        return FullscreenSupport(True, "")

    return FullscreenSupport(False, "Windows VT 模式不可用")


def safe_print(message: str = "", *, stream=None) -> None:
    stream = sys.stdout if stream is None else stream
    try:
        stream.write(message)
        stream.write("\n")
        if hasattr(stream, "flush"):
            stream.flush()
    except Exception:
        pass


class AlternateScreenSession:
    def __init__(self, stdout=None):
        self.stdout = sys.stdout if stdout is None else stdout
        self._entered = False

    def __enter__(self):
        self.enter()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.exit()
        return False

    def enter(self) -> None:
        if self._entered:
            return
        try:
            self.stdout.write(ANSI_ENTER_ALT_SCREEN)
            if hasattr(self.stdout, "flush"):
                self.stdout.flush()
        finally:
            self._entered = True

    def exit(self) -> None:
        if not self._entered:
            return
        try:
            self.stdout.write(ANSI_EXIT_ALT_SCREEN)
            if hasattr(self.stdout, "flush"):
                self.stdout.flush()
        except Exception:
            pass
        finally:
            self._entered = False
