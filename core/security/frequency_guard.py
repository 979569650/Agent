import time
from collections import deque


class SlidingWindowFrequencyGuard:
    """简单滑动窗口频控器（进程内）。"""

    def __init__(self, limit: int, window_seconds: int = 60):
        self.limit = max(1, int(limit))
        self.window_seconds = max(1, int(window_seconds))
        self._events = deque()

    def allow(self) -> bool:
        now = time.time()
        boundary = now - self.window_seconds

        while self._events and self._events[0] < boundary:
            self._events.popleft()

        if len(self._events) >= self.limit:
            return False

        self._events.append(now)
        return True


