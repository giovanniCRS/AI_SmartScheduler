"""Token counting and rate limiting for the Groq free tier
(8000 tokens/minute)."""
import threading
import time

try:
    import tiktoken
    _ENC = tiktoken.get_encoding("cl100k_base")  # closest available proxy;
    # qwen has no public tiktoken encoding, cl100k gives a solid estimate.
except Exception:  # pragma: no cover - tiktoken not installed / no internet
    tiktoken = None
    _ENC = None


def count_tokens(text: str) -> int:
    """Best-effort token count. Falls back to a chars/4 heuristic if
    tiktoken isn't available (e.g. offline sandbox)."""
    if not text:
        return 0
    if _ENC is not None:
        return len(_ENC.encode(text))
    return max(1, len(text) // 4)


class TokenBucket:
    """Simple token-bucket rate limiter: `capacity` tokens refill fully
    every `window_seconds`. `wait_for(n)` blocks the caller until n tokens
    are available, then debits them."""

    def __init__(self, capacity: int, window_seconds: float = 60.0):
        self.capacity = capacity
        self.window_seconds = window_seconds
        self._tokens = float(capacity)
        self._last_refill = time.monotonic()
        self._lock = threading.Lock()

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_refill
        if elapsed <= 0:
            return
        refill_rate = self.capacity / self.window_seconds  # tokens/sec
        self._tokens = min(self.capacity, self._tokens + elapsed * refill_rate)
        self._last_refill = now

    def wait_for(self, n_tokens: int, poll_interval: float = 0.5) -> float:
        """Block until n_tokens are available, debit them, return seconds
        waited. Raises ValueError if n_tokens exceeds bucket capacity
        (would never be satisfiable)."""
        if n_tokens > self.capacity:
            raise ValueError(
                f"Requested {n_tokens} tokens exceeds bucket capacity {self.capacity}"
            )
        waited = 0.0
        while True:
            with self._lock:
                self._refill()
                if self._tokens >= n_tokens:
                    self._tokens -= n_tokens
                    return waited
            time.sleep(poll_interval)
            waited += poll_interval
