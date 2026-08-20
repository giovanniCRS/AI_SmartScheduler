"""Groq API wrapper with token-bucket rate limiting (8000 tok/min free
tier), exponential-backoff retries, and per-call logging.
"""
import time
from typing import Optional

import config
from llm.token_counter import TokenBucket, count_tokens
from utils.exceptions import RateLimitError
from utils.logger import setup_logger

logger = setup_logger(__name__)

try:
    import groq
except ImportError:  # pragma: no cover - offline / not installed
    groq = None


class GroqClient:
    """Thin wrapper around groq.Groq that enforces the free-tier rate
    limit locally (so we get smooth waits instead of 429s) and retries
    transient failures with exponential backoff."""

    def __init__(
        self,
        api_key: str = config.GROQ_API_KEY,
        model: str = config.MODEL_NAME,
        rate_limit: int = config.RATE_LIMIT_TOKENS_PER_MINUTE,
        window_seconds: float = config.RATE_LIMIT_WINDOW_SECONDS,
    ):
        if groq is None:
            raise ImportError(
                "The 'groq' package is not installed. Run "
                "`pip install groq` (see requirements.txt)."
            )
        self.client = groq.Groq(api_key=api_key)
        self.model = model
        self.token_bucket = TokenBucket(rate_limit, window_seconds)
        self.total_tokens_used = 0
        self.total_calls = 0

    def call(
        self,
        prompt: str,
        max_tokens: int = 2000,
        temperature: float = config.TEMPERATURE,
        response_format: Optional[dict] = None,
        reasoning_format: Optional[str] = config.REASONING_FORMAT,
        reasoning_effort: Optional[str] = None,
    ) -> str:
        """Blocking call: waits for rate-limit headroom, calls the API
        with retry/backoff, logs usage, and returns the text response.

        response_format: pass {"type": "json_object"} for calls that must
        return valid JSON (e.g. the preference agent). Leave None for
        free-form text/code (drafting, refinement). Automatically dropped
        and retried without it if Groq rejects the request server-side
        (400 json_validate_failed) -- callers should still be able to
        parse whatever plain text comes back in that case.
        reasoning_format: "hidden" strips the model's chain-of-thought
        from `content` for reasoning models (Qwen3, GPT-OSS, DeepSeek-R1
        distill, ...). If the connected model doesn't support the
        parameter, the call is transparently retried without it.
        reasoning_effort: "none" disables reasoning entirely (Qwen3 only)
        for calls that don't need it, e.g. simple extraction -- also
        auto-dropped if unsupported by the connected model.
        """
        prompt_tokens_est = count_tokens(prompt)
        budget_est = prompt_tokens_est + max_tokens

        waited = self.token_bucket.wait_for(min(budget_est, self.token_bucket.capacity))
        if waited > 0:
            logger.info(f"Rate limiter: waited {waited:.1f}s for token budget")

        last_error: Optional[Exception] = None
        drop_reasoning_format = False
        drop_reasoning_effort = False
        drop_response_format = False
        for attempt in range(1, config.MAX_RETRIES + 1):
            try:
                kwargs = dict(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                if response_format is not None and not drop_response_format:
                    kwargs["response_format"] = response_format
                if reasoning_format and not drop_reasoning_format:
                    kwargs["reasoning_format"] = reasoning_format
                if reasoning_effort and not drop_reasoning_effort:
                    kwargs["reasoning_effort"] = reasoning_effort

                response = self.client.chat.completions.create(**kwargs)
                content = response.choices[0].message.content or ""

                if not content.strip():
                    # Some reasoning models leave `content` empty when
                    # thinking wasn't hidden/parsed correctly; the actual
                    # answer may be sitting in a `reasoning` field instead.
                    reasoning_fallback = _extract_reasoning_fallback(response)
                    if reasoning_fallback:
                        logger.info(
                            "Groq response had empty content; recovered "
                            "text from the `reasoning` field instead."
                        )
                        content = reasoning_fallback

                usage = getattr(response, "usage", None)
                actual_tokens = (
                    getattr(usage, "total_tokens", None) or budget_est
                )
                self.total_tokens_used += actual_tokens
                self.total_calls += 1
                logger.info(
                    f"Groq call ok (attempt {attempt}): "
                    f"~{actual_tokens} tokens, model={self.model}"
                )
                return content
            except Exception as e:  # groq raises various *Error subclasses
                last_error = e
                msg = str(e).lower()

                # Unsupported-parameter and server-side JSON-validation
                # failures are retried immediately (no backoff burn) with
                # the offending option stripped, since the params -- not
                # transient load -- are the problem.
                if "reasoning_format" in msg and reasoning_format and not drop_reasoning_format:
                    drop_reasoning_format = True
                    logger.info("Model rejected reasoning_format; retrying without it")
                    continue
                if "reasoning_effort" in msg and reasoning_effort and not drop_reasoning_effort:
                    drop_reasoning_effort = True
                    logger.info("Model rejected reasoning_effort; retrying without it")
                    continue
                if (
                    "json_validate_failed" in msg or "failed to validate json" in msg
                ) and response_format is not None and not drop_response_format:
                    drop_response_format = True
                    logger.info(
                        "Groq rejected response_format=json_object for this "
                        "generation (json_validate_failed); retrying as plain "
                        "text -- caller must be able to extract JSON itself."
                    )
                    continue

                is_rate_error = "rate" in msg or "429" in msg
                backoff = config.BACKOFF_BASE_SECONDS * (2 ** (attempt - 1))
                logger.info(
                    f"Groq call failed (attempt {attempt}/{config.MAX_RETRIES}): "
                    f"{e}. Retrying in {backoff:.1f}s"
                    + (" [rate limited]" if is_rate_error else "")
                )
                time.sleep(backoff)

        raise RateLimitError(
            f"Groq call failed after {config.MAX_RETRIES} attempts: {last_error}"
        )


def strip_reasoning_tags(text: str) -> str:
    """Defensive cleanup: if reasoning_format="hidden" wasn't honored (or
    the model wasn't a Groq reasoning model to begin with, e.g. after
    swapping providers), strip any literal <think>...</think> block the
    model left in `content` before it's parsed as JSON/code."""
    import re

    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _extract_reasoning_fallback(response) -> str:
    """Best-effort recovery when content is empty: some SDK/model
    combinations put the full text in message.reasoning instead."""
    try:
        message = response.choices[0].message
        reasoning = getattr(message, "reasoning", None)
        return reasoning or ""
    except (AttributeError, IndexError):
        return ""
