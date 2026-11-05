"""Thin async client for an OpenAI-compatible vLLM endpoint."""

import asyncio
import os
import re
import time
from dataclasses import dataclass

from openai import AsyncOpenAI

THINK_RE = re.compile(r"<think>.*?</think>\s*", flags=re.DOTALL)


@dataclass
class Completion:
    text: str
    finish_reason: str | None
    prompt_tokens: int
    completion_tokens: int
    latency_s: float
    error: str | None = None
    tokens: list[str] | None = None  # Phase 1: generated tokens (decoded from their UTF-8 bytes), if requested
    logprobs: list[float] | None = None  # Phase 1: log-probability of each generated token


def tokens_from_logprobs(content) -> tuple[list[str], list[float]]:
    """Token strings and logprobs from an OpenAI-style `logprobs.content` list.

    Tokens are rebuilt from their UTF-8 `bytes` (a multi-byte character can be split across tokens, so
    the per-token `token` strings may not concatenate to the text); a token whose bytes are not a valid
    UTF-8 sequence on its own is merged with the following token(s) until they are. The concatenation of
    the returned tokens equals the raw completion text, which lets the caller map every character of the
    output to the token (and logprob) that produced it.
    """
    tokens, lps, buf, buf_lp = [], [], b"", []
    for item in content:
        b = bytes(item.bytes) if item.bytes is not None else item.token.encode("utf-8")
        buf += b
        buf_lp.append(float(item.logprob))
        try:
            s = buf.decode("utf-8")
        except UnicodeDecodeError:
            continue
        tokens.append(s)
        lps.append(min(buf_lp))  # merged partial-character tokens: keep the least likely piece
        buf, buf_lp = b"", []
    if buf:  # trailing undecodable bytes: keep them as replacement characters
        tokens.append(buf.decode("utf-8", errors="replace"))
        lps.append(min(buf_lp))
    return tokens, lps


class LLM:
    def __init__(self, endpoint_cfg: dict, gen_cfg: dict, base_url: str | None = None):
        key_env = endpoint_cfg["api_key_env"]
        api_key = os.environ.get(key_env)
        if not api_key:
            raise SystemExit(f"Environment variable {key_env} is not set (bearer token for the vLLM endpoint).")
        self.base_url = base_url or endpoint_cfg["base_url"]
        self.model = endpoint_cfg["model"]
        self.enable_thinking = bool(endpoint_cfg.get("enable_thinking", False))
        self.max_retries = int(endpoint_cfg.get("max_retries", 3))
        self.logprobs = bool(endpoint_cfg.get("logprobs", False))  # Phase 1: per-token logprobs of the output
        # Qwen3's chat template takes enable_thinking; other families (e.g. Mistral served with vLLM's mistral
        # tokenizer mode) reject chat_template_kwargs with HTTP 400, so the kwarg is only sent for Qwen3 models.
        send = endpoint_cfg.get("send_thinking_kwarg", "auto")
        self.send_thinking_kwarg = ("qwen3" in self.model.lower()) if send == "auto" else bool(send)
        self.gen = gen_cfg
        self.client = AsyncOpenAI(
            base_url=self.base_url, api_key=api_key, timeout=float(endpoint_cfg.get("timeout_s", 900)), max_retries=0
        )
        self.sem = asyncio.Semaphore(int(endpoint_cfg.get("concurrency", 64)))

    async def check(self) -> list[str]:
        """Fail fast if the endpoint is unreachable or does not serve the configured model."""
        models = [m.id async for m in self.client.models.list()]
        if self.model not in models:
            raise SystemExit(f"Endpoint {self.base_url} serves {models}, not {self.model!r}")
        return models

    async def chat(self, messages: list[dict], max_tokens: int) -> Completion:
        async with self.sem:
            last_err = None
            for attempt in range(self.max_retries + 1):
                t0 = time.perf_counter()
                try:
                    resp = await self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=self.gen["temperature"],
                        top_p=self.gen["top_p"],
                        seed=self.gen["seed"],
                        logprobs=self.logprobs or None,
                        extra_body=({"chat_template_kwargs": {"enable_thinking": self.enable_thinking}}
                                    if self.send_thinking_kwarg else {}),
                    )
                    choice = resp.choices[0]
                    text = THINK_RE.sub("", choice.message.content or "")
                    usage = resp.usage
                    tokens = lps = None
                    if self.logprobs and choice.logprobs and choice.logprobs.content:
                        tokens, lps = tokens_from_logprobs(choice.logprobs.content)
                    return Completion(
                        text=text,
                        finish_reason=choice.finish_reason,
                        prompt_tokens=usage.prompt_tokens if usage else -1,
                        completion_tokens=usage.completion_tokens if usage else -1,
                        latency_s=time.perf_counter() - t0,
                        tokens=tokens,
                        logprobs=lps,
                    )
                except Exception as e:  # network / server errors: retry with backoff
                    last_err = f"{type(e).__name__}: {e}"
                    if attempt < self.max_retries:
                        await asyncio.sleep(2**attempt)
            return Completion("", None, -1, -1, 0.0, error=last_err)
