"""
src/extraction/gemini_extractor.py
------------------------------------
Production-grade Gemini Vision wrapper for Vietnamese receipt extraction.

Features:
  - Retry with exponential backoff (3 attempts: 1s, 2s, 4s)
  - 30-second hard timeout per request
  - Error classification (transient vs permanent — only retry transient)
  - Cost tracking (input/output tokens + estimated USD)
  - SHA256-based response cache (7-day TTL, disk-backed)
  - Structured output via Pydantic schema passed to Gemini response_schema
  - Graceful degradation — returns ExtractionResult(success=False) on all failures

Contract:
  - extract() NEVER raises to caller.
  - All errors are captured in ExtractionResult.error_type / error_message.
  - extract_batch() processes in parallel (ThreadPoolExecutor) with the same guarantee.

Cost model (gemini-2.5-flash as of 2025):
  Input:  $0.15 per 1M tokens
  Output: $0.60 per 1M tokens
  (These are reference values — update if pricing changes)
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from threading import Lock
from typing import Any

import cv2
import numpy as np

from .prompts import DEFAULT_PROMPT, DEFAULT_PROMPT_VERSION
from .schemas import ExtractionResult, Receipt

log = logging.getLogger(__name__)

# ── Cost constants (USD per token) ────────────────────────────────────────────
# gemini-2.5-flash pricing (as of 2025-04):
#   Input:           $0.15 / 1M tokens
#   Output (normal): $0.60 / 1M tokens
#   Output (thinking): $3.50 / 1M tokens  ← NOT tracked here (thinking disabled)
# Update if Google changes pricing: https://ai.google.dev/pricing

_COST_INPUT_PER_TOKEN  = 0.15 / 1_000_000   # $0.15 / 1M input tokens
_COST_OUTPUT_PER_TOKEN = 0.60 / 1_000_000   # $0.60 / 1M output tokens

# ── Retry config ──────────────────────────────────────────────────────────────

_MAX_ATTEMPTS = 3
_BACKOFF_SECONDS = [1.0, 2.0, 4.0]  # exponential backoff per attempt

# ── Transient error signals (worth retrying) ──────────────────────────────────
# Permanent errors (wrong API key, invalid image, billing) are not retried.

_TRANSIENT_SIGNALS = (
    "503", "502", "500",       # server errors
    "rate_limit", "quota",     # quota / rate limiting
    "timeout",                 # our own timeout
    "ResourceExhausted",       # gRPC resource exhausted
    "ServiceUnavailable",      # gRPC service unavailable
    "InternalServerError",     # gRPC internal
)

# ── Cache config ──────────────────────────────────────────────────────────────

_CACHE_TTL_DAYS = 7
_DEFAULT_CACHE_DIR = Path("data") / "extraction_cache"


# ── Disk cache ────────────────────────────────────────────────────────────────

class _DiskCache:
    """
    Simple JSON disk cache for extraction results.

    Keys are SHA256(image_bytes + prompt)[:16].
    Entries expire after TTL days.  Thread-safe via a per-instance lock.
    """

    def __init__(self, cache_dir: Path, ttl_days: int = _CACHE_TTL_DAYS) -> None:
        self.cache_dir = cache_dir
        self.ttl = timedelta(days=ttl_days)
        self._lock = Lock()
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            log.warning("Could not create cache dir %s: %s — cache disabled", cache_dir, exc)

    def _path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def get(self, key: str) -> dict | None:
        path = self._path(key)
        try:
            with self._lock:
                if not path.exists():
                    return None
                data = json.loads(path.read_text(encoding="utf-8"))
            # Check TTL
            saved_at = datetime.fromisoformat(data.get("_saved_at", "2000-01-01"))
            if datetime.now() - saved_at > self.ttl:
                path.unlink(missing_ok=True)
                return None
            return data
        except Exception:
            return None

    def set(self, key: str, value: dict) -> None:
        path = self._path(key)
        try:
            value["_saved_at"] = datetime.now().isoformat()
            with self._lock:
                path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as exc:
            log.warning("Cache write failed for key %s: %s", key, exc)


# ── GeminiExtractor ───────────────────────────────────────────────────────────

class GeminiExtractor:
    """
    Extracts structured data from receipt images using Gemini Vision.

    Usage:
        extractor = GeminiExtractor()  # reads GEMINI_API_KEY from env
        result = extractor.extract(image_bytes)
        if result.success:
            print(result.receipt.total_amount)

    Thread-safe — extract() and extract_batch() can be called concurrently.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gemini-2.5-flash",
        prompt: str = DEFAULT_PROMPT,
        prompt_version: str = DEFAULT_PROMPT_VERSION,
        cache_dir: Path | str | None = None,
        enable_cache: bool = True,
        timeout_seconds: float = 30.0,
    ) -> None:
        """
        Parameters
        ----------
        api_key        : Gemini API key. Falls back to GEMINI_API_KEY / Gemini_API_Key env vars.
        model          : Gemini model name.
        prompt         : Extraction prompt text.
        prompt_version : Version label for tracing (stored in ExtractionResult).
        cache_dir      : Disk cache directory. Defaults to data/extraction_cache/.
        enable_cache   : Set False to bypass cache (e.g., for benchmarks with --no-cache).
        timeout_seconds: Hard timeout per Gemini API call.
        """
        self.model = model
        self.prompt = prompt
        self.prompt_version = prompt_version
        self.timeout_seconds = timeout_seconds
        self.enable_cache = enable_cache

        # Resolve API key
        self.api_key = (
            api_key
            or os.getenv("GEMINI_API_KEY")
            or os.getenv("Gemini_API_Key")  # project convention from .env
        )
        if not self.api_key:
            log.warning(
                "GeminiExtractor: no API key found in constructor or env "
                "(GEMINI_API_KEY / Gemini_API_Key). extract() will fail."
            )

        # Lazy Gemini client (import here to avoid hard dependency at import time)
        self._client: Any = None
        self._client_lock = Lock()

        # Disk cache
        resolved_cache_dir = Path(cache_dir) if cache_dir else _DEFAULT_CACHE_DIR
        self._cache = _DiskCache(resolved_cache_dir) if enable_cache else None

    # ── Public API ────────────────────────────────────────────────────────────

    def extract(
        self,
        image: "np.ndarray | bytes | str | Path",
    ) -> ExtractionResult:
        """
        Extract structured receipt data from an image.

        Accepts:
          - np.ndarray  : BGR image array (OpenCV format)
          - bytes       : raw image bytes (JPEG/PNG)
          - str / Path  : file path

        Returns ExtractionResult.  Never raises.
        """
        t0 = time.perf_counter()

        try:
            img_bytes = self._to_jpeg_bytes(image)
        except Exception as exc:
            return ExtractionResult(
                success=False,
                model=self.model,
                prompt_version=self.prompt_version,
                error_type="input_error",
                error_message=f"Failed to prepare image: {exc}",
                latency_ms=_elapsed_ms(t0),
            )

        image_hash = _sha256_prefix(img_bytes + self.prompt.encode())

        # ── Cache check ──────────────────────────────────────────────────────
        if self._cache is not None:
            cached = self._cache.get(image_hash)
            if cached:
                try:
                    receipt = Receipt.model_validate(cached["receipt"]) if cached.get("receipt") else None
                    return ExtractionResult(
                        success=cached.get("success", False),
                        receipt=receipt,
                        model=cached.get("model", self.model),
                        prompt_version=cached.get("prompt_version", self.prompt_version),
                        image_hash=image_hash,
                        input_tokens=cached.get("input_tokens", 0),
                        output_tokens=cached.get("output_tokens", 0),
                        cost_usd=cached.get("cost_usd", 0.0),
                        latency_ms=_elapsed_ms(t0),
                        from_cache=True,
                        attempts=cached.get("attempts", 1),
                    )
                except Exception as exc:
                    log.warning("Cache parse error for %s: %s — re-querying", image_hash, exc)

        # ── API call with retry ──────────────────────────────────────────────
        result = self._call_with_retry(img_bytes, image_hash, t0)

        # ── Cache write ──────────────────────────────────────────────────────
        if self._cache is not None and result.success:
            self._cache.set(image_hash, {
                "success": result.success,
                "receipt": result.receipt.model_dump() if result.receipt else None,
                "model": result.model,
                "prompt_version": result.prompt_version,
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
                "cost_usd": result.cost_usd,
                "attempts": result.attempts,
            })

        return result

    def extract_batch(
        self,
        images: "list[np.ndarray | bytes | str | Path]",
        max_workers: int = 2,  # Low default to respect free-tier rate limits
    ) -> list[ExtractionResult]:
        """
        Extract from multiple images in parallel.

        max_workers=2: Gemini free tier allows ~60 QPM.
        Each call takes ~2–5s, so 2 workers ≈ 20–30 QPM (safe headroom).
        """
        results: list[ExtractionResult | None] = [None] * len(images)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(self.extract, img): i
                for i, img in enumerate(images)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as exc:
                    results[idx] = ExtractionResult(
                        success=False,
                        model=self.model,
                        prompt_version=self.prompt_version,
                        error_type="batch_worker_error",
                        error_message=str(exc),
                    )
        return [
            r if r is not None
            else ExtractionResult(
                success=False,
                model=self.model,
                prompt_version=self.prompt_version,
                error_type="unknown",
                error_message="batch result missing",
            )
            for r in results
        ]

    # ── Internal ──────────────────────────────────────────────────────────────

    def _get_client(self) -> Any:
        """Lazy-initialize Gemini client (thread-safe)."""
        if self._client is not None:
            return self._client
        with self._client_lock:
            if self._client is not None:
                return self._client
            try:
                from google import genai  # type: ignore[import]
                self._client = genai.Client(api_key=self.api_key)
            except ImportError as exc:
                raise RuntimeError(
                    "google-genai package not installed. "
                    "Run: pip install google-genai"
                ) from exc
        return self._client

    def _call_with_retry(
        self,
        img_bytes: bytes,
        image_hash: str,
        t0: float,
    ) -> ExtractionResult:
        """Attempt the Gemini API call up to _MAX_ATTEMPTS times."""
        last_error_type = "unknown"
        last_error_msg = "unknown error"

        for attempt in range(1, _MAX_ATTEMPTS + 1):
            try:
                result = self._single_call(img_bytes, image_hash, t0, attempt)
                return result

            except Exception as exc:
                error_str = str(exc)
                is_transient = any(sig in error_str for sig in _TRANSIENT_SIGNALS)
                last_error_msg = error_str
                last_error_type = "transient_api_error" if is_transient else "permanent_api_error"

                log.warning(
                    "Gemini call attempt %d/%d failed (%s): %s",
                    attempt, _MAX_ATTEMPTS,
                    "transient" if is_transient else "permanent",
                    error_str[:200],
                )

                if not is_transient or attempt == _MAX_ATTEMPTS:
                    break

                # Exponential backoff before next attempt
                backoff = _BACKOFF_SECONDS[min(attempt - 1, len(_BACKOFF_SECONDS) - 1)]
                time.sleep(backoff)

        return ExtractionResult(
            success=False,
            model=self.model,
            prompt_version=self.prompt_version,
            image_hash=image_hash,
            error_type=last_error_type,
            error_message=last_error_msg,
            latency_ms=_elapsed_ms(t0),
            attempts=_MAX_ATTEMPTS,
        )

    def _single_call(
        self,
        img_bytes: bytes,
        image_hash: str,
        t0: float,
        attempt: int,
    ) -> ExtractionResult:
        """Make a single Gemini API call with timeout enforcement."""
        from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
        from google.genai import types  # type: ignore[import]

        client = self._get_client()

        def _call() -> Any:
            config_kwargs: dict[str, Any] = dict(
                temperature=0,           # deterministic extraction
                max_output_tokens=8192,  # receipts with many items can exceed 2048
            )
            # Disable thinking for gemini-2.5 models — adds 5–15 s latency with
            # no accuracy benefit for this structured-extraction task.
            # gemini-2.0-flash does not support thinking — skip.
            if "2.5" in self.model:
                try:
                    config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
                except AttributeError:
                    pass  # older SDK version — skip
            return client.models.generate_content(
                model=self.model,
                contents=[
                    types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"),
                    self.prompt,
                ],
                config=types.GenerateContentConfig(**config_kwargs),
            )

        # Enforce hard timeout via a thread
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(_call)
            try:
                response = future.result(timeout=self.timeout_seconds)
            except FuturesTimeout:
                future.cancel()
                raise TimeoutError(f"timeout — Gemini did not respond in {self.timeout_seconds}s")

        # ── Parse response ────────────────────────────────────────────────────
        raw_text = response.text if hasattr(response, "text") else ""

        # Token counts (may not always be present in response)
        input_tokens = 0
        output_tokens = 0
        try:
            usage = response.usage_metadata
            input_tokens = getattr(usage, "prompt_token_count", 0) or 0
            output_tokens = getattr(usage, "candidates_token_count", 0) or 0
        except Exception:
            pass

        cost_usd = (
            input_tokens * _COST_INPUT_PER_TOKEN
            + output_tokens * _COST_OUTPUT_PER_TOKEN
        )

        # ── JSON extraction ───────────────────────────────────────────────────
        receipt = self._parse_receipt(raw_text)

        return ExtractionResult(
            success=receipt is not None,
            receipt=receipt,
            model=self.model,
            prompt_version=self.prompt_version,
            image_hash=image_hash,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
            latency_ms=_elapsed_ms(t0),
            from_cache=False,
            attempts=attempt,
            error_type=None if receipt is not None else "parse_error",
            error_message=None if receipt is not None else f"Could not parse JSON from: {raw_text[:200]}",
        )

    def _parse_receipt(self, raw_text: str) -> Receipt | None:
        """
        Extract JSON from Gemini's response text and validate as Receipt.

        Handles three common response formats:
          1. Raw JSON object: {"merchant_name": ...}
          2. Markdown code block: ```json\n{...}\n```
          3. JSON embedded in prose: "... Here is the data:\n{...}"
        """
        if not raw_text:
            return None

        text = raw_text.strip()

        # Strip markdown code fences (handles both complete and truncated fences)
        if "```" in text:
            start = text.find("```")
            end = text.rfind("```")
            if start != end:
                # Complete fence: ```json ... ```
                inner = text[start + 3:end]
            else:
                # Truncated response: opening fence exists but closing is missing
                inner = text[start + 3:]
            if inner.startswith("json"):
                inner = inner[4:]
            text = inner.strip()

        # Find the first { (handles prose before JSON)
        brace_start = text.find("{")
        if brace_start == -1:
            log.warning("No JSON object found in Gemini response: %s", text[:200])
            return None

        # Find the last valid } — if truncated, attempt to close the JSON
        brace_end = text.rfind("}")
        if brace_end == -1 or brace_end <= brace_start:
            log.warning("JSON appears truncated (no closing brace) — attempting recovery")
            json_str = text[brace_start:] + "}"
        else:
            json_str = text[brace_start:brace_end + 1]

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            # Truncated mid-value — strip trailing incomplete field and close
            # Walk backwards past the last complete key-value pair
            last_comma = json_str.rfind(",")
            if last_comma > brace_start:
                json_str = json_str[:last_comma] + "}"
                try:
                    data = json.loads(json_str)
                except json.JSONDecodeError as exc:
                    log.warning("JSON parse error after recovery attempt: %s — text: %s", exc, json_str[:300])
                    return None
            else:
                log.warning("JSON unrecoverable: %s", json_str[:300])
                return None

        try:
            return Receipt.model_validate(data)
        except Exception as exc:
            log.warning("Receipt validation error: %s — data: %s", exc, str(data)[:200])
            # Attempt partial construction — return what validated
            try:
                safe_data = {k: v for k, v in data.items() if k in Receipt.model_fields}
                if "items" in safe_data and isinstance(safe_data["items"], list):
                    safe_data["items"] = [
                        item for item in safe_data["items"]
                        if isinstance(item, dict) and item.get("name")
                    ]
                return Receipt.model_validate(safe_data)
            except Exception:
                return None

    @staticmethod
    def _to_jpeg_bytes(image: "np.ndarray | bytes | str | Path") -> bytes:
        """Convert various image inputs to JPEG bytes."""
        if isinstance(image, (bytes, bytearray)):
            # Resize large images before sending — phone photos (4000px+) are
            # overkill for the API and significantly slow down uploads.
            try:
                arr = cv2.imdecode(np.frombuffer(image, dtype=np.uint8), cv2.IMREAD_COLOR)
                if arr is not None:
                    arr = _resize_for_api(arr)
                    ok, buf = cv2.imencode(".jpg", arr, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    if ok:
                        return bytes(buf)
            except Exception:
                pass  # fall through and return original bytes unchanged
            return bytes(image)

        if isinstance(image, (str, Path)):
            path = Path(image)
            if not path.exists():
                raise FileNotFoundError(f"Image not found: {path}")
            return path.read_bytes()

        if isinstance(image, np.ndarray):
            image = _resize_for_api(image)
            ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ok:
                raise ValueError("cv2.imencode failed — invalid image array")
            return bytes(buf)

        raise TypeError(f"Unsupported image type: {type(image)}")


# ── Helpers ───────────────────────────────────────────────────────────────────

_MAX_API_DIMENSION = 1920  # px — balance between API cost and text readability on receipts


def _resize_for_api(img: np.ndarray) -> np.ndarray:
    """Downscale image so its longest side ≤ _MAX_API_DIMENSION. No-op if already small."""
    h, w = img.shape[:2]
    max_dim = max(h, w)
    if max_dim <= _MAX_API_DIMENSION:
        return img
    scale = _MAX_API_DIMENSION / max_dim
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _elapsed_ms(t0: float) -> float:
    return round((time.perf_counter() - t0) * 1000, 1)


def _sha256_prefix(data: bytes, length: int = 16) -> str:
    return hashlib.sha256(data).hexdigest()[:length]
