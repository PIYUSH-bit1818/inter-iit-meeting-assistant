"""Stage 2: domain-aware transcript refinement (LLM A).

Takes the ``RawTranscript`` from speech-to-text and returns a separate
``RefinedTranscript``. The raw transcript is never modified.

Flow per recording:
1. Segments are sent to the model in batches (with the whole transcript as
   read-only context when there is more than one batch, so terminology stays
   consistent). Timestamps are never sent, so the model cannot change them.
2. The model returns strict JSON (schema-constrained): one entry per segment
   with the refined text and the edits it made.
3. Every proposed segment goes through ``guards.check_refinement``. Anything
   suspicious is rejected and the raw text is kept for that segment.

Gemini-specific code lives in ``GeminiRefineProvider``.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from functools import lru_cache
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, ValidationError

from ..config import REPO_ROOT, Settings, get_settings
from ..schemas import (
    RawTranscript,
    RefinedTranscript,
    RefinementStatus,
    Segment,
    SegmentRefinement,
    SpanChange,
)
from .guards import check_refinement

log = logging.getLogger(__name__)

PROMPT_VERSION = "transcript_refinement_v1"
PROMPT_PATH = REPO_ROOT / "prompts" / f"{PROMPT_VERSION}.txt"
EFFORT_LEVELS = ("low", "medium", "high", "xhigh", "max")
# Re-ask once if the model's JSON is unusable, then give up on the batch.
INVALID_OUTPUT_RETRIES = 1


class RefineError(Exception):
    """Refinement failed. ``message`` is user-facing; ``code`` is stable."""

    def __init__(self, message: str, code: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.retryable = retryable


@lru_cache
def load_prompt() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Model output contract
# ---------------------------------------------------------------------------


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ModelChange(_Strict):
    original_span: str
    refined_span: str
    reason: str


class ModelSegment(_Strict):
    segment_id: int
    original_text: str
    refined_text: str
    changed: bool
    changes: list[ModelChange]


class ModelOutput(_Strict):
    segments: list[ModelSegment]


OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "segments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "segment_id": {"type": "integer"},
                    "original_text": {"type": "string"},
                    "refined_text": {"type": "string"},
                    "changed": {"type": "boolean"},
                    "changes": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "original_span": {"type": "string"},
                                "refined_span": {"type": "string"},
                                "reason": {"type": "string"},
                            },
                            "required": ["original_span", "refined_span", "reason"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["segment_id", "original_text", "refined_text", "changed", "changes"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["segments"],
    "additionalProperties": False,
}


# ---------------------------------------------------------------------------
# Provider interface
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProviderReply:
    text: str
    stop_reason: str | None


class RefineProvider(Protocol):
    name: str
    model: str

    def complete(self, system: str, context: str | None, request: str) -> ProviderReply:
        """Send one request; return the model's JSON text and stop reason."""
        ...


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def refine(
    raw: RawTranscript,
    *,
    provider: RefineProvider | None = None,
    settings: Settings | None = None,
) -> RefinedTranscript:
    settings = settings or get_settings()
    provider = provider or GeminiRefineProvider.from_settings(settings)

    to_refine = [s for s in raw.segments if s.text.strip()]
    batches = _batch(to_refine, settings.refine_max_words_per_request)
    context = _context(raw.segments) if len(batches) > 1 else None

    proposals: dict[int, ModelSegment] = {}
    duplicates: set[int] = set()
    for batch in batches:
        for item in _refine_batch(provider, batch, context):
            if item.segment_id in proposals:
                duplicates.add(item.segment_id)
            proposals[item.segment_id] = item

    known = {s.id for s in raw.segments}
    unknown = sorted(set(proposals) - known)
    if unknown:
        log.warning("refinement returned unknown segment ids %s; ignored", unknown)

    refinements = [_review(seg, proposals.get(seg.id), seg.id in duplicates) for seg in raw.segments]
    accepted = sum(r.status == RefinementStatus.REFINED for r in refinements)
    rejected = sum(r.status == RefinementStatus.FALLBACK for r in refinements)
    log.info("refinement: %d refined, %d rejected, %d unchanged",
             accepted, rejected, len(refinements) - accepted - rejected)

    return RefinedTranscript(
        segments=[
            Segment(id=r.segment_id, start=r.start, end=r.end, text=r.refined_text)
            for r in refinements
        ],
        refinements=refinements,
        # The model(s) that actually answered (differs from the primary if a
        # fallback model was used)
        refine_model=", ".join(getattr(provider, "models_used", None) or [provider.model]),
        prompt_version=PROMPT_VERSION,
    )


# ---------------------------------------------------------------------------
# Batching and requests
# ---------------------------------------------------------------------------


def _batch(segments: list[Segment], max_words: int) -> list[list[Segment]]:
    batches: list[list[Segment]] = []
    current: list[Segment] = []
    words = 0
    for seg in segments:
        n = len(seg.text.split())
        if current and words + n > max_words:
            batches.append(current)
            current, words = [], 0
        current.append(seg)
        words += n
    if current:
        batches.append(current)
    return batches


def _context(segments: list[Segment]) -> str:
    lines = "\n".join(f"[{s.id}] {s.text}" for s in segments if s.text.strip())
    return (
        "Full meeting transcript, for terminology context only. "
        "Do not refine or return these lines unless asked below.\n\n" + lines
    )


def _refine_batch(provider: RefineProvider, batch: list[Segment], context: str | None) -> list[ModelSegment]:
    request = (
        "Refine these transcript segments. Return exactly one entry per segment.\n\n"
        + json.dumps({"segments": [{"segment_id": s.id, "text": s.text} for s in batch]},
                     ensure_ascii=False)
    )
    for attempt in range(INVALID_OUTPUT_RETRIES + 1):
        reply = provider.complete(load_prompt(), context, request)

        if reply.stop_reason == "blocked":
            raise RefineError(
                "The refinement model declined to process this transcript.", code="refused"
            )
        if reply.stop_reason == "max_tokens":
            if len(batch) > 1:
                mid = len(batch) // 2
                log.info("refinement output truncated; splitting batch of %d", len(batch))
                return (_refine_batch(provider, batch[:mid], context)
                        + _refine_batch(provider, batch[mid:], context))
            raise RefineError(
                "The refinement output was too long for a single segment.", code="output_truncated"
            )

        try:
            return ModelOutput.model_validate_json(reply.text).segments
        except ValidationError as exc:
            log.warning("invalid refinement output (attempt %d): %s", attempt + 1,
                        exc.errors(include_input=False)[:3])

    raise RefineError(
        "The refinement model returned output that could not be understood.",
        code="invalid_response",
    )


# ---------------------------------------------------------------------------
# Per-segment review
# ---------------------------------------------------------------------------


def _review(seg: Segment, proposal: ModelSegment | None, duplicated: bool) -> SegmentRefinement:
    base = dict(segment_id=seg.id, start=seg.start, end=seg.end, original_text=seg.text)
    unchanged = SegmentRefinement(**base, refined_text=seg.text, changed=False,
                                  status=RefinementStatus.UNCHANGED)

    if not seg.text.strip():
        return unchanged
    if proposal is None:
        return _fallback(base, ["segment missing from model output"], None)
    if duplicated:
        return _fallback(base, ["segment returned more than once by the model"], proposal.refined_text)

    proposed = proposal.refined_text.strip()
    if proposed == seg.text.strip():
        return unchanged

    issues = []
    if _squash(proposal.original_text) != _squash(seg.text):
        issues.append("model misquoted the original text")
    pairs = [(c.original_span, c.refined_span) for c in proposal.changes]
    issues += check_refinement(seg.text, proposed, pairs)
    if issues:
        return _fallback(base, issues, proposed)

    changes = [
        SpanChange(original_span=c.original_span, refined_span=c.refined_span, reason=c.reason)
        for c in proposal.changes if c.original_span.strip()
    ] or _diff_changes(seg.text, proposed)
    return SegmentRefinement(**base, refined_text=proposed, changed=True,
                             status=RefinementStatus.REFINED, changes=changes)


def _fallback(base: dict, issues: list[str], rejected: str | None) -> SegmentRefinement:
    log.info("segment %s: refinement rejected: %s", base["segment_id"], "; ".join(issues))
    return SegmentRefinement(**base, refined_text=base["original_text"], changed=False,
                             status=RefinementStatus.FALLBACK, issues=issues,
                             rejected_text=rejected)


def _diff_changes(raw: str, refined: str) -> list[SpanChange]:
    """Word-level diff, used when the model edited text but listed no changes."""
    a, b = raw.split(), refined.split()
    changes = []
    for op, a0, a1, b0, b1 in SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if op != "equal" and a0 < a1:
            changes.append(SpanChange(original_span=" ".join(a[a0:a1]),
                                      refined_span=" ".join(b[b0:b1]),
                                      reason="not itemised by the model; derived from diff"))
    return changes


def _squash(text: str) -> str:
    text = text.translate(str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"'}))
    return re.sub(r"\s+", " ", text).strip()


# ---------------------------------------------------------------------------
# Gemini provider
# ---------------------------------------------------------------------------

THINKING_LEVELS = ("minimal", "low", "medium", "high")
# Finish reasons that mean the response was withheld rather than completed
_BLOCKED_FINISH = {"SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII", "LANGUAGE", "OTHER"}
# Errors worth trying the next fallback model for
_OVERLOADED = {429, 503}


class GeminiRefineProvider:
    name = "gemini"

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        fallback_models: tuple[str, ...] = (),
        thinking_level: str = "low",
        timeout: float = 300,
        retry_attempts: int = 4,
        max_output_tokens: int = 32768,
        client: Any = None,
    ) -> None:
        self.model = model
        self.fallback_models = fallback_models
        self.thinking_level = thinking_level
        self.max_output_tokens = max_output_tokens
        self.models_used: list[str] = []
        if client is None:
            from google import genai
            from google.genai import types

            client = genai.Client(
                api_key=api_key,
                http_options=types.HttpOptions(
                    timeout=int(timeout * 1000),  # milliseconds
                    # SDK retries 408/429/5xx with exponential backoff
                    retry_options=types.HttpRetryOptions(attempts=retry_attempts),
                ),
            )
        self._client = client

    @classmethod
    def from_settings(cls, settings: Settings, *, client: Any = None) -> GeminiRefineProvider:
        key = settings.gemini_api_key.get_secret_value().strip() if settings.gemini_api_key else ""
        if not key:
            raise RefineError(
                "Transcript refinement is not configured: GEMINI_API_KEY is missing. "
                "Add a Gemini API key (from aistudio.google.com) to your .env file.",
                code="missing_api_key",
            )
        if not settings.refine_model.strip():
            raise RefineError(
                "Transcript refinement is not configured: REFINE_MODEL is empty.", code="config_error"
            )
        if settings.refine_thinking_level not in THINKING_LEVELS:
            raise RefineError(
                f"REFINE_THINKING_LEVEL must be one of {', '.join(THINKING_LEVELS)}.",
                code="config_error",
            )
        fallbacks = tuple(m.strip() for m in settings.refine_fallback_models.split(",") if m.strip())
        return cls(
            api_key=key,
            model=settings.refine_model.strip(),
            fallback_models=fallbacks,
            thinking_level=settings.refine_thinking_level,
            timeout=settings.refine_timeout_seconds,
            retry_attempts=settings.refine_retry_attempts,
            max_output_tokens=settings.refine_max_output_tokens,
            client=client,
        )

    def complete(self, system: str, context: str | None, request: str) -> ProviderReply:
        from google.genai import errors

        models = (self.model, *self.fallback_models)
        for i, model in enumerate(models):
            try:
                response = self._client.models.generate_content(
                    model=model,
                    contents=[context, request] if context else request,
                    config=self._config(system),
                )
            except errors.APIError as exc:
                if exc.code in _OVERLOADED and i + 1 < len(models):
                    log.warning("Gemini model %s unavailable (HTTP %s); trying %s",
                                model, exc.code, models[i + 1])
                    continue
                raise _api_error(exc, model) from exc
            except Exception as exc:
                if not _is_transport_error(exc):
                    raise
                timed_out = "Timeout" in type(exc).__name__
                raise RefineError(
                    "The refinement service timed out." if timed_out else
                    "Could not reach the refinement service. Check the network connection.",
                    code="timeout" if timed_out else "network_error",
                    retryable=True,
                ) from exc
            if model not in self.models_used:
                self.models_used.append(model)
            return _to_reply(response)
        raise AssertionError("unreachable")

    def _config(self, system: str) -> Any:
        from google.genai import types

        # Temperature stays at the default: Google recommends 1.0 for Gemini 3
        # models (lower values can cause looping). Faithfulness comes from the
        # schema-constrained output, the prompt and the deterministic guards.
        return types.GenerateContentConfig(
            system_instruction=system,
            response_mime_type="application/json",
            response_json_schema=OUTPUT_SCHEMA,
            max_output_tokens=self.max_output_tokens,
            thinking_config=types.ThinkingConfig(thinking_level=self.thinking_level),
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )


def _to_reply(response: Any) -> ProviderReply:
    feedback = getattr(response, "prompt_feedback", None)
    if feedback is not None and getattr(feedback, "block_reason", None):
        return ProviderReply("", "blocked")
    candidates = getattr(response, "candidates", None) or []
    if not candidates:
        return ProviderReply("", "blocked")
    candidate = candidates[0]
    finish = getattr(candidate.finish_reason, "name", candidate.finish_reason)
    if finish == "MAX_TOKENS":
        return ProviderReply("", "max_tokens")
    if finish in _BLOCKED_FINISH:
        return ProviderReply("", "blocked")
    content = getattr(candidate, "content", None)
    parts = getattr(content, "parts", None) or []
    text = "".join(p.text for p in parts if getattr(p, "text", None) and not getattr(p, "thought", False))
    return ProviderReply(text, "complete")


def _api_error(exc: Any, model: str) -> RefineError:
    # Never include exc.message in user-facing text: keep provider details in logs.
    code, status = exc.code, (exc.status or "")
    log.warning("Gemini returned HTTP %s %s for model %s", code, status, model)
    if code in (401, 403) or _mentions_invalid_key(exc):
        return RefineError("The Gemini API key was rejected. Check GEMINI_API_KEY.", code="auth_failed")
    if code == 404:
        return RefineError(f"Refinement model '{model}' was not found. Check REFINE_MODEL.",
                           code="model_not_found")
    if code == 429:
        return RefineError(
            "The Gemini free-tier rate limit or quota was reached. Please try again shortly.",
            code="rate_limited", retryable=True)
    if code == 413:
        return RefineError("The transcript was too large for a single refinement request.",
                           code="request_too_large")
    if code >= 500:
        return RefineError("The refinement service is temporarily unavailable. Please try again.",
                           code="provider_error", retryable=True)
    return RefineError(f"The refinement service rejected the request (HTTP {code}).",
                       code="provider_rejected")


def _mentions_invalid_key(exc: Any) -> bool:
    message = str(getattr(exc, "message", "") or "").lower()
    status = str(getattr(exc, "status", "") or "").upper()
    return "api key not valid" in message or "API_KEY_INVALID" in status or "api_key_invalid" in message


def _is_transport_error(exc: Exception) -> bool:
    """Connection/timeout errors from httpx or httpx2 (the SDK may use either)."""
    mro = type(exc).__mro__
    modules = {cls.__module__.split(".")[0] for cls in mro}
    names = {cls.__name__ for cls in mro}
    return bool(modules & {"httpx", "httpx2", "httpcore", "httpcore2"}) and bool(
        names & {"TransportError", "TimeoutException", "NetworkError", "ConnectError"}
    )
