"""Shared Google Gemini client for the LLM stages.

Both LLM stages (transcript refinement and meeting documentation) call Gemini
the same way: ``generateContent`` with a strict JSON schema, a system prompt,
retries with backoff, optional fallback models when the primary is overloaded,
and error mapping to user-facing messages that never contain provider text or
the API key. Each stage passes its own model, schema, settings name and error
class, so the stages stay separate and independently configurable.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from pydantic import SecretStr

log = logging.getLogger(__name__)

THINKING_LEVELS = ("minimal", "low", "medium", "high")
# Finish reasons that mean the response was withheld rather than completed
_BLOCKED_FINISH = {"SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII", "LANGUAGE", "OTHER"}
# Errors worth trying the next fallback model for
_OVERLOADED = {429, 503}


class LLMError(Exception):
    """An LLM stage failed. ``message`` is user-facing; ``code`` is stable."""

    def __init__(self, message: str, code: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.retryable = retryable


@dataclass(frozen=True)
class ProviderReply:
    text: str
    stop_reason: str | None  # "complete" | "max_tokens" | "blocked"


@dataclass(frozen=True)
class StageLabels:
    """Names used in configuration checks and user-facing messages."""

    stage: str  # e.g. "refinement"
    model_setting: str  # e.g. "REFINE_MODEL"
    thinking_setting: str  # e.g. "REFINE_THINKING_LEVEL"


def check_config(
    api_key: SecretStr | None,
    model: str,
    thinking_level: str,
    labels: StageLabels,
    error_cls: type[LLMError],
) -> str:
    """Validate stage configuration; return the API key value."""
    key = api_key.get_secret_value().strip() if api_key else ""
    if not key:
        raise error_cls(
            f"{labels.stage.capitalize()} is not configured: GEMINI_API_KEY is missing. "
            "Add a Gemini API key (from aistudio.google.com) to your .env file.",
            code="missing_api_key",
        )
    if not model.strip():
        raise error_cls(
            f"{labels.stage.capitalize()} is not configured: {labels.model_setting} is empty.",
            code="config_error",
        )
    if thinking_level not in THINKING_LEVELS:
        raise error_cls(
            f"{labels.thinking_setting} must be one of {', '.join(THINKING_LEVELS)}.",
            code="config_error",
        )
    return key


def parse_model_list(value: str) -> tuple[str, ...]:
    return tuple(m.strip() for m in value.split(",") if m.strip())


class GeminiJsonModel:
    """Calls Gemini with a fixed JSON schema and maps failures to ``error_cls``."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        schema: dict[str, Any],
        labels: StageLabels,
        error_cls: type[LLMError],
        fallback_models: tuple[str, ...] = (),
        thinking_level: str = "low",
        timeout: float = 300,
        retry_attempts: int = 4,
        max_output_tokens: int = 32768,
        client: Any = None,
    ) -> None:
        self.model = model
        self.schema = schema
        self.labels = labels
        self.error_cls = error_cls
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
        self.client = client

    def generate(self, system: str, contents: Any) -> ProviderReply:
        from google.genai import errors

        models = (self.model, *self.fallback_models)
        for i, model in enumerate(models):
            try:
                response = self.client.models.generate_content(
                    model=model, contents=contents, config=self._config(system)
                )
            except errors.APIError as exc:
                if exc.code in _OVERLOADED and i + 1 < len(models):
                    log.warning("Gemini model %s unavailable (HTTP %s); trying %s",
                                model, exc.code, models[i + 1])
                    continue
                raise self._api_error(exc, model) from exc
            except Exception as exc:
                if not _is_transport_error(exc):
                    raise
                timed_out = "Timeout" in type(exc).__name__
                raise self.error_cls(
                    f"The {self.labels.stage} service timed out." if timed_out else
                    f"Could not reach the {self.labels.stage} service. Check the network connection.",
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
        # schema-constrained output, the prompts and the deterministic guards.
        return types.GenerateContentConfig(
            system_instruction=system,
            response_mime_type="application/json",
            response_json_schema=self.schema,
            max_output_tokens=self.max_output_tokens,
            thinking_config=types.ThinkingConfig(thinking_level=self.thinking_level),
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )

    def _api_error(self, exc: Any, model: str) -> LLMError:
        # Never include exc.message in user-facing text: keep provider details in logs.
        code, status = exc.code, (exc.status or "")
        stage, err = self.labels.stage, self.error_cls
        log.warning("Gemini returned HTTP %s %s for model %s (%s)", code, status, model, stage)
        if code in (401, 403) or _mentions_invalid_key(exc):
            return err("The Gemini API key was rejected. Check GEMINI_API_KEY.", code="auth_failed")
        if code == 404:
            return err(f"{stage.capitalize()} model '{model}' was not found. "
                       f"Check {self.labels.model_setting}.", code="model_not_found")
        if code == 429:
            return err("The Gemini free-tier rate limit or quota was reached. Please try again shortly.",
                       code="rate_limited", retryable=True)
        if code == 413:
            return err(f"The transcript was too large for a single {stage} request.",
                       code="request_too_large")
        if code >= 500:
            return err(f"The {stage} service is temporarily unavailable. Please try again.",
                       code="provider_error", retryable=True)
        return err(f"The {stage} service rejected the request (HTTP {code}).", code="provider_rejected")


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
