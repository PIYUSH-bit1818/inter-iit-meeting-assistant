"""Stage 3: meeting documentation (LLM B, separate from refinement).

Consumes only the ``RefinedTranscript`` (segment ids, timestamps, refined
text) and produces a ``MeetingRecord``: summary, minutes, decisions,
proposals (raised but not agreed) and action items.

The model returns strict, schema-constrained JSON in which every decision,
proposal and action item cites segment ids and an exact quote. Each item is
then validated deterministically (``guards``); anything unsupported is moved
to ``rejected_items`` with its reasons instead of being shown as fact. Items
are never rewritten.

Gemini call mechanics are shared with refinement (``gemini.py``), but this
stage uses its own model, prompt, schema and settings.
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, ValidationError

from ..config import REPO_ROOT, Settings, get_settings
from ..schemas import (
    ActionItem,
    Decision,
    MeetingMinute,
    MeetingRecord,
    Proposal,
    RecordItemKind,
    RefinedTranscript,
    RejectedItem,
)
from .gemini import (
    GeminiJsonModel,
    LLMError,
    ProviderReply,
    StageLabels,
    check_config,
    parse_model_list,
)
from .guards import (
    check_deadline,
    check_evidence,
    check_facts,
    check_negation,
    check_owner,
    has_agreement,
    has_commitment,
)

log = logging.getLogger(__name__)

PROMPT_VERSION = "meeting_documentation_v1"
PROMPT_PATH = REPO_ROOT / "prompts" / f"{PROMPT_VERSION}.txt"
INVALID_OUTPUT_RETRIES = 1


class DocumentError(LLMError):
    """Documentation failed. ``message`` is user-facing; ``code`` is stable."""


@lru_cache
def load_prompt() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Model output contract
# ---------------------------------------------------------------------------


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class OutMinute(_Strict):
    topic: str
    discussion: str
    segment_ids: list[int]


class OutDecision(_Strict):
    decision: str
    evidence_segment_ids: list[int]
    evidence_quote: str


class OutProposal(_Strict):
    proposal: str
    evidence_segment_ids: list[int]
    evidence_quote: str


class OutActionItem(_Strict):
    task: str
    owner: str | None
    deadline: str | None
    evidence_segment_ids: list[int]
    evidence_quote: str


class ModelOutput(_Strict):
    summary: str
    minutes: list[OutMinute]
    decisions: list[OutDecision]
    proposals: list[OutProposal]
    action_items: list[OutActionItem]


def _obj(properties: dict[str, Any]) -> dict[str, Any]:
    return {"type": "object", "properties": properties,
            "required": list(properties), "additionalProperties": False}


_IDS = {"type": "array", "items": {"type": "integer"}}
_STR = {"type": "string"}
_NULLABLE_STR = {"type": ["string", "null"]}

OUTPUT_SCHEMA: dict[str, Any] = _obj({
    "summary": _STR,
    "minutes": {"type": "array", "items": _obj({"topic": _STR, "discussion": _STR, "segment_ids": _IDS})},
    "decisions": {"type": "array", "items": _obj(
        {"decision": _STR, "evidence_segment_ids": _IDS, "evidence_quote": _STR})},
    "proposals": {"type": "array", "items": _obj(
        {"proposal": _STR, "evidence_segment_ids": _IDS, "evidence_quote": _STR})},
    "action_items": {"type": "array", "items": _obj({
        "task": _STR, "owner": _NULLABLE_STR, "deadline": _NULLABLE_STR,
        "evidence_segment_ids": _IDS, "evidence_quote": _STR})},
})


# ---------------------------------------------------------------------------
# Provider interface
# ---------------------------------------------------------------------------


class DocumentProvider(Protocol):
    name: str
    model: str

    def complete(self, system: str, request: str) -> ProviderReply:
        ...


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def document(
    refined: RefinedTranscript,
    *,
    provider: DocumentProvider | None = None,
    settings: Settings | None = None,
) -> MeetingRecord:
    settings = settings or get_settings()
    segments = {s.id: s.text for s in refined.segments}
    if not any(t.strip() for t in segments.values()):
        raise DocumentError("The transcript is empty, so there is nothing to document.",
                            code="empty_transcript")
    provider = provider or GeminiDocumentProvider.from_settings(settings)

    request = "Write the meeting record for this transcript.\n\n" + json.dumps(
        {"segments": [{"segment_id": s.id, "start": s.start, "end": s.end, "text": s.text}
                      for s in refined.segments if s.text.strip()]},
        ensure_ascii=False,
    )
    output = _generate(provider, request)
    record = validate_output(output, segments)
    record.document_model = ", ".join(getattr(provider, "models_used", None) or [provider.model])
    log.info("documentation: %d decisions, %d proposals, %d action items, %d minutes, %d rejected",
             len(record.decisions), len(record.proposals), len(record.action_items),
             len(record.minutes), len(record.rejected_items))
    return record


def _generate(provider: DocumentProvider, request: str) -> ModelOutput:
    for attempt in range(INVALID_OUTPUT_RETRIES + 1):
        reply = provider.complete(load_prompt(), request)
        if reply.stop_reason == "blocked":
            raise DocumentError("The documentation model declined to process this transcript.",
                                code="refused")
        if reply.stop_reason == "max_tokens":
            raise DocumentError("The meeting record was too long to generate in one response.",
                                code="output_truncated")
        try:
            return ModelOutput.model_validate_json(reply.text)
        except ValidationError as exc:
            log.warning("invalid documentation output (attempt %d): %s", attempt + 1,
                        exc.errors(include_input=False)[:3])
    raise DocumentError("The documentation model returned output that could not be understood.",
                        code="invalid_response")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_output(output: ModelOutput, segments: dict[int, str]) -> MeetingRecord:
    """Keep only items grounded in ``segments``; reject the rest with reasons."""
    full_text = " ".join(segments[i] for i in sorted(segments))
    rejected: list[RejectedItem] = []

    def reject(kind: RecordItemKind, item: BaseModel, reasons: list[str]) -> None:
        log.info("rejected %s: %s", kind.value, "; ".join(reasons))
        rejected.append(RejectedItem(kind=kind, content=item.model_dump(), reasons=reasons))

    def cited(ids: list[int]) -> str:
        return " ".join(segments[i] for i in sorted(set(ids)) if i in segments)

    # Summary: may not state facts absent from the whole transcript
    summary: str | None = output.summary.strip() or None
    issues = ["summary is empty"] if summary is None else check_facts(summary, full_text)
    if issues:
        log.info("rejected summary: %s", "; ".join(issues))
        rejected.append(RejectedItem(kind=RecordItemKind.SUMMARY,
                                     content={"summary": output.summary}, reasons=issues))
        summary = None

    minutes = []
    for m in output.minutes:
        evidence = cited(m.segment_ids)
        issues = _check_ids(m.segment_ids, segments) or (
            check_facts(m.discussion, evidence) + check_facts(m.topic, evidence, check_names=False))
        built, err = _build(MeetingMinute, m)
        if issues or err:
            reject(RecordItemKind.MINUTE, m, issues + err)
        else:
            minutes.append(built)

    decisions = []
    for d in output.decisions:
        issues = check_evidence(d.evidence_segment_ids, d.evidence_quote, segments)
        if not issues:
            evidence = cited(d.evidence_segment_ids)
            issues = check_facts(d.decision, evidence) + check_negation(d.decision, d.evidence_quote)
            if not has_agreement(evidence):
                issues.append("no explicit agreement in the cited evidence (not a decision)")
        built, err = _build(Decision, d)
        if issues or err:
            reject(RecordItemKind.DECISION, d, issues + err)
        else:
            decisions.append(built)

    proposals = []
    for p in output.proposals:
        issues = check_evidence(p.evidence_segment_ids, p.evidence_quote, segments)
        if not issues:
            issues = check_facts(p.proposal, cited(p.evidence_segment_ids)) + check_negation(
                p.proposal, p.evidence_quote)
        built, err = _build(Proposal, p)
        if issues or err:
            reject(RecordItemKind.PROPOSAL, p, issues + err)
        else:
            proposals.append(built)

    actions = []
    for a in output.action_items:
        built, err = _build(ActionItem, a)  # normalises "Unknown"/"TBD"/"" to None
        issues = check_evidence(a.evidence_segment_ids, a.evidence_quote, segments)
        if not issues and built is not None:
            evidence = cited(a.evidence_segment_ids)
            issues = (check_facts(a.task, evidence)
                      + check_negation(a.task, a.evidence_quote)
                      + check_owner(built.owner, evidence)
                      + check_deadline(built.deadline, evidence))
            if not has_commitment(evidence):
                issues.append("no explicit assignment or commitment in the cited evidence")
        if issues or err:
            reject(RecordItemKind.ACTION_ITEM, a, issues + err)
        else:
            actions.append(built)

    return MeetingRecord(
        summary=summary, minutes=minutes, decisions=decisions, proposals=proposals,
        action_items=actions, rejected_items=rejected, document_model="", prompt_version=PROMPT_VERSION,
    )


def _check_ids(ids: list[int], segments: dict[int, str]) -> list[str]:
    if not ids:
        return ["no segment ids"]
    missing = sorted({i for i in ids if i not in segments})
    return [f"segment ids do not exist: {missing}"] if missing else []


def _build(model: type[BaseModel], item: BaseModel) -> tuple[Any, list[str]]:
    try:
        return model(**item.model_dump()), []
    except ValidationError as exc:
        fields = sorted({".".join(str(p) for p in e["loc"]) for e in exc.errors()})
        return None, [f"invalid or empty fields: {fields}"]


# ---------------------------------------------------------------------------
# Gemini provider
# ---------------------------------------------------------------------------

LABELS = StageLabels(stage="documentation", model_setting="DOCUMENT_MODEL",
                     thinking_setting="DOCUMENT_THINKING_LEVEL")


class GeminiDocumentProvider:
    name = "gemini"

    def __init__(self, llm: GeminiJsonModel) -> None:
        self._llm = llm
        self.model = llm.model

    @property
    def models_used(self) -> list[str]:
        return self._llm.models_used

    @classmethod
    def from_settings(cls, settings: Settings, *, client: Any = None) -> GeminiDocumentProvider:
        key = check_config(settings.gemini_api_key, settings.document_model,
                           settings.document_thinking_level, LABELS, DocumentError)
        if settings.document_model.strip() == settings.refine_model.strip():
            raise DocumentError(
                "DOCUMENT_MODEL must be a different model from REFINE_MODEL so the two "
                "LLM stages stay distinct.",
                code="config_error",
            )
        return cls(GeminiJsonModel(
            api_key=key,
            model=settings.document_model.strip(),
            schema=OUTPUT_SCHEMA,
            labels=LABELS,
            error_cls=DocumentError,
            fallback_models=parse_model_list(settings.document_fallback_models),
            thinking_level=settings.document_thinking_level,
            timeout=settings.document_timeout_seconds,
            retry_attempts=settings.document_retry_attempts,
            max_output_tokens=settings.document_max_output_tokens,
            client=client,
        ))

    def complete(self, system: str, request: str) -> ProviderReply:
        return self._llm.generate(system, request)
