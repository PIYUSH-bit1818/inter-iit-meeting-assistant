"""Stage 2: domain-aware transcript refinement (LLM A). Implemented in a later phase."""

from ..schemas import RawTranscript, RefinedTranscript


def refine(raw: RawTranscript, glossary: list[str] | None = None) -> RefinedTranscript:
    raise NotImplementedError
