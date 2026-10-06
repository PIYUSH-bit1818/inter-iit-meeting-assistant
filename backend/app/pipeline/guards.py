"""Deterministic checks that keep LLM output faithful to the recording.

- Refinement edits: reject changes to numbers, negation, unseen names, or
  spans that do not occur in the source segment.
- Meeting record: drop items whose evidence does not match the transcript and
  null out owners/deadlines not present in the cited segments.

Implemented in a later phase.
"""
