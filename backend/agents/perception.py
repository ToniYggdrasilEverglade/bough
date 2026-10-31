"""Perception: turns raw input into a structured Observation using plain rules (no model call).

The original text is preserved in `raw_text`. Modalities are pluggable: register a Perceiver for
"image", "audio" or "sensor" later without touching the rest of the pipeline.
"""
from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Dict, List

from ..config import Settings
from ..errors import InputRejected
from ..memory.text import STOPWORDS, tokenize
from ..schemas import Constraint, Entity, Observation, RawInput

_SAVE = re.compile(
    r"^\s*(?:please\s+)?(?:remember(?:\s+that)?|(?:save|store)\s+(?:this|that)?\s*(?:to|in)\s+"
    r"(?:my\s+|your\s+)?(?:long[- ]term\s+)?memory)\s*[:,\-]?\s*(.*)$", re.I | re.S)
_PREF = re.compile(r"\b(prefer|prefers|favou?rite|like to|always|never|i like|i love|i hate|i dislike)\b", re.I)
_QUESTION_START = re.compile(
    r"^\s*(what|who|when|where|why|how|which|can|could|should|would|is|are|do|does|did|will)\b", re.I)
_REQUEST_START = re.compile(
    r"^\s*(please\s+)?(write|summari[sz]e|translate|explain|describe|list|give|tell|compare|draft|create|help|show)\b", re.I)
_ANAPHORA = re.compile(
    r"^\s*(?:please\s+)?(?:do|fix|summari[sz]e|explain|translate|continue|rewrite|check|improve|review|finish|"
    r"answer|edit|shorten|expand|analy[sz]e)\s+(?:it|this|that|them|these|those|the same)\s*[.!?]*\s*$", re.I)

_MAX_WORDS = re.compile(r"\b(?:in|under|within|less than|at most|no more than|max(?:imum)?(?: of)?)\s+(\d{1,4})\s+words?\b|"
                        r"\b(\d{1,4})\s+words?\s+or\s+(?:fewer|less)\b", re.I)
_MAX_SENT = re.compile(r"\b(?:in|under|within|at most|no more than|max(?:imum)?(?: of)?)\s+(\d{1,2})\s+sentences?\b", re.I)
_FORBID = re.compile(r"\b(?:don'?t|do not|never)\s+(?:mention|say|use|include)\s+[\"']?([A-Za-z][\w-]{1,29})[\"']?", re.I)
_CONCISE = re.compile(r"\b(briefly|be brief|be concise|concise|short answer|tl;?dr)\b", re.I)
_NO_BULLETS = re.compile(r"\b(no|without|avoid)\s+bullet(?:s| points?)\b", re.I)
_BULLETS = re.compile(r"\b(?:in|as|using|with)\s+bullet(?:s| points?)\b", re.I)
_FORMAL = re.compile(r"\b(formal(?:ly)?|formal tone)\b", re.I)
_QUOTED = re.compile(r"[\"“]([^\"”]{1,60})[\"”]")
_NUMBER = re.compile(r"\b\d+(?:\.\d+)?\b")
_PROPER = re.compile(r"\b[A-Z][a-zA-Z]{2,}\b")


class Perceiver(ABC):
    modality: str

    @abstractmethod
    def perceive(self, raw: RawInput) -> Observation: ...


def extract_constraints(text: str) -> List[Constraint]:
    found: List[Constraint] = []

    def add(type_, value, src):
        found.append(Constraint(f"c{len(found) + 1}", type_, value, src.strip()))

    for m in _MAX_WORDS.finditer(text):
        add("max_words", int(m.group(1) or m.group(2)), m.group(0))
    for m in _MAX_SENT.finditer(text):
        add("max_sentences", int(m.group(1)), m.group(0))
    for m in _FORBID.finditer(text):
        add("forbid_term", m.group(1).lower(), m.group(0))
    m = _CONCISE.search(text)
    if m:
        add("style", "concise", m.group(0))
    m = _FORMAL.search(text)
    if m:
        add("style", "formal", m.group(0))
    m = _NO_BULLETS.search(text)
    if m:
        add("format", "no_bullets", m.group(0))
    elif (m := _BULLETS.search(text)):
        add("format", "bullets", m.group(0))
    return found[:10]


def extract_entities(text: str, limit: int = 10) -> List[Entity]:
    ents: List[Entity] = []
    seen = set()

    def add(t, ty):
        if t.lower() not in seen and len(ents) < limit:
            seen.add(t.lower())
            ents.append(Entity(t, ty))

    for m in _QUOTED.finditer(text):
        add(m.group(1), "quoted")
    for m in _NUMBER.finditer(text):
        add(m.group(0), "number")
    for m in _PROPER.finditer(text):
        if m.group(0).lower() not in STOPWORDS and m.start() != 0:
            add(m.group(0), "proper_noun")
    return ents


class TextPerceiver(Perceiver):
    modality = "text"

    def perceive(self, raw: RawInput) -> Observation:
        text = raw.data
        norm = " ".join(text.split())
        save = _SAVE.match(norm)
        if save:
            body = save.group(1).strip()
            return Observation(
                raw_text=text, request=body or norm, intent="save_memory", entities=extract_entities(body),
                constraints=[], underspecified=not body, missing=[] if body else ["what you want me to remember"],
                save_text=body or None, save_kind=("preference" if _PREF.search(body) else "fact") if body else None)
        intent = "question" if (norm.endswith("?") or _QUESTION_START.match(norm)) else \
                 "request" if _REQUEST_START.match(norm) else "statement"
        missing: List[str] = []
        if _ANAPHORA.match(norm):
            missing.append("what “it/this/that” refers to")
        elif not tokenize(norm):
            missing.append("what you would like me to do")
        return Observation(raw_text=text, request=norm, intent=intent, entities=extract_entities(norm),
                           constraints=extract_constraints(norm), underspecified=bool(missing), missing=missing)


class Perception:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._by_modality: Dict[str, Perceiver] = {}
        self.register(TextPerceiver())

    def register(self, p: Perceiver) -> None:
        self._by_modality[p.modality] = p

    def perceive(self, raw: RawInput) -> Observation:
        p = self._by_modality.get(raw.modality)
        if p is None:
            raise InputRejected(f"modality {raw.modality!r} is not supported yet")
        return p.perceive(raw)
