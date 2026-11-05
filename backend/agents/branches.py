"""Branches: rule-based inspectors that filter the root model's free-form reply.

Bough layout.  The root model answers undisturbed (one plain-text call, no schema).  Every branch
below reads the SAME draft and reports checks; none of them calls the model, so they cannot be talked
into anything by the draft.  The Decision step (bottom of this file) turns the reports into one of:

* publish            - every check passed
* publish_with_fixes - only deterministic fixes were needed (forbidden terms, length limits, secrets)
* repair_then_publish- the model is asked ONCE to revise its draft against the failed checks; the
                       repaired text is inspected again and used only if it fails fewer checks
* fallback_notice    - the draft is unusable (empty) and nothing could be repaired

Branch names: constraints | completeness | directness | grounding | safety | size
Severity:     fix    - a Python fix exists (applied in the decision step)
              repair - needs the model to rewrite
              block  - draft cannot be published at all
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..schemas import Constraint, Observation, jsonable
from .language import apply_constraints

# ---------------------------------------------------------------- patterns
FILLER_OPENERS = re.compile(
    r"^\s*(?:(?:great|good|excellent|awesome|interesting)\s+question[!.:,]?"
    r"|(?:certainly|sure|absolutely|of course|definitely)[!.:,]"
    r"|i(?:'d| would) be (?:happy|glad) to"
    r"|thank(?:s| you) for (?:your|the) (?:question|request|message))",
    re.I,
)
PLACEHOLDERS = re.compile(
    r"(?:^\s*(?:#|//|--|/\*)?\s*(?:\.\.\.|…)\s*(?:\*/)?\s*$"            # a line that is only ...
    r"|\bTODO\b"
    r"|your (?:code|logic|implementation) (?:goes )?here"
    r"|rest of (?:the|your) (?:code|file|function|implementation)"
    r"|remaining (?:code|implementation)"
    r"|(?:left|leave) (?:this |that )?as an exercise"
    r"|implement(?:ation)? (?:goes|left) here"
    r"|fill (?:this|that|these) in"
    r"|# \.\.\. (?:other|more|etc)"
    r"|\[(?:insert|add|your) [^\]]{1,40}\])",
    re.I | re.M,
)
CODE_REQUEST = re.compile(
    r"\b(?:write|implement|create|make|build|code|generate|give me|show me|fix|refactor|convert|port)\b.{0,80}?"
    r"\b(?:function|script|program|class|module|code|snippet|component|endpoint|api|query|regex|loop|algorithm|test)s?\b",
    re.I | re.S,
)
TOOL_CLAIMS = re.compile(
    r"\bI(?:'ve| have)? (?:just )?(?:searched|browsed|googled|looked (?:it |this )?up|visited|opened|checked)"
    r"\b(?: the)? (?:web|internet|online|site|page|docs?|documentation|url|link)",
    re.I,
)
MEMORY_CLAIMS = re.compile(
    r"(?:as you(?:'ve| have)? (?:told|mentioned|said)(?: to)? me(?: before| earlier)?"
    r"|from (?:your|my) (?:saved )?memor(?:y|ies)"
    r"|(?:according to|based on) (?:your|my) (?:saved )?memor(?:y|ies)"
    r"|I remember (?:that )?you)",
    re.I,
)
SECRET_PATTERNS = [
    ("groq_key", re.compile(r"\bgsk_[A-Za-z0-9]{20,}\b")),
    ("openai_style_key", re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{20,}\b")),
    ("github_token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b")),
    ("aws_access_key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b")),
    ("supabase_secret_key", re.compile(r"\bsb_secret_[A-Za-z0-9_-]{10,}\b")),
]
FENCE = re.compile(r"^```", re.M)
QUESTION_LINE = re.compile(r"\?\s*$")


@dataclass
class Check:
    branch: str
    name: str
    ok: bool
    severity: str = "info"            # info | fix | repair | block
    detail: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return jsonable(self)


@dataclass
class BranchReport:
    checks: List[Check] = field(default_factory=list)

    @property
    def failed(self) -> List[Check]:
        return [c for c in self.checks if not c.ok]

    @property
    def passed(self) -> int:
        return sum(1 for c in self.checks if c.ok)

    @property
    def blocked(self) -> bool:
        return any(c.severity == "block" for c in self.failed)

    @property
    def needs_repair(self) -> bool:
        return any(c.severity == "repair" for c in self.failed)

    @property
    def needs_fix(self) -> bool:
        return any(c.severity == "fix" for c in self.failed)

    @property
    def score(self) -> float:
        return round(self.passed / len(self.checks), 3) if self.checks else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "checks": [c.to_dict() for c in self.checks],
            "passed": self.passed,
            "total": len(self.checks),
            "failed": [c.name for c in self.failed],
            "score": self.score,
            "blocked": self.blocked,
            "needs_repair": self.needs_repair,
            "needs_fix": self.needs_fix,
        }


# ---------------------------------------------------------------- helpers
def _strip_code(text: str) -> str:
    """Prose only: fenced blocks removed so code comments don't trip prose rules."""
    return re.sub(r"```.*?```", " ", text, flags=re.S)


def _sentences(text: str) -> List[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+|\n+", text.strip()) if s.strip()]


def _words(text: str) -> int:
    return len(text.split())


def asks_for_code(obs: Observation) -> bool:
    return obs.domain == "code" and bool(CODE_REQUEST.search(obs.request))


def redact_secrets(text: str) -> tuple[str, List[str]]:
    hits: List[str] = []
    for name, pat in SECRET_PATTERNS:
        if pat.search(text):
            hits.append(name)
            text = pat.sub("[REDACTED]", text)
    return text, hits


# ---------------------------------------------------------------- branches
class Branches:
    def __init__(self, max_chars: int = 16000):
        self.max_chars = max_chars

    def inspect(self, obs: Observation, draft: str, *, user_memories: int, tool_result: Optional[Dict[str, Any]],
                round_no: int = 1) -> BranchReport:
        r = BranchReport()
        text = draft or ""
        prose = _strip_code(text)
        stripped = text.strip()

        # size ----------------------------------------------------------------
        r.checks.append(Check("size", "non_empty", bool(stripped), "block", "" if stripped else "draft is empty"))
        if not stripped:
            return r
        over = len(text) > self.max_chars
        r.checks.append(Check("size", "within_max_chars", not over, "fix",
                              f"{len(text)} chars > {self.max_chars}" if over else ""))

        # constraints ---------------------------------------------------------
        for c in obs.constraints:
            r.checks.append(self._constraint_check(c, text, prose))

        # completeness (code) -------------------------------------------------
        if obs.domain == "code":
            m = PLACEHOLDERS.search(text)
            r.checks.append(Check("completeness", "no_placeholders", m is None, "repair",
                                  f"placeholder found: “{m.group(0).strip()[:60]}”" if m else ""))
            fences = len(FENCE.findall(text))
            r.checks.append(Check("completeness", "fences_closed", fences % 2 == 0, "repair",
                                  "unclosed ``` block" if fences % 2 else ""))
            if asks_for_code(obs):
                r.checks.append(Check("completeness", "code_block_present", fences >= 2, "repair",
                                      "code was requested but the reply has no fenced code block" if fences < 2 else ""))

        # directness ----------------------------------------------------------
        m = FILLER_OPENERS.match(prose)
        r.checks.append(Check("directness", "no_filler_opener", m is None, "repair",
                              f"opens with “{m.group(0).strip()}”" if m else ""))
        sents = _sentences(prose)
        q_lines = [s for s in sents if QUESTION_LINE.search(s)]
        question_only = bool(sents) and len(q_lines) == len(sents)
        if obs.underspecified:
            r.checks.append(Check("directness", "answer_or_valid_question", True, "info",
                                  "request was underspecified; a question is acceptable"))
        else:
            trailing_q = bool(sents) and QUESTION_LINE.search(sents[-1]) is not None and _words(prose) < 60
            bad = question_only or trailing_q
            r.checks.append(Check("directness", "answers_instead_of_asking", not bad, "repair",
                                  ("reply is only questions" if question_only else
                                   "short reply that ends by asking the user instead of answering") if bad else ""))

        # grounding -----------------------------------------------------------
        m = TOOL_CLAIMS.search(prose)
        tool_ok = bool(tool_result and tool_result.get("ok"))
        r.checks.append(Check("grounding", "no_fake_tool_use", not (m and not tool_ok), "repair",
                              f"claims “{m.group(0)}” but no page was fetched" if (m and not tool_ok) else ""))
        m = MEMORY_CLAIMS.search(prose)
        r.checks.append(Check("grounding", "no_fake_memory", not (m and user_memories == 0), "repair",
                              f"claims “{m.group(0)}” but no user memory was retrieved" if (m and user_memories == 0) else ""))

        # safety --------------------------------------------------------------
        _, hits = redact_secrets(text)
        r.checks.append(Check("safety", "no_secret_like_tokens", not hits, "fix",
                              "secret-like token(s): " + ", ".join(hits) if hits else ""))
        return r

    @staticmethod
    def _constraint_check(c: Constraint, text: str, prose: str) -> Check:
        if c.type == "forbid_term":
            hit = re.search(rf"\b{re.escape(str(c.value))}\b", text, re.I) is not None
            return Check("constraints", f"forbid_term:{c.value}", not hit, "fix",
                         f"contains forbidden term “{c.value}”" if hit else "")
        if c.type == "max_words":
            n = _words(text)
            over = n > int(c.value) > 0
            return Check("constraints", f"max_words:{c.value}", not over, "fix", f"{n} words" if over else "")
        if c.type == "max_sentences":
            n = len(_sentences(prose))
            over = n > int(c.value) > 0
            return Check("constraints", f"max_sentences:{c.value}", not over, "fix", f"{n} sentences" if over else "")
        if c.type == "format":
            want = str(c.value).lower()
            has_bullets = re.search(r"^\s*(?:[-*•]|\d+[.)])\s+", prose, re.M) is not None
            if "bullet" in want or "list" in want:
                return Check("constraints", f"format:{c.value}", has_bullets, "repair",
                             "" if has_bullets else "bullet list requested, none present")
            if "no bullet" in want or "paragraph" in want or "prose" in want:
                return Check("constraints", f"format:{c.value}", not has_bullets, "repair",
                             "bullets present but prose requested" if has_bullets else "")
        return Check("constraints", f"{c.type}:{c.value}", True, "info", "not machine-checkable")


# ---------------------------------------------------------------- deterministic fixes
def apply_fixes(text: str, obs: Observation, max_chars: int) -> tuple[str, List[str]]:
    text, fixes = apply_constraints(text, obs.constraints)
    text, hits = redact_secrets(text)
    if hits:
        fixes.append("redacted_secret_like_tokens:" + ",".join(hits))
    if len(text) > max_chars:
        text = text[: max_chars - 1].rstrip() + "…"
        fixes.append(f"clipped_to_max_chars:{max_chars}")
    return text, fixes


def classify(text: str, obs: Observation, tool_result: Optional[Dict[str, Any]]) -> str:
    """Action kind of the final text, for the UI and feedback stats."""
    if tool_result:
        return "web_fetch"
    sents = _sentences(_strip_code(text))
    if sents and all(QUESTION_LINE.search(s) for s in sents):
        return "clarify"
    return "answer"
