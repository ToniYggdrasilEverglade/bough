"""Naive tokenisation used for memory search and relevance scoring (deliberately simple)."""
from __future__ import annotations

import re
from typing import List, Set

STOPWORDS = frozenset("""
a an the and or but if then so of to in on at by for with from as is are was were be been being am
i me my mine we us our you your he she it its they them their this that these those there here
do does did doing done have has had having will would can could should may might must shall
what which who whom whose when where why how not no yes just also very really about into over
please tell give show let get got make made say said any some more most other such than too
""".split())

_SUFFIXES = ("ences", "ence", "ings", "ing", "ed", "es", "s")
_WORD = re.compile(r"[a-z0-9']+")


def stem(word: str) -> str:
    for suf in _SUFFIXES:
        if word.endswith(suf) and len(word) - len(suf) >= 3:
            return word[: -len(suf)]
    return word


def tokenize(text: str) -> List[str]:
    out = []
    for w in _WORD.findall(text.lower()):
        w = w.strip("'")
        if not w or w in STOPWORDS:
            continue
        out.append(stem(w))
    return out


def token_set(text: str) -> Set[str]:
    return set(tokenize(text))
