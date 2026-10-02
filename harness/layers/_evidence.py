"""Shared evidence helpers for `critic` and `citation_checker`.

Everything here mirrors how `arena/scorer.py` judges a claim, so a layer
decides with the SAME yardstick the scorer uses:

* text is compared after `norm` (NFC, casefold, whitespace collapsed) —
  exactly `arena.scorer._norm`; a raw `in` test would delete claims the
  scorer would have credited (a real model changes case or spacing);
* a claim is supported only by ONE LINE of a document (`_supports`);
* claims shorter than `MIN_SUPPORT_CHARS` can never be SUPPORTED, longer
  than `MAX_CLAIM_CHARS` are OVERLONG, more than `MAX_CLAIMS_PER_DOC` on one
  document are REDUNDANT.

Nothing here ever produces claim text: helpers only answer questions and
return SUBSTRINGS of what the model wrote (trimming is allowed, rewriting
is not — README §8.2).
"""

from __future__ import annotations

import re
import unicodedata

#: Mirrors of the scorer's limits (arena/scorer.py).
MIN_SUPPORT_CHARS = 12
MAX_CLAIM_CHARS = 500
MAX_CLAIMS_PER_DOC = 4

#: A trimmed claim must still say something: shorter fragments are too
#: generic to cover a required fact and would only risk `IRRELEVANT`.
MIN_TRIM_WORDS = 6
MIN_TRIM_CHARS = 30

#: Words of a claim considered when trimming; bounds the window search.
MAX_TRIM_TOKENS = 240

#: Characters that may be peeled off the ends of a quotation (a model adds
#: a closing period or wraps the quote in quotation marks). Peeling keeps
#: the result a substring of the model's own text.
EDGE_CHARS = " \t\r\n.,;:!?\"'`“”‘’«»()[]{}…-–—*"

_WS_RE = re.compile(r"\s+")
_TOKEN_RE = re.compile(r"\S+")


def norm(text) -> str:
    """`arena.scorer._norm`: NFC, casefold, whitespace collapsed, stripped."""
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    return _WS_RE.sub(" ", unicodedata.normalize("NFC", text).casefold()).strip()


class Evidence:
    """What this run actually saw, indexed for claim checks."""

    def __init__(self, ctx) -> None:
        raw = ctx.observed_text or ""
        self.observed = norm(raw)
        corpus = getattr(ctx, "corpus", None)
        docs = getattr(corpus, "docs", None) if corpus is not None else None
        docs = [d for d in docs if getattr(d, "doc_id", None)] if isinstance(docs, list) else []
        self.has_corpus = bool(docs)
        self._by_id = {d.doc_id: d for d in docs}
        # Ranked candidate sources: documents that came back WHOLE from a
        # clean fetch first, then documents only seen as search hits (the
        # scorer still counts those as retrieved). Never an unseen doc —
        # citing one is `UNRETRIEVED`.
        self.fetched = [d for d in docs if isinstance(d.body, str) and d.body and d.body in raw]
        fetched_ids = {d.doc_id for d in self.fetched}
        self.mentioned = [d for d in docs if d.doc_id not in fetched_ids and d.doc_id in raw]
        self._lines: dict = {}

    # -- questions ----------------------------------------------------

    def seen(self, normalised: str) -> bool:
        return bool(normalised) and normalised in self.observed

    def get(self, doc_id):
        return self._by_id.get(doc_id.strip()) if isinstance(doc_id, str) else None

    def is_retrieved(self, doc) -> bool:
        return doc is not None and (doc in self.fetched or doc in self.mentioned)

    def lines(self, doc) -> tuple:
        cached = self._lines.get(doc.doc_id)
        if cached is None:
            body = doc.body if isinstance(doc.body, str) else ""
            cached = tuple(line for line in (norm(raw) for raw in body.splitlines()) if line)
            self._lines[doc.doc_id] = cached
        return cached

    def supports(self, doc, normalised: str) -> bool:
        if doc is None or len(normalised) < MIN_SUPPORT_CHARS:
            return False
        return any(normalised in line for line in self.lines(doc))

    def source(self, normalised: str, prefer=None):
        """doc_id of a RETRIEVED document with one line containing the text."""
        if len(normalised) < MIN_SUPPORT_CHARS:
            return None
        if prefer is not None and self.is_retrieved(prefer) and self.supports(prefer, normalised):
            return prefer.doc_id
        for doc in self.fetched + self.mentioned:
            if self.supports(doc, normalised):
                return doc.doc_id
        return None

    def verifiable(self, normalised: str) -> bool:
        """Would the scorer find this text in a line of a retrieved doc?"""
        if not (MIN_SUPPORT_CHARS <= len(normalised) <= MAX_CLAIM_CHARS):
            return False
        if not self.seen(normalised):
            return False
        if not self.has_corpus:
            return True
        return self.source(normalised) is not None

    # -- trimming -----------------------------------------------------

    def best_trim(self, text: str):
        """Longest verbatim, line-scoped SUBSTRING of `text`, or None.

        Walks word windows from longest to shortest, peels quotation marks
        and punctuation off both ends, and returns `(substring, doc_id)`
        for the first window the run has seen AND a retrieved document
        supports on one line. The substring is sliced out of `text`
        itself, so it is still the model's own writing.
        """
        if not isinstance(text, str) or not text:
            return None
        spans = [(m.start(), m.end()) for m in _TOKEN_RE.finditer(text)][:MAX_TRIM_TOKENS]
        n = len(spans)
        for size in range(n, MIN_TRIM_WORDS - 1, -1):
            for start in range(0, n - size + 1):
                if spans[start + size - 1][1] - spans[start][0] > MAX_CLAIM_CHARS + 50:
                    continue
                piece = text[spans[start][0]: spans[start + size - 1][1]].strip(EDGE_CHARS)
                normalised = norm(piece)
                if not (MIN_TRIM_CHARS <= len(normalised) <= MAX_CLAIM_CHARS):
                    continue
                if not self.seen(normalised):
                    continue
                if not self.has_corpus:
                    return piece, None
                doc_id = self.source(normalised)
                if doc_id is not None:
                    return piece, doc_id
        return None
