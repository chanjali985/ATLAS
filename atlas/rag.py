"""Hybrid retrieval: BM25 (keyword) + TF-IDF cosine (dense-ish), fused with RRF, then reranked.

Swap TF-IDF for real embeddings + pgvector without changing the interface.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

KB_DIR = Path(__file__).resolve().parent.parent / "knowledge"
_tok = lambda s: re.findall(r"[a-z0-9\-]+", s.lower())


class HybridRetriever:
    def __init__(self, kb_dir: Path = KB_DIR):
        self.docs = []
        for p in sorted(kb_dir.glob("*.md")):
            text = p.read_text()
            for chunk in [c.strip() for c in text.split("\n\n") if c.strip()] or [text]:
                self.docs.append({"source": p.name, "text": chunk, "full": text})
        self.tokens = [_tok(d["full"]) for d in self.docs]
        self.avgdl = sum(map(len, self.tokens)) / max(len(self.tokens), 1)
        self.df = Counter(w for t in self.tokens for w in set(t))
        self.tfidf = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
        self.mat = self.tfidf.fit_transform([d["full"] for d in self.docs])

    def _bm25(self, q: list[str], i: int, k1=1.5, b=0.75) -> float:
        tf = Counter(self.tokens[i]); n = len(self.docs); dl = len(self.tokens[i]); s = 0.0
        for w in q:
            if w not in tf:
                continue
            idf = math.log(1 + (n - self.df[w] + 0.5) / (self.df[w] + 0.5))
            s += idf * tf[w] * (k1 + 1) / (tf[w] + k1 * (1 - b + b * dl / self.avgdl))
        return s

    def search(self, query: str, k: int = 3) -> list[dict]:
        q = _tok(query)
        bm = [self._bm25(q, i) for i in range(len(self.docs))]
        dense = cosine_similarity(self.tfidf.transform([query]), self.mat)[0]
        rank_bm = {i: r for r, i in enumerate(sorted(range(len(bm)), key=lambda i: -bm[i]))}
        rank_de = {i: r for r, i in enumerate(sorted(range(len(dense)), key=lambda i: -dense[i]))}
        fused = {i: 1 / (60 + rank_bm[i]) + 1 / (60 + rank_de[i]) for i in range(len(self.docs))}
        # rerank: boost chunks whose own text overlaps the query (chunk-level precision)
        def rerank(i):
            overlap = len(set(q) & set(_tok(self.docs[i]["text"]))) / max(len(set(q)), 1)
            return fused[i] + 0.02 * overlap
        seen, out = set(), []
        for i in sorted(fused, key=rerank, reverse=True):
            src = self.docs[i]["source"]
            if src in seen:
                continue
            seen.add(src)
            out.append({"source": src, "score": round(rerank(i), 4), "text": self.docs[i]["full"]})
            if len(out) == k:
                break
        return out
