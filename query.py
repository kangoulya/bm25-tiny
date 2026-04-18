#!/usr/bin/env python3
"""
bm25-tiny — query
Pure-Python BM25 retrieval over a JSON index produced by ingest.py.

CLI:
    python query.py "your question" [--k 3]

Library:
    from query import retrieve, format_context
    hits = retrieve("your question", k=5)

Index path (env-overridable):
    BM25_INDEX = ./store/bm25.json
"""
import json
import os
import re
import sys
import unicodedata
from pathlib import Path

INDEX = Path(os.environ.get("BM25_INDEX", "./store/bm25.json"))
K1 = 1.5
B = 0.75

STOPWORDS = set("""
le la les un une des du de d l a au aux et ou ni mais donc car si que qui quoi
dont ou est etre suis es sommes etes sont ete j je tu il elle on nous vous ils
elles me te se lui leur notre votre mon ma mes ton ta tes son sa ses ce cet
cette ces pas plus moins tres bien mal pour par sans avec dans sur sous entre
vers chez comme aussi alors puis deja encore the a an of in on for to is are
was were be been being have has had do does did will would could should may
might can i you he she it we they them us my your his her its our their this
that these those not no yes as at by from but or and if then so
""".split())

_INDEX_CACHE = None


def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def tokenize(text: str):
    text = strip_accents(text.lower())
    toks = re.findall(r"[a-z0-9]{2,}", text)
    return [t for t in toks if t not in STOPWORDS]


def _load():
    global _INDEX_CACHE
    if _INDEX_CACHE is None:
        if not INDEX.exists():
            return None
        _INDEX_CACHE = json.loads(INDEX.read_text(encoding="utf-8"))
    return _INDEX_CACHE


def retrieve(query: str, k: int = 3):
    """Returns list of dicts: [{text, source, chunk, score}]"""
    idx = _load()
    if not idx:
        return []
    q_terms = tokenize(query)
    if not q_terms:
        return []
    avgdl = idx["avgdl"]
    idf = idx["idf"]
    results = []
    for d in idx["docs"]:
        dl = d["length"]
        tf = d["tf"]
        score = 0.0
        for t in q_terms:
            if t not in tf:
                continue
            f = tf[t]
            i = idf.get(t, 0.0)
            denom = f + K1 * (1 - B + B * dl / avgdl)
            score += i * (f * (K1 + 1)) / denom
        if score > 0:
            results.append({
                "text": d["text"],
                "source": d["source"],
                "chunk": d["chunk"],
                "score": score,
            })
    results.sort(key=lambda r: r["score"], reverse=True)
    return results[:k]


def format_context(passages, header: str = "=== Retrieved context ===") -> str:
    if not passages:
        return ""
    lines = [header]
    for p in passages:
        lines.append(f"[{p[\"source\"]}] {p[\"text\"]}")
    lines.append("=== End context ===")
    return "\n".join(lines)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: query.py \"your question\" [--k 3]")
        sys.exit(1)
    q = sys.argv[1]
    k = 3
    if "--k" in sys.argv:
        k = int(sys.argv[sys.argv.index("--k") + 1])
    results = retrieve(q, k=k)
    if not results:
        print("[info] No result. Did you run ingest.py?")
        sys.exit(0)
    for p in results:
        print(f"\n--- {p[\"source\"]} #{p[\"chunk\"]} (score={p[\"score\"]:.3f}) ---")
        print(p["text"][:400])
