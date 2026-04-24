#!/usr/bin/env python3
"""
bm25-tiny — query
Pure-Python BM25 retrieval over a JSON index produced by ingest.py.

CLI:
    # Single-index (default)
    python query.py "your question" [--k 3]

    # Multi-scope: read bm25_<scope>.json, merge and re-rank
    python query.py "your question" --scope public --scope private

Library:
    from query import retrieve, format_context
    hits = retrieve("your question", k=5)
    hits = retrieve("your question", k=5, scopes=["public", "private"])

Index path (env-overridable):
    BM25_INDEX = ./store/bm25.json
    BM25_STORE = ./store            (used when --scope is given)
"""
import json
import os
import re
import sys
import unicodedata
from pathlib import Path

INDEX = Path(os.environ.get("BM25_INDEX", "./store/bm25.json"))
STORE = Path(os.environ.get("BM25_STORE", "./store"))
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

_INDEX_CACHE = {}


def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def tokenize(text: str):
    text = strip_accents(text.lower())
    toks = re.findall(r"[a-z0-9]{2,}", text)
    return [t for t in toks if t not in STOPWORDS]


def _load(path: Path):
    key = str(path)
    if key not in _INDEX_CACHE:
        if not path.exists():
            _INDEX_CACHE[key] = None
        else:
            _INDEX_CACHE[key] = json.loads(path.read_text(encoding="utf-8"))
    return _INDEX_CACHE[key]


def _score_against(idx, q_terms, scope_label=""):
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
            r = {
                "text": d["text"],
                "source": d["source"],
                "chunk": d["chunk"],
                "score": score,
            }
            if scope_label:
                r["scope"] = scope_label
            results.append(r)
    return results


def retrieve(query: str, k: int = 3, scopes=None):
    """Returns list of dicts: [{text, source, chunk, score, scope?}].

    scopes: optional list of scope names, e.g. ["public", "private"]. Each scope
    loads store/bm25_<scope>.json. Results across scopes are merged and re-ranked.
    If scopes is None, falls back to the single BM25_INDEX file.
    """
    q_terms = tokenize(query)
    if not q_terms:
        return []

    results = []
    if scopes:
        for s in scopes:
            idx = _load(STORE / f"bm25_{s}.json")
            if idx:
                results.extend(_score_against(idx, q_terms, scope_label=s))
    else:
        idx = _load(INDEX)
        if idx:
            results.extend(_score_against(idx, q_terms))

    results.sort(key=lambda r: r["score"], reverse=True)
    return results[:k]


def format_context(passages, header: str = "=== Retrieved context ===") -> str:
    if not passages:
        return ""
    lines = [header]
    for p in passages:
        tag = f"[{p['scope']}:{p['source']}]" if p.get("scope") else f"[{p['source']}]"
        lines.append(f"{tag} {p['text']}")
    lines.append("=== End context ===")
    return "\n".join(lines)


def parse_args(argv):
    import argparse
    ap = argparse.ArgumentParser(description="bm25-tiny query")
    ap.add_argument("query", help="query string")
    ap.add_argument("--k", type=int, default=3, help="top-k results (default: 3)")
    ap.add_argument("--scope", action="append", default=None,
                    help="scope name; reads store/bm25_<scope>.json. Repeatable.")
    return ap.parse_args(argv)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: query.py \"your question\" [--k 3] [--scope NAME ...]")
        sys.exit(1)
    args = parse_args(sys.argv[1:])
    results = retrieve(args.query, k=args.k, scopes=args.scope)
    if not results:
        print("[info] No result. Did you run ingest.py?")
        sys.exit(0)
    for p in results:
        tag = f"{p.get('scope', '')}:" if p.get("scope") else ""
        print(f"\n--- {tag}{p['source']} #{p['chunk']} (score={p['score']:.3f}) ---")
        print(p["text"][:400])
