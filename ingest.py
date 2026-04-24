#!/usr/bin/env python3
"""
bm25-tiny — ingest
Pure-Python BM25 indexer. Zero deps beyond stdlib (optional: pypdf, python-docx).

Builds a JSON BM25 index from a source directory of .md / .txt / .pdf / .docx.

Why pure Python? numpy>=2 requires x86-64-v2 (popcnt, sse4.2). Some VPS, old
hardware, QEMU images or minimal containers only expose x86-64-v1 (sse, sse2).
Rather than pinning numpy<2, this module does BM25 by hand: tokenization,
inverted index, Okapi BM25 scoring. Plenty fast for small-to-medium corpora
(a few MB of text).

Usage:
    # Single-index (default)
    python ingest.py [sources_dir] [store_dir]

    # Multi-scope: sources_dir contains subdirs, one index per subdir
    python ingest.py --scopes public,private
    # -> sources/public/  -> store/bm25_public.json
    # -> sources/private/ -> store/bm25_private.json

Defaults (env-overridable):
    BM25_SOURCES = ./sources
    BM25_STORE   = ./store
"""
import argparse
import json
import math
import os
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

# FR + EN stopwords (kept deliberately short and accent-stripped).
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


def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def tokenize(text: str):
    text = strip_accents(text.lower())
    toks = re.findall(r"[a-z0-9]{2,}", text)
    return [t for t in toks if t not in STOPWORDS]


def read_file(path: Path) -> str:
    ext = path.suffix.lower()
    try:
        if ext in (".md", ".txt"):
            return path.read_text(encoding="utf-8", errors="ignore")
        if ext == ".pdf":
            try:
                from pypdf import PdfReader
                r = PdfReader(str(path))
                return "\n\n".join((p.extract_text() or "") for p in r.pages)
            except Exception as e:
                print(f"[warn] pdf {path}: {e}")
                return ""
        if ext == ".docx":
            try:
                from docx import Document
                d = Document(str(path))
                return "\n".join(p.text for p in d.paragraphs)
            except Exception as e:
                print(f"[warn] docx {path}: {e}")
                return ""
    except Exception as e:
        print(f"[warn] read {path}: {e}")
    return ""


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    text = re.sub(r"\r\n?", "\n", text)
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks = []
    buf = ""
    for p in paragraphs:
        if len(buf) + len(p) + 2 <= size:
            buf = (buf + "\n\n" + p).strip()
        else:
            if buf:
                chunks.append(buf)
            if len(p) > size:
                for i in range(0, len(p), size - overlap):
                    chunks.append(p[i:i + size])
                buf = ""
            else:
                buf = p
    if buf:
        chunks.append(buf)
    return chunks


def build_index(sources_dir: Path, index_path: Path, label: str = ""):
    """Scan sources_dir and write a BM25 index to index_path. Returns number of chunks."""
    sources_dir.mkdir(parents=True, exist_ok=True)
    index_path.parent.mkdir(parents=True, exist_ok=True)

    files = [p for p in sources_dir.rglob("*") if p.is_file() and p.suffix.lower() in (".md", ".txt", ".pdf", ".docx")]
    if not files:
        print(f"[{label or 'info'}] No sources in {sources_dir}. Drop .md/.txt/.pdf/.docx files there.")
        return 0

    docs = []
    df = Counter()

    for f in sorted(files):
        text = read_file(f)
        if not text.strip():
            continue
        rel = str(f.relative_to(sources_dir))
        n_chunks = 0
        for i, c in enumerate(chunk_text(text)):
            toks = tokenize(c)
            if not toks:
                continue
            tf = Counter(toks)
            docs.append({
                "source": rel,
                "chunk": i,
                "text": c,
                "tf": dict(tf),
                "length": len(toks),
            })
            for term in tf:
                df[term] += 1
            n_chunks += 1
        tag = f"[{label}] " if label else ""
        print(f"{tag}{rel}: {n_chunks} chunks")

    N = len(docs)
    if N == 0:
        print(f"[{label or 'err'}] No chunk generated.")
        return 0

    avgdl = sum(d["length"] for d in docs) / N
    idf = {t: math.log((N - n + 0.5) / (n + 0.5) + 1) for t, n in df.items()}

    index = {
        "version": 1,
        "N": N,
        "avgdl": avgdl,
        "idf": idf,
        "docs": docs,
    }
    index_path.write_text(json.dumps(index, ensure_ascii=False), encoding="utf-8")
    size_kb = index_path.stat().st_size / 1024
    tag = f"[{label}] " if label else ""
    print(f"\n{tag}{len(files)} files, {N} chunks, BM25 index {size_kb:.1f} KB -> {index_path}")
    return N


def parse_args():
    ap = argparse.ArgumentParser(description="bm25-tiny ingest")
    ap.add_argument("sources", nargs="?", default=None, help="sources directory (default: $BM25_SOURCES or ./sources)")
    ap.add_argument("store", nargs="?", default=None, help="store directory (default: $BM25_STORE or ./store)")
    ap.add_argument("--scopes", help="comma-separated scope names; each <sources>/<scope>/ becomes bm25_<scope>.json")
    return ap.parse_args()


def main():
    args = parse_args()
    sources = Path(args.sources or os.environ.get("BM25_SOURCES", "./sources"))
    store = Path(args.store or os.environ.get("BM25_STORE", "./store"))

    if args.scopes:
        scopes = [s.strip() for s in args.scopes.split(",") if s.strip()]
        total = 0
        for scope in scopes:
            total += build_index(sources / scope, store / f"bm25_{scope}.json", label=scope)
        if total == 0:
            sys.exit(1)
    else:
        n = build_index(sources, store / "bm25.json")
        if n == 0:
            sys.exit(1)


if __name__ == "__main__":
    main()
