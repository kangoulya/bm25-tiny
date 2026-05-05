#!/usr/bin/env python3
"""MCP server — bm25-tiny RAG Search.

Exposes BM25 retrieval as MCP tools for LLM context injection.
Works with any corpus indexed by ingest.py.

Usage:
    # Default: single index (store/bm25.json)
    python mcp_server.py

    # Multi-scope: loads store/bm25_<scope>.json per scope
    BM25_SCOPES=public,private python mcp_server.py

Paths are env-overridable:
    BM25_STORE  = ./store   (directory with bm25*.json files)
    BM25_INDEX  = ./store/bm25.json  (fallback when no --scope given)
    BM25_SCOPES = public,private     (comma-separated scope names)

Requires: mcp[cli]  (pip install "mcp[cli]")
"""
import os
import sys

# So we can import query.py from the same directory
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from query import retrieve, format_context

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    print("mcp[cli] not installed. Run: pip install 'mcp[cli]'", file=sys.stderr)
    sys.exit(1)

_SCOPES_ENV = os.environ.get("BM25_SCOPES", "").strip()
DEFAULT_SCOPES = [s.strip() for s in _SCOPES_ENV.split(",") if s.strip()] if _SCOPES_ENV else None

mcp = FastMCP("bm25-tiny", log_level="WARNING")


@mcp.tool()
def search(query: str, k: int = 3, scopes: list[str] | None = None) -> str:
    """Search the BM25 index and return formatted passages.

    Args:
        query: Search terms or question (e.g. 'manifesto resilience')
        k: Number of passages to return (default: 3, max recommended: 5)
        scopes: Scope names to search. If omitted, uses BM25_SCOPES env var.
                If that's also unset, searches the default single index.
                Example: ['public', 'private'] to merge both corpora.

    Returns:
        Formatted passages ready for LLM prompt injection.
    """
    if scopes is None:
        scopes = DEFAULT_SCOPES
    passages = retrieve(query, k=k, scopes=scopes)
    if not passages:
        return "[info] No results in BM25 index. Did you run ingest.py?"
    return format_context(passages)


@mcp.tool()
def search_raw(query: str, k: int = 3, scopes: list[str] | None = None) -> str:
    """Search the BM25 index — returns full text without truncation.

    Same as search() but returns complete passage text (up to 800 chars each)
    including scores. Better for detailed analysis than quick prompt injection.

    Args:
        query: Search terms or question
        k: Number of passages (default: 3)
        scopes: Scope names (see search() for details)

    Returns:
        Full passages with scores, source and complete text.
    """
    if scopes is None:
        scopes = DEFAULT_SCOPES
    passages = retrieve(query, k=k, scopes=scopes)
    if not passages:
        return "[info] No results."
    parts = []
    for p in passages:
        src = p["source"].split("/")[-1]
        text = p["text"].strip()
        if len(text) > 800:
            text = text[:800].rsplit(" ", 1)[0] + "..."
        tag = f"{p.get('scope', '')}:" if p.get("scope") else ""
        parts.append(f"[{tag}{src} #{p['chunk']} score={p['score']:.3f}] {text}")
    return "\n---\n".join(parts)


if __name__ == "__main__":
    mcp.run()
