# bm25-tiny

> Pure-Python BM25 retrieval for machines that numpy left behind.

Two files, stdlib only, ~300 lines. Ingests `.md` / `.txt` / `.pdf` / `.docx`,
builds a JSON BM25 index, runs lexical retrieval on it. No numpy, no torch,
no ChromaDB, no FAISS — nothing that needs `x86-64-v2`.

---

## EN

### Why this exists

`numpy >= 2.0` requires the `x86-64-v2` micro-architecture baseline (popcnt,
sse4.2, ...). Some environments still expose only `x86-64-v1` (sse, sse2):

- older QEMU / KVM images with a conservative `-cpu` model,
- low-tier VPS on aging hosts,
- Raspberry Pi and other non-x86 boards where pre-built wheels are scarce,
- minimal container bases where installing a C toolchain is not welcome.

On those hosts, modern embedding stacks refuse to start. `bm25-tiny` works
anywhere CPython 3.8+ works. For a small poetic / technical corpus (a few MB),
BM25 is also genuinely competitive with dense retrieval — it just wont wow
anyone at a conference.

### Install

```bash
git clone https://github.com/kangoulya/bm25-tiny.git
cd bm25-tiny
# Optional extras for PDF / DOCX:
pip install pypdf python-docx
```

### Use

```bash
mkdir -p sources store
cp ~/notes/*.md sources/
python ingest.py              # builds store/bm25.json
python query.py "my question" --k 5
```

As a library:

```python
from query import retrieve, format_context

hits = retrieve("quantum gardening", k=3)
prompt_context = format_context(hits)
```

Paths are env-overridable:

```bash
BM25_SOURCES=/data/docs BM25_STORE=/data/index python ingest.py
BM25_INDEX=/data/index/bm25.json python query.py "..."
```

### How it works

- **Tokenization**: NFD accent strip -> lowercase -> `[a-z0-9]{2,}` ->
  FR/EN stopword filter.
- **Chunking**: paragraph-aware, target 800 chars, 100-char overlap on
  oversized paragraphs.
- **Scoring**: Okapi BM25 (`k1 = 1.5`, `b = 0.75`), IDF with the
  `+ 1` variant (always non-negative).
- **Index**: a single JSON file (`{version, N, avgdl, idf, docs[]}`) held
  in memory on first query, ~10 kB per small source file.

### Origin

Extracted from the [KanGouLya](https://kangoulya.org) stack, where it
powers RAG for an LLM assistant on a VPS whose QEMU host does not expose
`x86-64-v2`. The same hosts also run the
[keyboards-as-shields](https://github.com/kangoulya/keyboards-as-shields)
manifesto.

### License

MIT. Do whatever. A note of where you got it is kind but not required.

---

## FR

### Pourquoi ca existe

`numpy >= 2.0` exige la base micro-architecture `x86-64-v2` (popcnt, sse4.2,
...). Certains environnements n exposent encore que `x86-64-v1` (sse, sse2) :

- vieilles images QEMU / KVM avec un `-cpu` conservateur,
- VPS low-cost sur hotes vieillissants,
- Raspberry Pi et autres cartes non-x86 avec peu de wheels precompiles,
- bases de conteneurs minimales ou installer un toolchain C est mal venu.

Sur ces hotes, les stacks d embeddings modernes refusent de demarrer.
`bm25-tiny` tourne partout ou CPython 3.8+ tourne. Pour un petit corpus
poetique / technique (quelques Mo), BM25 reste honorable face au dense
retrieval — il n impressionnera juste personne en conference.

### Installation

```bash
git clone https://github.com/kangoulya/bm25-tiny.git
cd bm25-tiny
# Extras optionnels pour PDF / DOCX :
pip install pypdf python-docx
```

### Usage

```bash
mkdir -p sources store
cp ~/notes/*.md sources/
python ingest.py              # construit store/bm25.json
python query.py "ma question" --k 5
```

Comme bibliotheque :

```python
from query import retrieve, format_context

hits = retrieve("jardinage quantique", k=3)
contexte_prompt = format_context(hits)
```

Les chemins sont surchargeables par variables d environnement :

```bash
BM25_SOURCES=/data/docs BM25_STORE=/data/index python ingest.py
BM25_INDEX=/data/index/bm25.json python query.py "..."
```

### Origine

Extrait de la stack [KanGouLya](https://kangoulya.org), ou il alimente un
RAG pour un assistant LLM sur un VPS dont l hote QEMU n expose pas
`x86-64-v2`. Les memes hotes font aussi tourner le manifeste
[keyboards-as-shields](https://github.com/kangoulya/keyboards-as-shields).

### Licence

MIT. Faites-en ce que vous voulez. Un clin d oeil a l origine est
apprecie mais pas exige.
