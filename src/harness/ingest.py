from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from harness.config import CORPUS_DIR, settings
from harness.embeddings import get_embeddings

_NAME_RE = re.compile(r"^(?P<form>10-[KQ])_(?P<period>\d{4}-\d{2}-\d{2})_(?P<acc>.+)\.txt$")


def _metadata_for(path: Path) -> dict:
    m = _NAME_RE.match(path.name)
    if not m:
        return {"source": path.name}
    return {
        "source": path.name,
        "form": m.group("form"),
        "reporting_date": m.group("period"),
        "fiscal_year": m.group("period")[:4],
        "accession": m.group("acc"),
    }


def load_corpus(corpus_dir: Path) -> list[Document]:
    files = sorted(corpus_dir.glob("*.txt"))
    if not files:
        raise SystemExit(f"No .txt filings in {corpus_dir}. Run scripts/download_corpus.py first.")
    docs = []
    for f in files:
        text = f.read_text(encoding="utf-8")
        docs.append(Document(page_content=text, metadata=_metadata_for(f)))
    return docs


def split_documents(docs: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(docs)


def build_store(reset: bool = False) -> Chroma:
    if reset and settings.chroma_dir.exists():
        shutil.rmtree(settings.chroma_dir)
        print(f"Reset: removed {settings.chroma_dir}")

    docs = load_corpus(CORPUS_DIR)
    print(f"Loaded {len(docs)} filings from {CORPUS_DIR}")
    chunks = split_documents(docs)
    print(f"Split into {len(chunks)} chunks "
          f"(size={settings.chunk_size}, overlap={settings.chunk_overlap})")

    print(f"Embedding with backend='{settings.embedding_backend}' "
          f"(model may download on first run)...")
    store = Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        collection_name=settings.collection_name,
        persist_directory=str(settings.chroma_dir),
    )
    print(f"Persisted {store._collection.count()} vectors to {settings.chroma_dir}")
    return store


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reset", action="store_true", help="wipe the store before ingesting")
    args = ap.parse_args()
    store = build_store(reset=args.reset)

    hits = store.similarity_search("What was total revenue for the fiscal year?", k=3)
    print("\nSmoke query -> top hits:")
    for h in hits:
        snippet = h.page_content[:90].replace("\n", " ")
        print(f"  [{h.metadata.get('source')}] {snippet}...")


if __name__ == "__main__":
    main()
