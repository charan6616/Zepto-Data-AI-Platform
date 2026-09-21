"""Ingestion: docs/ -> one chunk per document -> embed -> ChromaDB.

Run once:  python ingest.py
Creates/updates the persistent "zepto_policies" collection under data/chroma/
(for 8 documents: ids doc_01 .. doc_08).
"""

from pathlib import Path

from app.store import COLLECTION_NAME, count_chunks, get_collection, upsert_chunks

DOCS_DIR = Path(__file__).resolve().parent / "docs"

TITLES = {
    "doc_01": "Delivery Policy",
    "doc_02": "Returns & Refunds",
    "doc_03": "Membership Tiers",
    "doc_04": "Order Tracking",
    "doc_05": "Order Cancellation Policy",
    "doc_06": "Damaged or Missing Items",
    "doc_07": "Gift Cards",
    "doc_08": "Customer Support Hours",
}


def load_documents() -> tuple[list[str], list[str], list[dict]]:
    ids = sorted(p.stem for p in DOCS_DIR.glob("doc_*.txt"))
    texts = [(DOCS_DIR / f"{cid}.txt").read_text(encoding="utf-8").strip() for cid in ids]
    metadatas = [{"doc": cid, "title": TITLES[cid]} for cid in ids]
    return ids, texts, metadatas


def main() -> None:
    ids, texts, metadatas = load_documents()
    n = upsert_chunks(ids, texts, metadatas)
    print(f"Upserted {n} chunks into ChromaDB collection '{COLLECTION_NAME}' "
          f"({count_chunks()} stored).")


if __name__ == "__main__":
    main()