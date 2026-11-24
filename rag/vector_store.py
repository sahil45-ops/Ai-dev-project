"""
Vector Store — Manages ChromaDB collections for storing and retrieving enriched tech-debt issues.
"""

from __future__ import annotations

import os
from typing import Any

from rag.embedder import IssueEmbedder, format_issue_document


def _sanitize_metadata(issue: dict[str, Any]) -> dict[str, Any]:
    """
    Ensure all metadata values are primitive types (str, int, float, bool)
    supported by ChromaDB metadata filters.
    """
    clean = {}
    for key, value in issue.items():
        if isinstance(value, (str, int, float, bool)):
            clean[key] = value
        elif value is None:
            clean[key] = ""
        else:
            clean[key] = str(value)
    return clean


class DebtVectorStore:
    """ChromaDB collection manager for technical debt issues."""

    def __init__(
        self,
        collection_name: str = "techdebt_collection",
        persist_dir: str | None = None,
        embedder: IssueEmbedder | None = None,
    ) -> None:
        import chromadb

        self.collection_name = collection_name
        self.embedder = embedder or IssueEmbedder()

        if persist_dir:
            os.makedirs(persist_dir, exist_ok=True)
            self.client = chromadb.PersistentClient(path=persist_dir)
        else:
            self.client = chromadb.Client()

        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "Enriched Technical Debt Issues"},
        )

    def ingest_issues(self, issues: list[dict[str, Any]], batch_size: int = 64) -> int:
        """
        Ingest a list of enriched issue records into ChromaDB.
        Returns the number of ingested items.
        """
        if not issues:
            return 0

        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        # Bound document, metadata and embedding memory to one batch.
        for offset in range(0, len(issues), batch_size):
            batch = issues[offset:offset + batch_size]
            documents = [format_issue_document(issue) for issue in batch]
            ids = []
            for idx, issue in enumerate(batch, offset):
                repo = issue.get("repo", "repo")
                func = issue.get("function", "module") or "module"
                ids.append(f"{repo}_{issue.get('tool', 'tool')}_{idx}_{func}_{issue.get('line', 0)}")
            self.collection.upsert(
                ids=ids, documents=documents,
                metadatas=[_sanitize_metadata(issue) for issue in batch],
                embeddings=self.embedder.encode_batch(documents),
            )
        return len(issues)

    def search_similar(
        self,
        query_text: str,
        n_results: int = 5,
        where_filter: dict | None = None,
    ) -> list[dict[str, Any]]:
        """
        Query ChromaDB for top-k similar technical-debt issues.
        Returns a list of dicts with 'document', 'metadata', 'distance', and 'id'.
        """
        query_embedding = self.embedder.encode_text(query_text)
        query_kwargs: dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": n_results,
        }
        if where_filter:
            query_kwargs["where"] = where_filter

        results = self.collection.query(**query_kwargs)

        items: list[dict[str, Any]] = []
        if not results or not results.get("ids") or not results["ids"][0]:
            return items

        ids = results["ids"][0]
        documents = results["documents"][0] if results.get("documents") else []
        metadatas = results["metadatas"][0] if results.get("metadatas") else []
        distances = results["distances"][0] if results.get("distances") else []

        for i in range(len(ids)):
            items.append({
                "id": ids[i],
                "document": documents[i] if i < len(documents) else "",
                "metadata": metadatas[i] if i < len(metadatas) else {},
                "distance": distances[i] if i < len(distances) else 0.0,
            })
        return items
