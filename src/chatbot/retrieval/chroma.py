"""Application-owned Markdown indexing and persistent Chroma retrieval."""

import hashlib
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from chatbot.retrieval.models import KnowledgeSnippet

_MAX_MARKDOWN_BYTES = 1_000_000
_MAX_SNIPPET_CHARS = 1_000


class ChromaKnowledgeBase:
    """Reuse one persistent Chroma collection for indexing and querying."""

    def __init__(
        self,
        *,
        persist_directory: Path,
        embeddings: Embeddings,
        collection_name: str = "chatbot-knowledge",
    ) -> None:
        persist_directory.mkdir(parents=True, exist_ok=True)
        self._store = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=str(persist_directory),
        )

    def index_directory(self, knowledge_directory: Path) -> int:
        """Replace the index with bounded chunks from shared Markdown files."""

        documents = load_markdown_documents(knowledge_directory)
        splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=150)
        chunks = splitter.split_documents(documents)
        self._store.reset_collection()
        if not chunks:
            return 0
        ids = [
            hashlib.sha256(
                f"{chunk.metadata.get('source', '')}\0{index}\0{chunk.page_content}".encode()
            ).hexdigest()
            for index, chunk in enumerate(chunks)
        ]
        self._store.add_documents(chunks, ids=ids)
        return len(chunks)

    def search(self, query: str, *, limit: int = 4) -> list[KnowledgeSnippet]:
        """Return a small, bounded set of snippets from the existing index."""

        normalized_query = query.strip()
        if not normalized_query:
            return []
        documents = self._store.similarity_search(normalized_query[:500], k=max(1, min(limit, 8)))
        return [
            KnowledgeSnippet(
                content=document.page_content[:_MAX_SNIPPET_CHARS],
                source=str(document.metadata.get("source", "unknown"))[:200],
            )
            for document in documents
        ]


def load_markdown_documents(directory: Path) -> list[Document]:
    """Load UTF-8 Markdown strictly from the configured application directory."""

    root = directory.expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"Knowledge directory does not exist: {root}")
    documents: list[Document] = []
    for path in sorted(root.rglob("*.md")):
        relative = path.relative_to(root)
        if any(part.startswith(".") for part in relative.parts):
            continue
        if path.stat().st_size > _MAX_MARKDOWN_BYTES:
            raise ValueError(f"Markdown file is too large: {relative.as_posix()}")
        content = path.read_text(encoding="utf-8").strip()
        if content:
            documents.append(
                Document(
                    page_content=content,
                    metadata={"source": relative.as_posix()},
                )
            )
    return documents
