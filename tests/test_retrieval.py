from pathlib import Path

from langchain_core.embeddings import Embeddings

from chatbot.retrieval.chroma import ChromaKnowledgeBase, load_markdown_documents


class KeywordEmbeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    @staticmethod
    def _embed(text: str) -> list[float]:
        lowered = text.lower()
        return [
            float(lowered.count("memory")),
            float(lowered.count("jwt")),
            0.01,
        ]


def test_markdown_loader_reads_only_application_owned_markdown(tmp_path: Path) -> None:
    (tmp_path / "guide.md").write_text("# Guide\n\nMemory requires consent.", encoding="utf-8")
    (tmp_path / "ignored.txt").write_text("not indexed", encoding="utf-8")

    documents = load_markdown_documents(tmp_path)

    assert len(documents) == 1
    assert documents[0].page_content.startswith("# Guide")
    assert documents[0].metadata["source"] == "guide.md"


def test_chroma_index_is_reused_for_bounded_search(tmp_path: Path) -> None:
    knowledge = tmp_path / "knowledge"
    knowledge.mkdir()
    (knowledge / "memory.md").write_text(
        "# Memory\n\nDurable memory requires explicit consent.", encoding="utf-8"
    )
    (knowledge / "auth.md").write_text(
        "# Authentication\n\nJWT access tokens expire after thirty minutes.", encoding="utf-8"
    )
    store = ChromaKnowledgeBase(
        persist_directory=tmp_path / "chroma",
        embeddings=KeywordEmbeddings(),
    )

    indexed = store.index_directory(knowledge)
    results = store.search("memory consent", limit=1)

    assert indexed == 2
    assert len(results) == 1
    assert results[0].source == "memory.md"
    assert "explicit consent" in results[0].content
