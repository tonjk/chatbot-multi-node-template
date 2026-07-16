from pathlib import Path

from chatbot.memory.repository import MemoryRepository
from chatbot.memory.schemas import MemoryCandidate


def repository_for(path: Path) -> MemoryRepository:
    return MemoryRepository(f"sqlite:///{path}")


def candidate(content: str = "I prefer concise answers") -> MemoryCandidate:
    return MemoryCandidate(
        should_store=True,
        content=content,
        category="preference",
        confidence=0.95,
        sensitive=False,
    )


def test_durable_memory_requires_explicit_consent(tmp_path: Path) -> None:
    repository = repository_for(tmp_path / "memory.db")

    assert repository.add("alice", "session-a", candidate(), consent=False) is None
    assert repository.list_for_subject("alice") == []

    stored = repository.add("alice", "session-a", candidate(), consent=True)

    assert stored is not None
    assert stored.content == "I prefer concise answers"
    assert len(repository.list_for_subject("alice")) == 1
    repository.close()


def test_durable_memory_rejects_sensitive_or_low_confidence_candidates(tmp_path: Path) -> None:
    repository = repository_for(tmp_path / "memory.db")
    password = candidate("My password is swordfish")
    uncertain = candidate("I might like long answers")
    uncertain = uncertain.model_copy(update={"confidence": 0.4})

    assert repository.add("alice", "session-a", password, consent=True) is None
    assert repository.add("alice", "session-a", uncertain, consent=True) is None
    assert repository.list_for_subject("alice") == []
    repository.close()


def test_memory_listing_and_deletion_are_owner_scoped(tmp_path: Path) -> None:
    repository = repository_for(tmp_path / "memory.db")
    alice_memory = repository.add("alice", "shared-session", candidate(), consent=True)
    bob_memory = repository.add(
        "bob", "shared-session", candidate("I prefer detailed answers"), consent=True
    )
    assert alice_memory is not None
    assert bob_memory is not None

    assert [item.id for item in repository.list_for_subject("alice")] == [alice_memory.id]
    assert repository.delete("bob", alice_memory.id) is False
    assert repository.delete("alice", alice_memory.id) is True
    assert repository.list_for_subject("alice") == []
    assert [item.id for item in repository.list_for_subject("bob")] == [bob_memory.id]
    repository.close()
