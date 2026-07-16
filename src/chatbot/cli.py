"""Operator commands for hashing credentials and indexing shared knowledge."""

import argparse
import getpass

import bcrypt
from langchain_openai import OpenAIEmbeddings

from chatbot.config import Settings
from chatbot.retrieval.chroma import ChromaKnowledgeBase


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="chatbot")
    subcommands = parser.add_subparsers(dest="command", required=True)
    subcommands.add_parser("hash-password", help="Generate a bcrypt hash using hidden input")
    subcommands.add_parser("ingest", help="Rebuild the Chroma index from shared Markdown")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "hash-password":
        password = getpass.getpass("Password: ")
        confirmation = getpass.getpass("Confirm password: ")
        if password != confirmation:
            raise SystemExit("Passwords do not match")
        if len(password) < 12:
            raise SystemExit("Password must contain at least 12 characters")
        print(bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode())
        return

    settings = Settings()
    settings.require_openai()
    embeddings = OpenAIEmbeddings(
        model=settings.openai_embedding_model,
        api_key=settings.openai_api_key.get_secret_value(),
    )
    knowledge = ChromaKnowledgeBase(
        persist_directory=settings.chroma_persist_directory,
        embeddings=embeddings,
    )
    indexed = knowledge.index_directory(settings.knowledge_directory)
    print(f"Indexed {indexed} Markdown chunks")


if __name__ == "__main__":
    main()
