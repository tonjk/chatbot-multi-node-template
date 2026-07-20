"""Interactively test the local chatbot API from a terminal."""

from __future__ import annotations

import getpass
import json
import os
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def post_json(url: str, payload: dict[str, Any], token: str | None = None) -> dict[str, Any]:
    headers = {"Content-Type": "application/json"}
    if token is not None:
        headers["Authorization"] = f"Bearer {token}"

    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with urlopen(request, timeout=30) as response:
        return json.load(response)


def print_chat_response(result: dict[str, Any]) -> None:
    print("\nAssistant:")
    for line in str(result.get("response", "")).splitlines() or [""]:
        print(line)

    print(f"Route: {result.get('route', 'unknown')}")
    for process in result.get("processes", []):
        marker = "active" if process.get("active") else "inactive"
        print(
            f"Process: {process.get('name', 'unknown')} | "
            f"status={process.get('status', 'unknown')} | "
            f"step={process.get('step', 'unknown')} | {marker}"
        )
    print()

from dotenv import load_dotenv
load_dotenv()

def main() -> None:
    print(os.getenv("AUTH_USERNAME"))
    base_url = os.getenv("CHATBOT_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
    username = os.getenv("AUTH_USERNAME")
    password = os.getenv("AUTH_PASSWORD_HASH")
    session_id = os.getenv("CHATBOT_SESSION_ID", "manual-test")

    if not username or not password:
        print("Please set AUTH_USERNAME and AUTH_PASSWORD environment variables.")
        return

    try:
        token_result = post_json(
            f"{base_url}/auth/token",
            {"username": username, "password": password},
        )
        access_token = str(token_result["access_token"])
    except (HTTPError, URLError, KeyError, ValueError) as error:
        print(f"Authentication failed: {error}")
        return

    print(f"Connected. Session: {session_id}. Type 'exit' or 'quit' to stop.\n")

    while True:
        try:
            message = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if message.lower() in {"exit", "quit"}:
            print("Goodbye.")
            break
        if not message:
            continue

        try:
            result = post_json(
                f"{base_url}/chat",
                {
                    "session_id": session_id,
                    "message": message,
                    "memory_consent": False,
                },
                token=access_token,
            )
            print_chat_response(result)
        except HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            print(f"Chat request failed ({error.code}): {detail}\n")
        except (URLError, ValueError) as error:
            print(f"Chat request failed: {error}\n")


if __name__ == "__main__":
    main()
