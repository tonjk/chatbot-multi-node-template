# Chatbot Multi-Node Template — Agent Guide

## Scope

Build a general-purpose Python chatbot with LangChain and LangGraph. Keep the first release small, secure by default, and runnable locally. Do not add application code, dependencies, or infrastructure unless the requested task needs it.

## Intended architecture

- Use Python 3.12+, `pyproject.toml`, and `uv`.
- Put application code in `src/chatbot/`; keep tests in `tests/`.
- Expose a synchronous FastAPI JSON API. Do not add token streaming in v1.
- Use OpenAI through `langchain-openai`; select the model through environment configuration, never hard-code API keys.
- Represent the workflow as a LangGraph state graph with Pydantic-validated state and route decisions.

## Graph responsibilities

The first graph contains these focused nodes:

1. `router` selects the next capability using structured, validated output.
2. `chat` handles ordinary conversation.
3. `retrieve` searches the shared application knowledge base.
4. `tools` invokes only registered, allow-listed tools.
5. `memory` extracts durable, confidence-checked facts or preferences.
6. `respond` produces the final answer and never exposes internal errors.

Keep node responsibilities separate. Do not turn the router into a second assistant, let retrieval write memory, or permit arbitrary tool calls from model text.

## State, identity, and memory

- Authenticate with a fixed username and bcrypt password hash supplied through `AUTH_USERNAME` and `AUTH_PASSWORD_HASH`.
- Issue HS256 JWT access tokens with `JWT_SECRET`, a 30-minute expiry, and the username in `sub`.
- Scope every graph checkpoint and long-term memory record to the authenticated subject and a `session_id`.
- Use short-term LangGraph checkpoint memory for each conversation.
- Long-term memory is opt-in: `memory_consent` defaults to `false`, and the `memory` node may write only when it is explicitly `true`.
- Store only useful facts/preferences with confidence checks. Never persist secrets, credentials, payment data, health data, or raw chat transcripts as durable memory.
- Use SQLite locally and document PostgreSQL for production. Provide authenticated endpoints to list and delete a user's saved memories.

## Retrieval and tools

- Start with one shared, application-managed Markdown knowledge base. Do not add user uploads or tenant collections in v1.
- Use a local Chroma store for the demo; document PostgreSQL with `pgvector` as the production option.
- Include only offline-safe example tools: calculator, current time, and knowledge-base search.
- Validate every tool input and output. Apply an output-size limit and return safe failures.

## API and safety baseline

- Provide `POST /auth/token`, `POST /chat`, `GET /memories`, and deletion endpoints for the authenticated user, plus health/readiness endpoints.
- Validate request models with Pydantic. Do not log request secrets, JWTs, model prompts containing sensitive data, or raw long-term memory.
- Emit structured JSON logs with correlation IDs for requests and graph nodes.
- Treat retrieved documents as untrusted input. Defend prompts against prompt injection and never let retrieved instructions bypass tool allow-lists or policy checks.
- Retry a transient model/tool failure once. For invalid routes or repeated failures, use `respond` to return a concise, non-sensitive failure message.

## Verification expectations

- Use `pytest`.
- Cover router validation, memory-consent behavior, memory ownership/deletion, and tool validation with unit tests.
- Cover important graph paths with mocked-model integration tests.
- Keep a small, versioned evaluation dataset. CI must not require real OpenAI calls or secrets.

## Delivery boundaries

- Provide a Dockerfile and Docker Compose plan for FastAPI plus PostgreSQL; retain SQLite/Chroma as the local zero-infrastructure default.
- Prefer small diffs, explicit configuration, and dependency injection at I/O boundaries.
- Do not add refresh tokens, role systems, OAuth, user registration, streaming, user uploads, or production-only infrastructure unless the user explicitly asks.

## Agent skill routing

- Read `skills/build-langgraph-chatbot/SKILL.md` before changing application behavior.
- Invoke `.agents/skills/ecosystem-primer/SKILL.md` first for framework work.
- Then use `langgraph-fundamentals`, `langgraph-persistence`, `langchain-rag`, or `langchain-dependencies` only when their component is in scope.
- Use the `openai-docs` skill for current OpenAI model/API decisions and `tdd` for behavior changes.
- Do not use Deep Agents, human-in-the-loop, or LangGraph CLI skills unless the requested feature actually needs them.
