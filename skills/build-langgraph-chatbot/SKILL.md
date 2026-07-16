---
name: build-langgraph-chatbot
description: Build, extend, test, or review this repository's general-purpose Python chatbot using LangChain, LangGraph, FastAPI, OpenAI, opt-in memory, shared Markdown retrieval, and safe allow-listed tools. Use when changing graph nodes, chatbot behavior, memory, retrieval, tools, API contracts, authentication, tests, dependencies, or deployment guidance in this repository.
---

# Build the LangGraph chatbot

Read the repository `AGENTS.md` and [the architecture reference](references/architecture.md) before changing this project.

## Select supporting skills

1. Invoke `.agents/skills/ecosystem-primer/SKILL.md` first for any framework work.
2. Load `langgraph-fundamentals` for graph code and `langgraph-persistence` for checkpoints or memory.
3. Load `langchain-rag` for retrieval and `langchain-dependencies` for package changes.
4. Use the system `openai-docs` skill for current OpenAI decisions.
5. Use `tdd` for behavior changes and build one observable vertical slice at a time.

Do not load Deep Agents, human-in-the-loop, orchestration, or LangGraph CLI skills unless the requested feature requires them.

## Workflow

1. Inspect the graph, runtime context, API contracts, persistence boundaries, and relevant tests.
2. Define the observable behavior and the narrow test that proves it.
3. Keep each node responsible for one capability. Return partial state updates.
4. Keep trusted identity, session, consent, and correlation data in runtime context, not model output.
5. Preserve subject/session isolation. Require `memory_consent: true` at the memory node and repository boundary.
6. Treat model output, retrieved content, and tool output as untrusted. Invoke only registered tools.
7. Reuse long-lived clients, compiled graphs, checkpointers, and indexes; keep checkpointed context bounded.
8. Run the narrow test, then the full offline checks. Report any unverified provider or production risk.

## Non-negotiable boundaries

- Keep the API synchronous and non-streaming in v1.
- Select OpenAI models through environment variables; never commit real secrets.
- Do not log bodies, tokens, passwords, raw retrieved chunks, or durable-memory contents.
- Keep the knowledge base shared, application-owned, and Markdown-only.
- Do not introduce registration, refresh tokens, roles, OAuth, arbitrary tools, or user uploads without an explicit request.

