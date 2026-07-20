# Architecture Reference

## Objective

Maintain a general-purpose multi-node chatbot using Python 3.12, LangChain, LangGraph, FastAPI, and OpenAI. Keep local development zero-infrastructure with SQLite and Chroma. Use PostgreSQL for production checkpoints and durable memory; keep pgvector as the documented production retrieval target behind the existing knowledge-search interface.

## Runtime flow

```mermaid
flowchart LR
  A["POST /chat + JWT"] --> B["Authenticate subject"]
  B --> C["Hash subject + session into thread_id"]
  C --> D["router"]
  D -->|"chat"| E["chat"]
  D -->|"retrieve"| F["retrieve"]
  D -->|"tools"| G["tools"]
  D -->|"process"| K["process control"]
  K -->|"continue"| L["registered process subgraph"]
  E --> H["memory"]
  F --> H
  G --> H
  L --> H
  K -->|"control response"| I
  H --> I["respond"]
  I --> J["Checkpoint + JSON log"]
```

The normal chat/retrieval/tool path uses one structured router call and one response call. Explicit
process controls bypass model routing, while natural-language process selection uses the same
structured router. An AI-routed `start` or `continue` creates or reactivates the selected process
and consumes that same message as process input. Explicit lifecycle actions retain their strict API
semantics. Process prompts and results return directly without a second general response call. The
memory node makes another call only with explicit consent. Invalid routes and repeated provider,
tool, or process failures go directly to a fixed safe response.

## Component contracts

| Area | Required behavior |
| --- | --- |
| Authentication | `POST /auth/token` checks `AUTH_USERNAME` and the bcrypt `AUTH_PASSWORD_HASH`, then returns a 30-minute HS256 JWT signed with `JWT_SECRET`. |
| Chat | `POST /chat` requires a JWT and caller-provided validated `session_id`; identity comes only from the verified JWT. |
| Checkpoints | Hash verified subject plus session ID for LangGraph `thread_id`; never use a raw caller session as the namespace. |
| Memory | Conversation checkpoints are short-term. Durable facts/preferences require per-turn consent, confidence, and deterministic sensitive-data rejection. |
| Memory control | Authenticated callers can list/delete only records matched by their owner key. |
| Retrieval | Index application-owned Markdown via an explicit CLI command. Treat all returned text as untrusted reference data. |
| Tools | Register calculator, current-time, and knowledge-search explicitly; validate Pydantic arguments and bound results. |
| Processes | Register reviewed code plug-ins at startup. Persist one bounded, validated record per plug-in in the parent subject/session checkpoint. Only one process may be active at a time. A plug-in may declare that its active waiting state yields to suspended-process reminders. |
| Process routing | Explicit API actions override AI routing. Pass only trusted plug-in descriptions and status/step metadata to the router, never collected process payloads. |
| Suspended process reminder | When no blocking process is active, append a bounded reminder for suspended processes. GeneralAsk yields reminders after answering. A short `continue` switches to the sole suspended process without consuming the control reply; `cancel` cancels it. Never resume automatically. |
| Built-in processes | `number_counter` uses bounded structured model extraction and sums five stored integers, `color_note` uses bounded structured model extraction and summarizes three unique colors, and `general_ask` returns short model-only answers without retrieval. Number and color inputs may accumulate across messages or arrive together in one message. |
| Observability | Emit JSON logs with validated request IDs; never serialize bodies, secrets, retrieved text, or memory values. |

## Performance boundaries

- Construct OpenAI clients, embeddings, Chroma, repositories, checkpointer, and compiled graph once per application lifespan.
- Never embed or rebuild the knowledge index during requests or startup.
- Send at most the latest 20 conversation messages to the model.
- Bound each retrieved snippet, total retrieval context, tool result, durable-memory context, and final answer.
- Use only one retry layer: `ChatOpenAI(max_retries=1)`.
- Use one local worker with SQLite; use PostgreSQL for concurrent production workers.

## Verification matrix

| Change | Minimum check |
| --- | --- |
| Router or edge | Test structured route validation and the observable graph path. |
| Memory | Test consent default/opt-in, non-sticky consent, sensitive rejection, owner list/delete. |
| Tool | Test unknown name, extra/invalid arguments, resource limits, and safe failure. |
| API/auth | Test login success/failure, invalid/expired token rejection, validation, and ownership. |
| Retrieval | Test Markdown-only loading, persistent indexing, bounded search, and injection isolation. |
| Persistence | Test subject/session isolation and reopening the local SQLite checkpointer. |
| Process | Test start/continue/switch/cancel/status, exact-step resumption, number/color thresholds, the GeneralAsk no-retrieval boundary, invalid plug-in output, and safe failures. |
| Logging | Assert sentinel credentials/tokens are absent from JSON output. |

CI must use fakes for model and embedding boundaries and must not require OpenAI secrets or network calls.

## Explicit v1 non-goals

- Response streaming
- Registration, refresh tokens, roles, OAuth, or external identity providers
- User-uploaded documents or tenant knowledge bases
- Arbitrary tools, browser access, or model-selected code execution
- Deep Agents, open-ended subagent orchestration, runtime workflow uploads, or human approval interrupts
- Real OpenAI calls in CI
