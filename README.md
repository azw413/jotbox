# Jotbox

Jotbox is a Rocket-based note taking application that combines a browser editor, Retrieval Augmented Generation (RAG) search, and a per-user SQLite/vector store. Users authenticate with Google, take notes in a ProseMirror-based editor, and can search or chat over their own knowledge base.

## Capabilities

- Google Sign-In gated access to a personal workspace (`src/main.rs`, `templates/index.html.hbs`).
- Markdown editor with CRUD actions, recent/related note lists, and a Cytoscape powered note graph (`templates/notes.html.hbs`, `html/js/cy-data.js`).
- Per-user SQLite tables plus [`sqlite-vec`](https://github.com/asg017/sqlite-vec) for high dimensional embedding search (`src/datastore.rs`).
- Document ingestion that generates embeddings through a HuggingFace Text Embedding Inference (TEI) server (`src/local_embeddings.rs`). Notes cannot be saved while this service is offline; errors are surfaced in the UI and plaintext is discarded after embeddings are created.
- `/api/chat/retrieve` + `/api/chat/generate` endpoints that power RAG chat through an Ollama-hosted OpenAI-compatible model (`src/openai.rs`, `templates/notes.html.hbs`).
- Client-side encryption for note bodies (Speck-64/128) so the database only stores ciphertext (`templates/notes.html.hbs`, `html/js/encryption.js`).
- Static front-end assets served by Rocket’s `FileServer` (`src/main.rs`, `html/`).

## Architecture Overview

```
+-------------------------+        +---------------------------+
| Browser (Bootstrap UI)  |<------>| Rocket routes             |
| - Google login          |   HTTPS| - Auth + session cookies  |
| - ProseMirror editor    |        | - JSON note/chat APIs     |
| - Cytoscape graph       |        +---------------------------+
| - Client-side Speck lib |                    |
+-------------------------+                    |
             |                                  v
             |                     +---------------------------+
             |                     | SQLite + sqlite-vec       |
             |                     | - collections/documents   |
             |                     | - per-user embeddings_*   |
             |                     +---------------------------+
             |                                  ^
             v                                  |
+-------------------------+        +---------------------------+
| HuggingFace TEI server  |        | OpenAI API                |
| (http://jura:8080/embed)|        | (chat + fallback embed)   |
+-------------------------+        +---------------------------+
```

- Each user gets a logical collection. Their embeddings live in tables named `embeddings_<collection>` that are created on demand (`DataStoreClient::check_embeddings`).
- Similarity search uses the `vec0` virtual table and `MATCH` queries; the UI graph reuses those distances to link notes.
- Chat arity: gather last two conversational turns + new query, pull top-3 related notes, then call OpenAI with that context.
- Static assets (CSS/JS/images) are mounted from `html/` so front-end changes do not require recompiling Rust.

## Requirements

- Rust 1.75+ (Rocket 0.5).
- SQLite3 plus the bundled `sqlite-vec` extension (`libsqlite3_sys` dynamically loads it at startup).
- A running TEI service reachable from the Rocket server at `http://jura:8080/embed`. You can use the official Docker image, e.g.

  ```bash
  docker run --rm -p 8080:80 \
    ghcr.io/huggingface/text-embedding-inference:1.5 \
    --model-id intfloat/multilingual-e5-large-instruct
  ```

- An Ollama instance (or any OpenAI-compatible chat API) reachable at `LLM_API_BASE` that serves `LLM_MODEL` (defaults work with `ollama serve` and `ollama pull gpt-oss:20b`).
- An OpenAI-compatible API key exported as `OPENAI_API_KEY` (or `LLM_API_KEY`) if your chat endpoint requires bearer auth; the Rocket server forwards prompts to this endpoint.
- A Google OAuth client configured for the frontend domain. Export its client id via `JOTBOX_GOOGLE_CLIENT_ID`.

## Configuration

Jotbox reads its secrets and service endpoints from environment variables at startup:

| Variable | Description |
|----------|-------------|
| `JOTBOX_SPECK_KEY` | 128-bit Speck key encoded as hex (e.g. `0xf0b3...`). Used to encrypt session ids. |
| `JOTBOX_GOOGLE_CLIENT_ID` | OAuth client id for Google sign-in. |
| `JOTBOX_EMBEDDINGS_URL` | URL of the Text Embedding Inference service (e.g. `http://localhost:8080/embed`). |
| `JOTBOX_DATA_DIR` | Root directory for encrypted note databases (defaults to `data`). Each user gets their own sharded subdirectory + SQLite file. |
| `OPENAI_API_KEY` / `LLM_API_KEY` | Bearer token used by the server when calling your LLM endpoint. |
| `LLM_API_BASE` | OpenAI-compatible chat endpoint (defaults to `http://jura:11434/v1` for Ollama). |
| `LLM_MODEL` | Model name served by the LLM endpoint (defaults to `gpt-oss:20b`). |
| `LLM_MAX_TOKENS` | Optional maximum tokens per chat completion (defaults to `256` to keep responses fast). |
| `LLM_TEMPERATURE` | Sampling temperature for chat completions (defaults to `0.6`). |
| `LLM_TOP_P` | Top-p nucleus sampling parameter (defaults to `0.95`). |
| `LLM_TOP_K` | Top-k sampling parameter (defaults to `20`). |
| `LLM_ENABLE_THINKING` | Set to `false` to disable the “thinking” phase for compatible models (defaults to `true`). |

## Running Locally

1. Install dependencies, run the TEI server and Ollama (or another LLM endpoint), and export the environment variables listed above.
2. If you have an older plaintext database, remove `data/jotbox.db` so existing rows don’t fail to decrypt. The server now creates all required tables automatically during startup, so no separate init script is required.

3. Start the Rocket app (debug mode listens on `localhost:8000`):

   ```bash
   cargo run
   ```
   For a production-like run with optimizations enabled:

   ```bash
   cargo run --release
   ```

4. Browse to `http://localhost:8000`, sign in with Google, set a local encryption password, and start creating notes.

   Each Google account now gets its own SQLite file at `data/<last-three-digits>/<google-id>.db`. The directory is created on demand, so you can delete a single user’s file to reset just their notes without touching the rest of the workspace.

The first login per collection automatically creates the matching `embeddings_<collection>` vector table. Static assets such as `html/jotbox.css` and `html/js/encryption.js` are served directly by Rocket.

## Encryption

- The first time you open the notes UI you’ll be prompted for a password. A WebCrypto PBKDF2 (SHA-256, 250k iterations) derived 128‑bit key is cached only in `sessionStorage`, and the salt lives in `localStorage`. If the cache is cleared you will be prompted again before decrypting notes or chatting.
- Every note body is encrypted locally (Speck-64/128 with a random IV per save) before `/api/save` sees it. The database only stores ciphertext, while titles remain clear so the UI can render lists.
- Retrieval APIs return ciphertext; the browser decrypts it before rendering or before assembling chat context.
- AI endpoints still receive plaintext contexts on the server so embeddings and chat prompts can be generated, but that plaintext is discarded immediately after use.

## API Surface

| Route            | Description                                             |
|------------------|---------------------------------------------------------|
| `GET /`          | Marketing/Google sign-in landing page.                  |
| `POST /`         | Validates Google token, creates session + embeddings.   |
| `GET /api/recent`| Six most recent notes for the user.                     |
| `POST /api/related` | Vector search for similar notes given free text.    |
| `GET /api/retrieve/<id>` | Fetch a specific note body.                    |
| `GET /api/delete/<id>`   | Delete note and matching embedding vector.     |
| `POST /api/save` | Insert/update a note, generate embeddings server-side, return id.   |
| `POST /api/chat/retrieve` | Return the ids/scores of the most relevant notes for a query + history. |
| `POST /api/chat/generate` | Generate a response using the client-supplied context via Ollama. |
| `GET /api/graph` | Return Cytoscape-friendly graph data.                   |

All JSON endpoints expect the `X-Session-Key` header that encodes the collection id.

## Project Layout

- `src/main.rs` – Rocket routing, auth, and session/token handling.
- `src/datastore.rs` – SQLite access layer and vector graph builder.
- `src/local_embeddings.rs` – Minimal client for the TEI embedding service.
- `src/openai.rs` – Minimal HTTP client for the OpenAI-compatible (Ollama) chat endpoint.
- `templates/` – Handlebars templates (`index`, `notes`).
- `html/` – Static assets (Bootstrap theme, editor JS, encryption helper, Cytoscape).
- `speck/` – Local Speck cipher implementation used for session/encryption helpers.

## Current Status

### Completed

- Baseline CRUD flow with markdown editing and RAG-powered related note suggestions.
- Graph visualization of semantic relationships plus search-based highlighting.
- Chat retrieval endpoint plus client-side completion that reuse stored notes as grounding context without sending plaintext to Rocket.
- SQLite schema bootstrap and per-user vector tables.
- Google sign-in wired through Rocket and frontend templates.
- HMAC-SHA256 session tokens with 24h expiry so credentials can be rotated without touching the database.
- Server-mediated TEI + LLM calls keep internal endpoints private; the browser only shares decrypted snippets for the current chat and the server discards them once the response returns.
- Client-held encryption keys and encrypted note storage; the server never sees plaintext at rest.
- Graceful error propagation when the TEI service or OpenAI span is unavailable.

### Outstanding / Gaps

- **Plaintext exposure to upstream services.** Embedding generation and chat completion require sending decrypted note text to the TEI server and LLM endpoint; isolate or harden those services accordingly.
- **Multidevice key sync.** The encryption password is browser-local; there is no UX for re-entering/rotating keys across devices or recovering from lost cookies.
- **Operational guardrails are thin.** There is no rate limiting, log redaction, or request validation around long bodies, so malicious payloads could force expensive embeddings or OpenAI calls.
- **Service dependencies lack backoff.** TEI/OpenAI outages bubble up to the user but the backend does not retry, debounce, or queue work.
- **Testing coverage is minimal.** Only the graph utility and embedding client have basic unit tests; there are no integration tests for APIs or persistence flows.
- **Local developer experience assumes specific services.** The TEI endpoint and database bootstrap still require manual steps; scripting/containers would simplify onboarding.

## Testing

Unit tests:

```bash
cargo test
```

The `local_embeddings` test makes a real HTTP call to the TEI server at `http://jura:8080/embed`, so either skip it or point the constant to a reachable service before running the suite.

## Next Steps

1. Add throttling and request validation around TEI/LLM calls so abusive clients cannot exhaust those backends.
2. Provide a UX for changing/re-entering encryption passwords (and consider optional multi-device key sync).
3. Add retry/backoff or queueing around TEI/OpenAI calls plus server-side request validation/rate limiting.
4. Expand automation around database migrations, TEI startup, and local dev scripts.
5. Grow test coverage (API-level tests around CRUD/encryption/chat) to prevent regressions while iterating on the RAG stack.
