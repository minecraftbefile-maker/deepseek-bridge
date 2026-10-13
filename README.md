# DeepSeek Web → OpenAI Bridge

A self-contained, OpenAI-compatible HTTP bridge for DeepSeek's web API (`chat.deepseek.com`). Point any OpenAI SDK client at `http://127.0.0.1:8000/v1` and it works — streaming, reasoning, sessions, and automatic token refresh included.

Built for local use. No paid API key required. No browser automation. No headless Chromium. Just a small Python server and a tiny browser extension that keeps your `userToken` in sync.

---

## Table of Contents

- [What It Does](#what-it-does)
- [How It Works](#how-it-works)
- [Requirements](#requirements)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Browser Extension Setup](#browser-extension-setup)
- [Using the Bridge](#using-the-bridge)
- [API Reference](#api-reference)
- [Troubleshooting](#troubleshooting)
- [FAQ](#faq)
- [Project Layout](#project-layout)
- [Security Notes](#security-notes)
- [Limitations](#limitations)
- [License](#license)

---

## What It Does

DeepSeek's public API requires a paid account and a formal API key. The web app at `chat.deepseek.com`, however, is free and uses an internal API that the browser talks to directly.

This project exposes that internal API as a standard **OpenAI-compatible endpoint**, so any tool that already speaks the OpenAI protocol (LangChain, LlamaIndex, Continue.dev, Cursor, Open WebUI, your own scripts) can talk to DeepSeek for free.

**Features:**

- OpenAI `/v1/chat/completions` and `/v1/models` endpoints
- Non-streaming **and** Server-Sent Events streaming
- `delta.reasoning_content` for `deepseek-reasoner` chain-of-thought
- System prompt folding (DeepSeek's web API has no `system` role)
- Multi-turn session continuity via `parent_message_id`
- Automatic 1-hour access-token refresh from a long-lived `userToken`
- Automatic `userToken` rotation via a browser extension — no restarts, no manual pasting
- Proof-of-work via DeepSeek's own WASM (correct and fast)
- Thread-safe PoW solver
- Optional shared-secret auth for the bridge itself
- Ban / risk-code detection with readable error messages
- `/health` endpoint with token age, session state, and expiry

---

## How It Works

The system has four parts. Data flows top to bottom:

```
    ┌──────────────────────────┐
    │  OpenAI SDK client       │
    │  (LangChain, Cursor,     │
    │   your own scripts)      │
    └────────────┬─────────────┘
                 │  HTTP /v1/chat/completions
                 ▼
    ┌──────────────────────────┐
    │  Python bridge           │   :8000
    │  ├─ TokenManager         │
    │  ├─ WASM PoW solver      │
    │  └─ SSE parser           │
    └────────────┬─────────────┘
                 │  HTTPS
                 ▼
    ┌──────────────────────────┐
    │  chat.deepseek.com       │
    │  /api/v0/...             │
    └──────────────────────────┘
                 ▲
                 │  userToken
                 │
    ┌────────────┴─────────────┐
    │  Browser extension       │
    │  (Chromium or Firefox)   │
    │  watches localStorage    │
    │  ["userToken"]           │
    │                          │
    │  POSTs to :9876/token    │
    └──────────────────────────┘

              :9876
    ┌──────────────────────────┐
    │  Token bridge (inside    │
    │  the Python process)     │
    │  receives extension push │
    └──────────────────────────┘
```

**In plain English:**

1. An OpenAI SDK client sends a standard `/v1/chat/completions` request to `http://127.0.0.1:8000`.
2. The Python bridge converts it into DeepSeek's internal `POST /api/v0/chat/completion` call — including solving a WASM proof-of-work challenge and attaching the current access token.
3. DeepSeek streams back SSE frames; the bridge parses them and re-emits them as OpenAI `chat.completion.chunk` events.
4. Meanwhile, a browser extension watches `localStorage["userToken"]` on `chat.deepseek.com` and POSTs changes to `http://127.0.0.1:9876/token`. The Python process writes the new token to disk and hot-swaps it into the running client within 5 seconds — no restart.


**Two-token model.** DeepSeek uses a two-tier token scheme:

| Token | Lifetime | Source | Purpose |
|---|---|---|---|
| `userToken` | Days–weeks | `localStorage` on `chat.deepseek.com` | Refresh token |
| Access token | 1 hour | `GET /api/v0/users/current` | API authorization header |

The bridge **auto-refreshes the access token** on a schedule and on 401. The extension **auto-rotates the `userToken`** whenever DeepSeek issues a new one. Together they make the system self-healing.

**Proof of work.** Every `/chat/completion` request must be signed with a `DeepSeekHashV1` PoW signature in the `x-ds-pow-response` header. The bridge runs DeepSeek's own `sha3_wasm_bg.wasm` in `wasmtime` — the exact same code the browser runs — so the signature is always correct.

---

## Requirements

- **Python** 3.10 or newer
- **A DeepSeek account** (free tier is fine)
- **A browser** — Chromium (Chrome/Edge/Brave/Opera) or Firefox 109+
- **Windows / Linux / macOS** — all work

Python packages are listed in `requirements.txt`. The only non-trivial one is `wasmtime`, which is a small, pure-wheel package.

---

## Quick Start

```bash
# 1. Clone or download the project
cd deepseek-bridge

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download DeepSeek's PoW WASM (one-time)
curl -o sha3_wasm_bg.wasm https://fe-static.deepseek.com/chat/static/sha3_wasm_bg.7b9ca65ddd.wasm

# 4. Create your .env
cp .env.example .env       # Linux/macOS
copy .env.example .env     # Windows

# 5. Start the server
python main.py