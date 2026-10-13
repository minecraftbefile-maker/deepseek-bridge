# DeepSeek Web → OpenAI Bridge

An OpenAI-compatible HTTP bridge for DeepSeek's web API. Point any OpenAI SDK client at `http://127.0.0.1:8000/v1` and it works — streaming, tool calls, sessions, and automatic token refresh included.

Built for local use. No paid API key required. No browser automation. Just a small Python server and a browser extension that keeps your `userToken`, device ID, and real browser fingerprint in sync.

---

## Table of Contents

- [What It Does](#what-it-does)
- [How It Works](#how-it-works)
- [Requirements](#requirements)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Browser Extension](#browser-extension)
- [Using the Bridge](#using-the-bridge)
- [Modes](#modes)
- [API Reference](#api-reference)
- [Troubleshooting](#troubleshooting)
- [FAQ](#faq)
- [Project Layout](#project-layout)
- [Security Notes](#security-notes)
- [License](#license)

---

## What It Does

DeepSeek's web app at `chat.deepseek.com` uses an internal API that the browser talks to directly. This project exposes that internal API as a standard **OpenAI-compatible endpoint**, so tools that already speak the OpenAI protocol can talk to DeepSeek for free.

**Features:**

- OpenAI `/v1/chat/completions` and `/v1/models` endpoints
- Non-streaming **and** Server-Sent Events streaming
- `delta.reasoning_content` for `deepseek-reasoner` chain-of-thought
- System prompt folding with strong authority framing
- Multi-turn session continuity via `parent_message_id`
- Automatic 1-hour access-token refresh from a long-lived `userToken`
- Automatic `userToken` rotation via a browser extension — no restarts, no manual pasting
- Proof-of-work via DeepSeek's own WASM (correct and fast, thread-safe)
- **Real browser fingerprint** — the exact User-Agent, Accept-Language, Sec-CH-UA, and cookie headers from your browser are replayed on every request
- **Agent mode** with JSON tool calls (`<tool_calls>[...]</tool_calls>`), parsed into standard OpenAI `tool_calls`
- **Stall recovery** — if the model announces an action without calling a tool, the bridge nudges it once
- **Empty-response recovery** — if DeepSeek returns nothing, the bridge asks for the final answer, then resets the session if that fails
- **Session size cap** — continuations beyond a threshold automatically fold into a fresh session
- `/api/fresh` endpoint to start a new chat programmatically
- Optional shared-secret auth for the bridge itself
- Ban / risk-code detection with readable error messages

---
## How It Works

The system has four parts. Data flows top to bottom.

```
    +-----------------------------+
    |  OpenAI SDK client          |
    |  (LangChain, Cursor,        |
    |   your own scripts)         |
    +--------------+--------------+
                   |
                   | HTTP /v1/chat/completions
                   v
    +-----------------------------+
    |  Python bridge       :8000  |
    |  - TokenManager             |
    |  - WASM PoW solver          |
    |  - SSE parser               |
    |  - Tool-call parser         |
    +--------------+--------------+
                   |
                   | HTTPS (with your browser's real fingerprint)
                   v
    +-----------------------------+
    |  chat.deepseek.com          |
    |  /api/v0/...                |
    +-----------------------------+
                   ^
                   | userToken + device ID + headers
                   |
    +--------------+--------------+
    |  Browser extension          |
    |  (Chromium or Firefox)      |
    |  watches localStorage,      |
    |  navigator.userAgent,       |
    |  fetch/XHR request headers  |
    |                             |
    |  POSTs to :9876/auth        |
    +-----------------------------+
                   |
                   v
    +-----------------------------+
    |  Token bridge         :9876 |
    |  (inside Python process)    |
    +-----------------------------+
```

**In plain English:**

1. An OpenAI SDK client sends a standard `/v1/chat/completions` request to `http://127.0.0.1:8000`.
2. The Python bridge converts it into DeepSeek's internal `POST /api/v0/chat/completion` call — including solving a WASM proof-of-work challenge and attaching the current access token.
3. DeepSeek streams back SSE frames; the bridge parses them and re-emits them as OpenAI `chat.completion.chunk` events.
4. Meanwhile, a browser extension watches `localStorage["userToken"]`, hooks the page's `fetch`/`XMLHttpRequest` calls to capture request headers, reads `navigator.userAgent` for browser identity, and POSTs everything to `http://127.0.0.1:9876/auth`. The Python process writes to disk and hot-swaps the values into the running client within a few seconds — no restart.
