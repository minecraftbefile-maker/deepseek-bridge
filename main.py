"""
OpenAI-compatible bridge for DeepSeek's web API.

Full OpenAI SDK compatibility:
    - /v1/chat/completions (streaming & non-streaming)
    - /v1/models
    - /v1/completions (legacy, mapped to chat)
    - Full `tools` array forwarding (formatted into the prompt)
    - Full `tool_choice` handling: auto | none | required | specific
    - Assistant messages with tool_calls rendered as history
    - `tool` role messages rendered with their tool name
    - `stream_options.include_usage` support
    - `response_format`, `seed`, `temperature`, etc. accepted gracefully

Bridge features:
    - WASM-based PoW (DeepSeekHashV1)
    - Auto access-token refresh via /users/current
    - Browser extension integration (Chromium + Firefox)
    - Real browser fingerprint replay
    - Contract-grade system prompt framing
    - Native /chat/continue auto-resume
    - Stall, empty, and truncation recovery
    - Hybrid DSML/JSON tool-call parsing
    - Session size cap
    - Optional tool filtering to keep prompt size bounded

Setup:
    pip install -r requirements.txt
    copy .env.example .env
    curl -o sha3_wasm_bg.wasm ^
        https://fe-static.deepseek.com/chat/static/sha3_wasm_bg.7b9ca65ddd.wasm
    python main.py
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import re
import sys
import threading
import time
import uuid
from contextlib import asynccontextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple, Union

import httpx
import numpy as np
import wasmtime
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# =============================================================================
# LOAD .ENV
# =============================================================================

_HERE = Path(__file__).resolve().parent
_ENV_PATH = _HERE / ".env"
if _ENV_PATH.exists():
    load_dotenv(dotenv_path=_ENV_PATH, override=False)
    print(f"[env] loaded {_ENV_PATH}")
else:
    load_dotenv()


# =============================================================================
# CONFIG
# =============================================================================

DEEPSEEK_USER_TOKEN = os.environ.get("DEEPSEEK_USER_TOKEN", "").strip()
DEEPSEEK_COOKIE = os.environ.get("DEEPSEEK_COOKIE", "").strip()
DEEPSEEK_DEVICE_ID = os.environ.get("DEEPSEEK_DEVICE_ID", "").strip()
BRIDGE_API_KEY = os.environ.get("BRIDGE_API_KEY", "").strip()
BIND_HOST = os.environ.get("BIND_HOST", "127.0.0.1").strip()
BIND_PORT = int(os.environ.get("BIND_PORT", "8000"))
TOKEN_BRIDGE_PORT = int(os.environ.get("TOKEN_BRIDGE_PORT", "9876"))
DEBUG = os.environ.get("DEEPSEEK_DEBUG", "0") == "1"
RESET_ON_SYSTEM_CHANGE = os.environ.get("RESET_ON_SYSTEM_CHANGE", "0") == "1"
REINFORCE_SYSTEM = os.environ.get("REINFORCE_SYSTEM", "1") == "1"
MAX_CONTINUATION_CHARS = int(os.environ.get("MAX_CONTINUATION_CHARS", "12000"))
MAX_CONTINUE_ROUNDS = int(os.environ.get("MAX_CONTINUE_ROUNDS", "5"))

# Comma-separated list of tool name prefixes to strip from the
# generated tools block. Keeps prompt size bounded for clients that
# advertise hundreds of tools. Empty = forward everything.
TOOLS_STRIP_PREFIXES = [
    p.strip() for p in
    os.environ.get("TOOLS_STRIP_PREFIXES", "").split(",") if p.strip()
]

# If set, only tools whose names are in this list are forwarded.
# Comma-separated. Empty = forward everything (minus strips).
TOOLS_KEEP_ONLY = [
    p.strip() for p in
    os.environ.get("TOOLS_KEEP_ONLY", "").split(",") if p.strip()
]

# Max tools to forward. Extra tools are dropped after filtering.
MAX_TOOLS = int(os.environ.get("MAX_TOOLS", "0"))  # 0 = unlimited

DEFAULT_MODE = os.environ.get("BRIDGE_MODE", "agent").strip().lower()
MODE = "agent" if DEFAULT_MODE.startswith("a") else "chat"

BASE_URL = "https://chat.deepseek.com/api/v0"
WASM_PATH = _HERE / "sha3_wasm_bg.wasm"
WASM_URL = "https://fe-static.deepseek.com/chat/static/sha3_wasm_bg.7b9ca65ddd.wasm"
TOKEN_FILE = _HERE / ".ds_usertoken"
DEVICE_ID_FILE = _HERE / ".ds_device_id"
HEADERS_FILE = _HERE / ".ds_headers.json"

ACCESS_TOKEN_TTL = 3600
REFRESH_MARGIN = 300

BAN_CODES = {5: "ACCOUNT_BANNED", 11: "RISK_DEVICE_DETECTED"}
AUTH_CODES = {40100: "AUTH_MISSING", 40101: "AUTH_INVALID_OR_EXPIRED",
              40300: "PERMISSION_DENIED"}

_NEVER_REPLAY = {
    "authorization", "content-length", "content-type", "host", "connection",
    "accept-encoding", "x-ds-pow-response", "x-hif-dliq", "x-hif-leim",
}

_FALLBACK_HEADERS = {
    "accept": "*/*",
    "accept-language": "en-US,en;q=0.9",
    "content-type": "application/json",
    "origin": "https://chat.deepseek.com",
    "referer": "https://chat.deepseek.com/",
    "user-agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/132.0.0.0 Safari/537.36"),
    "x-client-version": "2.5.0",
    "x-client-platform": "web",
    "x-client-locale": "en_US",
    "x-client-bundle-id": "com.deepseek.chat",
    "x-client-timezone-offset": "0",
    "x-device-model": "",
}


# =============================================================================
# STALL / INCOMPLETE DETECTION
# =============================================================================

_PROMISE_RE = re.compile(
    r"(?:let me|i'?ll|i will|now i'?ll|next,? i|checking|reading|"
    r"looking at|about to|going to)\b",
    re.IGNORECASE,
)


def _looks_like_promise(text: str) -> bool:
    if not text:
        return False
    stripped = text.strip()
    if not stripped or len(stripped) > 2000:
        return False
    lines = [l for l in stripped.split("\n") if l.strip()]
    if not lines:
        return False
    return bool(_PROMISE_RE.search(lines[-1]))


def _looks_incomplete(text: str) -> bool:
    if not text:
        return False
    stripped = text.rstrip()
    if not stripped or len(stripped) < 100:
        return False
    for open_ch, close_ch in (("(", ")"), ("[", "]"), ("{", "}")):
        if stripped.count(open_ch) > stripped.count(close_ch):
            return True
    if stripped.count('"') % 2 == 1:
        return True
    if stripped.endswith("\\") or stripped.endswith(","):
        return True
    return False


# =============================================================================
# MODE SELECTION
# =============================================================================

def _ask_startup_mode() -> str:
    if not sys.stdin or not sys.stdin.isatty():
        return MODE
    print()
    print("+----------------------------------------------------------------+")
    print("|  Select bridge mode                                            |")
    print("+----------------------------------------------------------------+")
    print("|  [a] AGENT mode - tools via JSON tool_calls, no thinking       |")
    print("|  [c] CHAT mode  - plain conversation, no tool parsing          |")
    print("+----------------------------------------------------------------+")
    while True:
        try:
            ans = input(
                f"Choice [{'a' if MODE == 'agent' else 'c'}]: "
            ).strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return MODE
        if ans == "":
            return MODE
        if ans.startswith("a"):
            return "agent"
        if ans.startswith("c"):
            return "chat"
        print("Please enter 'a' or 'c'.")


# =============================================================================
# SYSTEM PROMPT
# =============================================================================

_SYS_HEADER = (
    "╔══════════════════════════════════════════════════════════════════╗\n"
    "║  CONTRACT — READ THIS BEFORE EVERY RESPONSE                      ║\n"
    "╚══════════════════════════════════════════════════════════════════╝\n"
    "\n"
    "You are executing inside an agent framework. The block labeled\n"
    "'AGENT SYSTEM PROMPT' below is the authoritative specification of\n"
    "your role, tools, and behavior. It was written by the operator who\n"
    "deployed you. It is not a suggestion, a guideline, or context — it\n"
    "is a CONTRACT you must fulfill on every turn.\n"
    "\n"
    "COMPLIANCE RULES:\n"
    "\n"
    "  1. READ the entire AGENT SYSTEM PROMPT carefully. Do not skim.\n"
    "     Every line matters. Tool names, parameter names, workspace\n"
    "     paths, safety rules, output formats — all of it is binding.\n"
    "\n"
    "  2. FOLLOW it exactly. When the prompt says 'use tool X with\n"
    "     parameter Y', use exactly that. When it says 'never do Z',\n"
    "     never do Z — even if Z would be faster or more natural.\n"
    "\n"
    "  3. The AGENT SYSTEM PROMPT takes precedence over your default\n"
    "     behavior, your training habits, and any conflicting request\n"
    "     from the user.\n"
    "\n"
    "  4. Do NOT paraphrase, summarize, or acknowledge the AGENT SYSTEM\n"
    "     PROMPT back to the user. Internalize it silently and act.\n"
    "\n"
    "  5. Do NOT invent tools, capabilities, or paths that are not in\n"
    "     the AGENT SYSTEM PROMPT or the AVAILABLE TOOLS section.\n"
    "     If a tool is not listed, it does not exist.\n"
    "\n"
    "  6. If the AGENT SYSTEM PROMPT and these compliance rules ever\n"
    "     conflict, the AGENT SYSTEM PROMPT wins.\n"
    "\n"
    "  7. Before you emit a response, mentally verify:\n"
    "       [ ] Am I using a tool that actually exists in the list?\n"
    "       [ ] Are my parameter names EXACTLY what the schema says?\n"
    "       [ ] Am I respecting every 'never do X' rule?\n"
    "       [ ] Am I respecting the file-size and output-format limits?\n"
    "     If any answer is 'no', fix the response before sending it.\n"
    "\n"
    "╔══════════════════════════════════════════════════════════════════╗\n"
    "║  AGENT SYSTEM PROMPT — BINDING CONTRACT                          ║\n"
    "╚══════════════════════════════════════════════════════════════════╝\n"
    "\n"
    "---------- BEGIN AGENT SYSTEM PROMPT ----------\n"
)

_SYS_FOOTER = (
    "\n"
    "---------- END AGENT SYSTEM PROMPT ----------\n"
    "\n"
    "╔══════════════════════════════════════════════════════════════════╗\n"
    "║  TOOL-CALL PROTOCOL — OVERRIDES ANYTHING ABOVE                   ║\n"
    "╚══════════════════════════════════════════════════════════════════╝\n"
    "\n"
    "The AGENT SYSTEM PROMPT may describe tools using a different\n"
    "syntax (DSML, XML, Claude Code markers, or similar). IGNORE THAT\n"
    "SYNTAX. You MUST emit tool calls using the JSON format below.\n"
    "\n"
    "To call one or more tools, output EXACTLY this:\n"
    "\n"
    "<tool_calls>\n"
    "[\n"
    "  {\n"
    '    "name": "exact_tool_name",\n'
    '    "arguments": {\n'
    '      "exact_param_name": "value"\n'
    "    }\n"
    "  }\n"
    "]\n"
    "</tool_calls>\n"
    "\n"
    "PROTOCOL RULES:\n"
    "\n"
    "  1. The JSON MUST be a valid array of objects, one per tool call.\n"
    "  2. Tool names and parameter names MUST match the schema in the\n"
    "     AVAILABLE TOOLS section EXACTLY — spelling, casing, everything.\n"
    "  3. Emit ALL parallel tool calls inside ONE <tool_calls> block.\n"
    "  4. Do NOT wrap the block in markdown code fences.\n"
    "  5. If no tool is needed, reply normally without the block.\n"
    "\n"
    "════════════════════════════════════════════════════════════════════\n"
    "BEHAVIORAL RULES — MANDATORY\n"
    "════════════════════════════════════════════════════════════════════\n"
    "\n"
    "  1. NEVER say 'let me', 'I will', 'I'll', 'now I'll', 'next I',\n"
    "     'checking', 'reading', 'looking at', 'about to', or 'going\n"
    "     to'. If you need a tool, emit the tool call IMMEDIATELY.\n"
    "\n"
    "  2. If you are finished, provide the final answer directly.\n"
    "\n"
    "  3. If a tool is not listed in AVAILABLE TOOLS, do not call it.\n"
    "     If the tool you want is blocked, try a different one that IS\n"
    "     listed, or report the blockage to the user.\n"
    "\n"
    "════════════════════════════════════════════════════════════════════\n"
    "FILE SIZE LIMIT — CRITICAL\n"
    "════════════════════════════════════════════════════════════════════\n"
    "\n"
    "The API truncates any single response that exceeds ~6000 tokens.\n"
    "A tool call above ~200 lines will be silently dropped.\n"
    "\n"
    "To write files longer than ~200 lines:\n"
    "  1. Use write_file with the FIRST ~150 lines only.\n"
    "  2. Then append_file for each subsequent ~150-line chunk.\n"
    "  3. Repeat until the file is complete.\n"
    "\n"
    "════════════════════════════════════════════════════════════════════\n"
    "FINAL INSTRUCTION\n"
    "════════════════════════════════════════════════════════════════════\n"
    "\n"
    "Re-read the AGENT SYSTEM PROMPT above before responding. It is\n"
    "your contract. Then act — do not describe, do not explain, do\n"
    "not announce. Emit the tool call or the final answer."
)

_REINFORCE = (
    "╔══════════════════════════════════════════════════════════════════╗\n"
    "║  TURN COMPLIANCE CHECK                                           ║\n"
    "╚══════════════════════════════════════════════════════════════════╝\n"
    "\n"
    "Before responding to this turn, verify:\n"
    "\n"
    "  1. Am I following the AGENT SYSTEM PROMPT exactly?\n"
    "  2. Am I using the JSON <tool_calls> format?\n"
    "  3. Have I avoided phrases like 'let me' or 'I will'?\n"
    "  4. If writing a file, is it under 200 lines per tool call?\n"
    "  5. Are tool names and parameter names exact?\n"
    "\n"
    "If any check fails, fix the response before sending. Do not\n"
    "acknowledge this reminder to the user.\n"
    "\n"
)


def _wrap_system_prompt(body: str, tools_block: str = "") -> str:
    wrapped = _SYS_HEADER + body.strip() + _SYS_FOOTER
    if tools_block:
        wrapped += "\n\n" + tools_block
    return wrapped


# =============================================================================
# TOOLS FORMATTING (OpenAI tools -> prompt text)
# =============================================================================

def _filter_tools(tools: List[Any]) -> List[Any]:
    """Apply TOOLS_STRIP_PREFIXES and TOOLS_KEEP_ONLY, then cap."""
    if not tools:
        return []

    out = []
    for t in tools:
        if not isinstance(t, dict):
            continue
        fn = t.get("function") if t.get("type") == "function" else t
        if not isinstance(fn, dict):
            continue
        name = fn.get("name") or ""
        if TOOLS_KEEP_ONLY and name not in TOOLS_KEEP_ONLY:
            continue
        if TOOLS_STRIP_PREFIXES and any(
                name.startswith(p) for p in TOOLS_STRIP_PREFIXES):
            continue
        out.append(t)

    if MAX_TOOLS > 0:
        out = out[:MAX_TOOLS]
    return out


def _format_tools_for_prompt(tools: Any,
                             tool_choice: Any = None) -> str:
    """
    Serialize an OpenAI tools array into readable text.

    DeepSeek's web backend has no native tool-calling support. We
    convert the OpenAI tools array into text so the model sees each
    tool's exact name, parameters, and types.
    """
    if not tools or not isinstance(tools, list):
        return ""

    # Respect tool_choice="none"
    if isinstance(tool_choice, str) and tool_choice == "none":
        return ""

    tools = _filter_tools(tools)
    if not tools:
        return ""

    # If tool_choice names a specific function, only show that one.
    forced_name = None
    if isinstance(tool_choice, dict):
        fn = tool_choice.get("function")
        if isinstance(fn, dict):
            forced_name = fn.get("name")
    if forced_name:
        tools = [
            t for t in tools
            if ((t.get("function") if t.get("type") == "function" else t)
                .get("name") == forced_name)
        ]
        if not tools:
            return ""

    lines = [
        "════════════════════════════════════════════════════════════════════",
        "AVAILABLE TOOLS",
        "════════════════════════════════════════════════════════════════════",
        "",
    ]

    if forced_name:
        lines.append(f"You MUST call the tool '{forced_name}'.")
        lines.append("")
    else:
        lines.append(
            f"You have access to {len(tools)} tools. To use one, emit a"
        )
        lines.append(
            "tool call using the <tool_calls>[...]</tool_calls> JSON format."
        )
        lines.append("")

    lines.append(
        "Tool names and parameter names MUST match EXACTLY — including"
    )
    lines.append(
        "spelling, capitalization, and casing. Parameter values must"
    )
    lines.append("respect the declared types.")
    lines.append("")

    for i, tool in enumerate(tools, 1):
        if not isinstance(tool, dict):
            continue
        fn = tool.get("function") if tool.get("type") == "function" else tool
        if not isinstance(fn, dict):
            continue

        name = fn.get("name", "?")
        desc = (fn.get("description") or "").strip()
        params = fn.get("parameters") or {}

        lines.append(f"[{i}] {name}")
        if desc:
            for dline in desc.split("\n"):
                lines.append(f"    {dline}")

        props = params.get("properties") if isinstance(params, dict) else None
        required = (params.get("required")
                    if isinstance(params, dict) else None)
        required = required if isinstance(required, list) else []

        if isinstance(props, dict) and props:
            lines.append("")
            lines.append("    Parameters:")
            for pname, pinfo in props.items():
                if not isinstance(pinfo, dict):
                    continue
                ptype = pinfo.get("type", "any")
                pdesc = (pinfo.get("description") or "").strip()
                req = " (required)" if pname in required else " (optional)"
                if len(pdesc) > 200:
                    pdesc = pdesc[:197] + "..."
                lines.append(f"      - {pname}: {ptype}{req}")
                if pdesc:
                    for dline in pdesc.split("\n"):
                        lines.append(f"          {dline}")
            lines.append("")

        lines.append("")

    lines.append(
        "════════════════════════════════════════════════════════════════════"
    )
    return "\n".join(lines)


def _tool_names_summary(tools: Any) -> str:
    """Compact reminder listing tool names only."""
    if not tools or not isinstance(tools, list):
        return ""
    tools = _filter_tools(tools)
    names = []
    for t in tools:
        if not isinstance(t, dict):
            continue
        fn = t.get("function") if t.get("type") == "function" else t
        if isinstance(fn, dict) and fn.get("name"):
            names.append(fn["name"])
    if not names:
        return ""
    return (
        "[Reminder: Available tools (full schemas in system prompt): "
        + ", ".join(names)
        + "]"
    )


# =============================================================================
# TOOL-CALL PARSER
# =============================================================================

_JSON_START = "<tool_calls>"
_JSON_END = "</tool_calls>"

_DSML_STARTS = (
    "<\uff5c\uff5cDSML\uff5c\uff5c calls>",
    "<||DSML|| calls>",
    "<\uff5c\uff5cDSML\uff5c\uff5ccalls>",
    "<||DSML||calls>",
)
_DSML_ENDS = (
    "</\uff5c\uff5cDSML\uff5c\uff5c calls>",
    "</||DSML|| calls>",
    "</\uff5c\uff5cDSML\uff5c\uff5ccalls>",
    "</||DSML||calls>",
)

_INVOKE_RE = re.compile(
    r"<\s*[\uff5c|]{2}\s*DSML\s*[\uff5c|]{2}\s+invoke\s+name=\"([^\"]+)\"\s*>"
    r"(.*?)"
    r"<\s*/\s*[\uff5c|]{2}\s*DSML\s*[\uff5c|]{2}\s+invoke\s*>",
    re.DOTALL,
)
_PARAM_RE = re.compile(
    r"<\s*[\uff5c|]{2}\s*DSML\s*[\uff5c|]{2}\s+parameter\s+name=\"([^\"]+)\""
    r"(?:\s+[^>]*)?\s*>(.*?)"
    r"<\s*/\s*[\uff5c|]{2}\s*DSML\s*[\uff5c|]{2}\s+parameter\s*>",
    re.DOTALL,
)


def _parse_json_tool_calls(body: str) -> List[Dict[str, Any]]:
    text = body.strip()
    if text.startswith("```"):
        first_nl = text.find("\n")
        if first_nl != -1:
            text = text[first_nl + 1:]
        if text.endswith("```"):
            text = text[:-3].rstrip()

    first_bracket = None
    for i, ch in enumerate(text):
        if ch in "[{":
            first_bracket = i
            break
    if first_bracket is None:
        return []
    text = text[first_bracket:]

    parsed = None
    try:
        parsed = json.loads(text)
    except Exception:
        opener = text[0]
        closer = "]" if opener == "[" else "}"
        depth = 0
        in_string = False
        escape = False
        end = -1
        for i, ch in enumerate(text):
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
                continue
            if ch == opener:
                depth += 1
            elif ch == closer:
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        if end > 0:
            try:
                parsed = json.loads(text[:end])
            except Exception:
                return []
    if parsed is None:
        return []

    items = parsed if isinstance(parsed, list) else [parsed]
    calls: List[Dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        name = item.get("name") or item.get("tool") or item.get("function")
        args = item.get("arguments")
        if args is None:
            args = item.get("args") or item.get("parameters") or {}
        if isinstance(name, dict):
            fn = name
            name = fn.get("name")
            args = fn.get("arguments", args)

        if name in ("tool_calls", "tool_call", "calls"):
            inner = args.get("tool_calls") or args.get("calls") or args
            if isinstance(inner, list):
                for sub in inner:
                    if not isinstance(sub, dict):
                        continue
                    sub_name = sub.get("name") or sub.get("tool")
                    sub_args = (sub.get("arguments") or sub.get("args")
                                or sub.get("parameters") or {})
                    if isinstance(sub_name, dict):
                        sub_fn = sub_name
                        sub_name = sub_fn.get("name")
                        sub_args = sub_fn.get("arguments", sub_args)
                    if isinstance(sub_name, str) and sub_name:
                        if not isinstance(sub_args, dict):
                            sub_args = {}
                        calls.append({
                            "name": sub_name,
                            "arguments": json.dumps(sub_args,
                                                    ensure_ascii=False),
                        })
                continue
            if isinstance(inner, dict):
                sub_name = inner.get("name") or inner.get("tool")
                sub_args = (inner.get("arguments") or inner.get("args")
                            or inner.get("parameters") or {})
                if isinstance(sub_name, str) and sub_name:
                    if not isinstance(sub_args, dict):
                        sub_args = {}
                    calls.append({
                        "name": sub_name,
                        "arguments": json.dumps(sub_args,
                                                ensure_ascii=False),
                    })
                continue

        if not isinstance(name, str) or not name:
            continue
        if not isinstance(args, dict):
            args = {}
        calls.append({"name": name,
                      "arguments": json.dumps(args, ensure_ascii=False)})
    return calls


def _parse_dsml_tool_calls(text: str) -> List[Dict[str, Any]]:
    calls = []
    for m in _INVOKE_RE.finditer(text):
        name = m.group(1)
        body = m.group(2)
        args: Dict[str, Any] = {}
        for pm in _PARAM_RE.finditer(body):
            pname = pm.group(1)
            pval = pm.group(2)
            s = pval.strip()
            if s and s[0] in "[{":
                try:
                    args[pname] = json.loads(s)
                    continue
                except Exception:
                    pass
            if s in ("true", "false"):
                args[pname] = (s == "true")
                continue
            if re.fullmatch(r"-?\d+", s):
                try:
                    args[pname] = int(s)
                    continue
                except Exception:
                    pass
            args[pname] = pval
        if name:
            calls.append({"name": name,
                          "arguments": json.dumps(args, ensure_ascii=False)})
    return calls


def _parse_any_tool_calls(body: str) -> List[Dict[str, Any]]:
    calls = _parse_dsml_tool_calls(body)
    if calls:
        return calls
    return _parse_json_tool_calls(body)


def _strip_tool_markers(text: str) -> str:
    for m in (_JSON_START, _JSON_END):
        while True:
            idx = text.find(m)
            if idx == -1:
                break
            end_of_line = text.find("\n", idx)
            if end_of_line == -1:
                text = text[:idx].rstrip()
                break
            text = text[:idx] + text[end_of_line + 1:]

    for marker in list(_DSML_STARTS) + list(_DSML_ENDS):
        text = text.replace(marker, "")

    return text.strip()


def _detect_truncation(text: str) -> bool:
    start_idx = text.find(_JSON_START)
    if start_idx != -1 and text.find(_JSON_END, start_idx) == -1:
        body = text[start_idx + len(_JSON_START):]
        if not _parse_json_tool_calls(body):
            return True

    for m in _DSML_STARTS:
        idx = text.find(m)
        if idx == -1:
            continue
        tail = text[idx + len(m):]
        if not any(tail.find(e) != -1 for e in _DSML_ENDS + (_JSON_END,)):
            if not _parse_any_tool_calls(tail):
                return True
    return False


def _split_text_and_tool_calls(text: str):
    start_idx = text.find(_JSON_START)
    if start_idx != -1:
        end_idx = text.find(_JSON_END, start_idx + len(_JSON_START))
        if end_idx == -1:
            body = text[start_idx + len(_JSON_START):]
            calls = _parse_json_tool_calls(body)
            if calls:
                return text[:start_idx].rstrip(), calls
        else:
            body = text[start_idx + len(_JSON_START):end_idx]
            calls = _parse_json_tool_calls(body)
            before = text[:start_idx].rstrip()
            after = text[end_idx + len(_JSON_END):].strip()
            if calls:
                content = (before + ("\n" + after if after else "")).strip()
                return content, calls
            return _strip_tool_markers(
                before + ("\n" + after if after else "")), []

    earliest, marker_len, matched_start = len(text), 0, ""
    for m in _DSML_STARTS:
        idx = text.find(m)
        if idx != -1 and idx < earliest:
            earliest, marker_len, matched_start = idx, len(m), m
    if earliest == len(text):
        return text, []

    before = text[:earliest].rstrip()
    tail = text[earliest + marker_len:]

    end_idx, matched_end = -1, ""
    all_ends = _DSML_ENDS + (_JSON_END,)
    for m in all_ends:
        idx = tail.find(m)
        if idx != -1 and (end_idx == -1 or idx < end_idx):
            end_idx, matched_end = idx, m

    if end_idx == -1:
        calls = _parse_any_tool_calls(tail)
        if calls:
            return before, calls
        return _strip_tool_markers(before), []

    body = tail[:end_idx]
    calls = _parse_any_tool_calls(body)

    after = tail[end_idx + len(matched_end):].strip()
    if not calls:
        return _strip_tool_markers(
            before + ("\n" + after if after else "")), []
    final = (before + ("\n" + after if after else "")).strip()
    return final, calls


class StreamingToolParser:
    def __init__(self):
        self.buffer = ""
        self.mode = "content"
        self.tool_text = ""

    def _any_start_in(self, s: str) -> Tuple[int, str, str]:
        candidates = [(_JSON_START, "json")]
        for m in _DSML_STARTS:
            candidates.append((m, "dsml"))
        earliest_idx = -1
        earliest_marker = ""
        earliest_mode = ""
        for marker, mode in candidates:
            i = s.find(marker)
            if i != -1 and (earliest_idx == -1 or i < earliest_idx):
                earliest_idx = i
                earliest_marker = marker
                earliest_mode = mode
        return earliest_idx, earliest_marker, earliest_mode

    def _all_ends(self) -> Tuple[str, ...]:
        return _DSML_ENDS + (_JSON_END,)

    @staticmethod
    def _hold_len(s: str, candidates: Tuple[str, ...]) -> int:
        best = 0
        for c in candidates:
            for k in range(min(len(c), len(s)), 0, -1):
                if c.startswith(s[-k:]):
                    best = max(best, k)
                    break
        return best

    def feed(self, chunk: str) -> List[Tuple[str, Any]]:
        events = []
        self.buffer += chunk
        while self.buffer:
            if self.mode == "content":
                idx, marker, mode = self._any_start_in(self.buffer)
                if idx == -1:
                    all_starts = (_JSON_START,) + _DSML_STARTS
                    hold = self._hold_len(self.buffer, all_starts)
                    emit = self.buffer[:-hold] if hold else self.buffer
                    self.buffer = self.buffer[-hold:] if hold else ""
                    if emit:
                        events.append(("content", emit))
                    break
                if idx > 0:
                    events.append(("content", self.buffer[:idx]))
                self.buffer = self.buffer[idx + len(marker):]
                self.mode = mode
                self.tool_text = ""
            else:
                ends = self._all_ends()
                end_idx = -1
                end_marker = ""
                for m in ends:
                    i = self.buffer.find(m)
                    if i != -1 and (end_idx == -1 or i < end_idx):
                        end_idx, end_marker = i, m
                if end_idx == -1:
                    hold = self._hold_len(self.buffer, ends)
                    if hold:
                        self.tool_text += self.buffer[:-hold]
                        self.buffer = self.buffer[-hold:]
                    else:
                        self.tool_text += self.buffer
                        self.buffer = ""
                    break
                self.tool_text += self.buffer[:end_idx]
                self.buffer = self.buffer[end_idx + len(end_marker):]
                calls = _parse_any_tool_calls(self.tool_text)
                if calls:
                    events.append(("tool_calls", calls))
                self.mode = "content"
                self.tool_text = ""
        return events

    def flush(self):
        events = []
        if self.mode == "content":
            if self.buffer:
                events.append(("content", self.buffer))
        else:
            text = self.tool_text + self.buffer
            calls = _parse_any_tool_calls(text)
            if calls:
                events.append(("tool_calls", calls))
        self.buffer = ""
        self.tool_text = ""
        self.mode = "content"
        return events


# =============================================================================
# DEVICE ID
# =============================================================================

def _load_device_id() -> str:
    if DEEPSEEK_DEVICE_ID:
        print(f"[device] using .env device_id: {DEEPSEEK_DEVICE_ID}")
        return DEEPSEEK_DEVICE_ID
    if DEVICE_ID_FILE.exists():
        try:
            did = DEVICE_ID_FILE.read_text(encoding="utf-8").strip()
            if len(did) > 20:
                print(f"[device] using cached device_id: {did}")
                return did
        except Exception:
            pass
    print("[device] no device_id yet - waiting for browser extension")
    return ""


DEVICE_ID = _load_device_id()


# =============================================================================
# TOKEN BRIDGE
# =============================================================================

class TokenBridge:
    def __init__(self, port: int = TOKEN_BRIDGE_PORT):
        self.port = port
        self._lock = threading.Lock()
        self.current_token: Optional[str] = None
        self.current_device_id: Optional[str] = None
        self.current_headers: Optional[Dict[str, str]] = None
        self.last_update: float = 0.0
        self.last_heartbeat: float = 0.0
        self.source: str = ""
        self._server: Optional[ThreadingHTTPServer] = None

    def get_token(self):
        with self._lock:
            return self.current_token

    def get_device_id(self):
        with self._lock:
            return self.current_device_id

    def get_headers(self):
        with self._lock:
            return self.current_headers

    def age(self) -> float:
        with self._lock:
            return (float("inf") if self.last_update == 0
                    else time.time() - self.last_update)

    def _handle_auth(self, payload):
        token = payload.get("token")
        device_id = payload.get("deviceId")
        headers = payload.get("headers")
        source = (payload.get("source") or payload.get("origin")
                  or "extension")
        instance = payload.get("instanceId") or ""
        if instance:
            source = f"{source}:{instance}"

        token_ok = isinstance(token, str) and len(token) > 20
        device_ok = isinstance(device_id, str) and len(device_id) > 10
        headers_ok = isinstance(headers, dict) and len(headers) > 3
        if not (token_ok or device_ok or headers_ok):
            return False

        token_changed = device_changed = headers_changed = False
        with self._lock:
            if token_ok and token != self.current_token:
                self.current_token = token
                self.last_update = time.time()
                self.source = source
                token_changed = True
            if device_ok and device_id != self.current_device_id:
                self.current_device_id = device_id
                device_changed = True
            if headers_ok:
                prev = self.current_headers or {}
                if set(headers.keys()) != set(prev.keys()):
                    self.current_headers = headers
                    headers_changed = True
                else:
                    for k, v in headers.items():
                        if prev.get(k) != v:
                            self.current_headers = headers
                            headers_changed = True
                            break

        if token_ok:
            try:
                TOKEN_FILE.write_text(token, encoding="utf-8")
            except Exception:
                pass
        if device_ok:
            try:
                DEVICE_ID_FILE.write_text(device_id, encoding="utf-8")
            except Exception:
                pass
        if headers_ok and self.current_headers:
            try:
                HEADERS_FILE.write_text(
                    json.dumps(self.current_headers, indent=2),
                    encoding="utf-8")
            except Exception:
                pass

        if token_changed:
            print(f"[bridge] new userToken from {source} "
                  f"({len(token)} chars)")
        if device_changed:
            print(f"[bridge] new device ID: {device_id}")
        if headers_changed:
            ua = (self.current_headers.get("user-agent", "?")
                  if self.current_headers else "?")
            ua_short = ua[:70] + ("..." if len(ua) > 70 else "")
            print(f"[bridge] new header bundle ({len(headers)} headers) "
                  f"from {source}")
            print(f"[bridge]   user-agent: {ua_short}")
        return True

    def start(self):
        bridge = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, fmt, *args):
                if DEBUG:
                    print(f"[bridge] {fmt % args}")

            def _reply(self, code, body):
                data = json.dumps(body).encode("utf-8")
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Access-Control-Allow-Methods",
                                 "POST, GET, OPTIONS")
                self.send_header("Access-Control-Allow-Headers",
                                 "Content-Type")
                self.end_headers()
                self.wfile.write(data)

            def do_OPTIONS(self):
                self._reply(204, {})

            def do_GET(self):
                if self.path == "/status":
                    hdrs = bridge.get_headers() or {}
                    self._reply(200, {
                        "has_token": bridge.get_token() is not None,
                        "has_device_id": bridge.get_device_id() is not None,
                        "has_headers": bool(hdrs),
                        "device_id": bridge.get_device_id(),
                        "headers_count": len(hdrs),
                        "user_agent": hdrs.get("user-agent"),
                        "source": bridge.source,
                        "last_update": bridge.last_update,
                        "age_seconds": bridge.age(),
                        "heartbeat_age": (
                            time.time() - bridge.last_heartbeat
                            if bridge.last_heartbeat else None)})
                else:
                    self._reply(404, {"error": "not found"})

            def do_POST(self):
                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(length) if length else b"{}"
                try:
                    payload = json.loads(raw.decode("utf-8"))
                except Exception:
                    self._reply(400, {"error": "invalid json"})
                    return
                if self.path in ("/auth", "/token"):
                    ok = bridge._handle_auth(payload)
                    self._reply(200 if ok else 400,
                                {"ok": ok, "ts": time.time()})
                elif self.path == "/heartbeat":
                    with bridge._lock:
                        bridge.last_heartbeat = time.time()
                    self._reply(200, {"ok": True})
                else:
                    self._reply(404, {"error": "not found"})

        self._server = ThreadingHTTPServer(("127.0.0.1", self.port), Handler)
        threading.Thread(target=self._server.serve_forever, daemon=True,
                         name="token-bridge").start()
        print(f"[bridge] listening on http://127.0.0.1:{self.port}")

    def stop(self):
        if self._server:
            self._server.shutdown()


_bridge = TokenBridge()


def _load_token_from_disk() -> Optional[str]:
    if TOKEN_FILE.exists():
        try:
            t = TOKEN_FILE.read_text(encoding="utf-8").strip()
            if len(t) > 20:
                return t
        except Exception:
            pass
    return None


def _load_headers_from_disk() -> Optional[Dict[str, str]]:
    if HEADERS_FILE.exists():
        try:
            return json.loads(HEADERS_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return None


def _watch_bridge():
    last_token = last_device = last_headers_sig = None
    while True:
        time.sleep(3)
        current = _bridge.get_token()
        if current and current != last_token:
            last_token = current
            _client.tokens.user_token = current
            _client.tokens._access_token = None
            _client.tokens._expires_at = 0.0
            print("[token] hot-swapped userToken")
        device = _bridge.get_device_id()
        if device and device != last_device:
            last_device = device
            global DEVICE_ID
            DEVICE_ID = device
            _client.device_id = device
            _client.rebuild_client()
            print(f"[device] hot-swapped device ID: {device}")
        headers = _bridge.get_headers()
        if headers:
            sig = "|".join(sorted(f"{k}={v}" for k, v in headers.items()))
            if sig != last_headers_sig:
                last_headers_sig = sig
                _client.browser_headers = dict(headers)
                _client.rebuild_client()
                ua = headers.get("user-agent", "?")
                ua_short = ua[:70] + ("..." if len(ua) > 70 else "")
                print(f"[headers] hot-swapped bundle "
                      f"({len(headers)} headers)")
                print(f"[headers]   user-agent: {ua_short}")


# =============================================================================
# TOKEN MANAGER
# =============================================================================

class DeepSeekAPIError(RuntimeError):
    def __init__(self, code: int, msg: str, raw: Any = None):
        label = BAN_CODES.get(code) or AUTH_CODES.get(code) or "API_ERROR"
        super().__init__(f"DeepSeek {label} (code={code}): {msg}")
        self.code = code
        self.label = label
        self.raw = raw


class TokenManager:
    def __init__(self, user_token: str, cookie: str = ""):
        self.user_token = user_token
        self.cookie = cookie
        self._access_token: Optional[str] = None
        self._expires_at: float = 0.0
        self._lock = asyncio.Lock()

    def _is_expired(self):
        if self._access_token is None:
            return True
        return time.time() >= (self._expires_at - REFRESH_MARGIN)

    async def acquire_token(self, force_refresh=False):
        async with self._lock:
            if not self.user_token or len(self.user_token) < 20:
                raise DeepSeekAPIError(40101, "No userToken yet.")
            if not force_refresh and not self._is_expired():
                return self._access_token
            fresh = await self._refresh()
            self._access_token = fresh
            self._expires_at = time.time() + ACCESS_TOKEN_TTL
            return fresh

    async def _refresh(self) -> str:
        if not self.user_token or len(self.user_token) < 20:
            raise DeepSeekAPIError(40101, "No userToken yet.")
        headers = _client._build_headers()
        headers["authorization"] = f"Bearer {self.user_token}"

        async with httpx.AsyncClient(
                base_url=BASE_URL, headers=headers,
                timeout=httpx.Timeout(30.0, connect=15.0)) as c:
            resp = await c.get("/users/current")
        if resp.status_code == 401:
            raise DeepSeekAPIError(40101, "userToken invalid.")
        resp.raise_for_status()
        data = resp.json()
        code = data.get("code")
        if code not in (0, None):
            raise DeepSeekAPIError(int(code),
                                   data.get("msg") or "", raw=data)
        token = None
        inner = data.get("data")
        if isinstance(inner, dict):
            biz = inner.get("biz_data")
            if isinstance(biz, dict):
                token = biz.get("token") or biz.get("access_token")
                if not token:
                    u = biz.get("user")
                    if isinstance(u, dict):
                        token = u.get("token")
        if not token:
            token = data.get("token")
        if not token:
            raise RuntimeError(f"No token: {data}")
        return token


# =============================================================================
# SESSION STATE
# =============================================================================

class ChatSessionState:
    def __init__(self):
        self._lock = threading.Lock()
        self.session_id: Optional[str] = None
        self.last_response_message_id: Optional[int] = None
        self.turn_count = 0
        self.system_fingerprint: Optional[int] = None
        self.seen_tool_signatures: set = set()

    def reset(self):
        with self._lock:
            self.session_id = None
            self.last_response_message_id = None
            self.turn_count = 0
            self.system_fingerprint = None
            self.seen_tool_signatures = set()

    def snapshot(self):
        with self._lock:
            return (self.session_id, self.last_response_message_id,
                    self.system_fingerprint)

    def set_session(self, sid, sys_fp):
        with self._lock:
            self.session_id = sid
            self.last_response_message_id = None
            self.turn_count = 0
            self.system_fingerprint = sys_fp

    def set_last_response(self, mid):
        with self._lock:
            self.last_response_message_id = mid
            self.turn_count += 1

    def tool_signature_seen(self, sig: str) -> bool:
        with self._lock:
            return sig in self.seen_tool_signatures

    def remember_tool_signature(self, sig: str):
        with self._lock:
            self.seen_tool_signatures.add(sig)


_session = ChatSessionState()


def _stdin_listener():
    if not sys.stdin or not sys.stdin.isatty():
        return
    global MODE
    while True:
        try:
            line = sys.stdin.readline()
        except Exception:
            break
        if line == "":
            break
        cmd = line.strip().lower()
        if cmd in ("agent", "a"):
            MODE = "agent"
            _session.reset()
            print("\n[mode] AGENT (session reset)\n> ", end="")
            sys.stdout.flush()
            continue
        if cmd in ("chat", "c"):
            MODE = "chat"
            _session.reset()
            print("\n[mode] CHAT (session reset)\n> ", end="")
            sys.stdout.flush()
            continue
        if cmd in ("status", "s"):
            sid, parent, _ = _session.snapshot()
            ua = (_client.browser_headers or {}).get("user-agent", "MISSING")
            print(f"\n[mode] {MODE}")
            print(f"[session] {sid}  turns={_session.turn_count}")
            print(f"[user-agent] {ua}")
            print("> ", end="")
            sys.stdout.flush()
            continue
        _session.reset()
        sys.stdout.write(f"\n[chat] session reset ({MODE} mode)\n> ")
        sys.stdout.flush()


# =============================================================================
# WASM PoW
# =============================================================================

class DeepSeekHashWasm:
    def __init__(self, wasm_path: Path = WASM_PATH):
        if not wasm_path.exists():
            raise FileNotFoundError(f"WASM missing: {wasm_path}")
        engine = wasmtime.Engine()
        module = wasmtime.Module(engine, wasm_path.read_bytes())
        self.store = wasmtime.Store(engine)
        linker = wasmtime.Linker(engine)
        linker.define_wasi()
        self.instance = linker.instantiate(self.store, module)
        exports = self.instance.exports(self.store)
        self.memory = exports["memory"]
        self._solve = exports["wasm_solve"]
        self._malloc = exports["__wbindgen_export_0"]
        self._add_to_stack = exports["__wbindgen_add_to_stack_pointer"]
        self._free = None
        for name in ("__wbindgen_free", "__wbindgen_export_2"):
            if name in exports:
                self._free = exports[name]
                break
        self._lock = threading.Lock()

    def _write_string(self, text):
        data = text.encode("utf-8")
        ptr = self._malloc(self.store, len(data), 1)
        base = self.memory.data_ptr(self.store)
        for i, b in enumerate(data):
            base[ptr + i] = b
        return ptr, len(data)

    def _try_free(self, ptr, size):
        if self._free is None:
            return
        try:
            self._free(self.store, ptr, size)
        except Exception:
            self._free = None

    def solve(self, challenge, prefix, difficulty):
        with self._lock:
            retptr = self._add_to_stack(self.store, -16)
            c_ptr = c_len = p_ptr = p_len = 0
            try:
                c_ptr, c_len = self._write_string(challenge)
                p_ptr, p_len = self._write_string(prefix)
                self._solve(self.store, retptr, c_ptr, c_len,
                            p_ptr, p_len, float(difficulty))
                mem = self.memory.data_ptr(self.store)
                status = int.from_bytes(bytes(mem[retptr:retptr + 4]),
                                        byteorder="little", signed=True)
                if status == 0:
                    return None
                vb = bytes(mem[retptr + 8:retptr + 16])
                return int(np.frombuffer(vb, dtype=np.float64)[0])
            finally:
                self._add_to_stack(self.store, 16)
                self._try_free(c_ptr, c_len)
                self._try_free(p_ptr, p_len)


_wasm_solver = None
_wasm_init_lock = threading.Lock()


def get_wasm_solver():
    global _wasm_solver
    with _wasm_init_lock:
        if _wasm_solver is None:
            _wasm_solver = DeepSeekHashWasm()
        return _wasm_solver


class PoWSolver:
    @staticmethod
    def make_header(challenge_obj):
        salt = challenge_obj["salt"]
        expire_at = challenge_obj["expire_at"]
        difficulty = challenge_obj["difficulty"]
        challenge = challenge_obj["challenge"]
        prefix = f"{salt}_{expire_at}_"
        answer = get_wasm_solver().solve(challenge, prefix, difficulty)
        if answer is None:
            raise RuntimeError("PoW failed")
        payload = {
            "algorithm": challenge_obj.get("algorithm", "DeepSeekHashV1"),
            "challenge": challenge,
            "salt": salt,
            "answer": answer,
            "signature": challenge_obj.get("signature", ""),
            "target_path": challenge_obj.get("target_path",
                                             "/api/v0/chat/completion")}
        raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        return base64.b64encode(raw).decode("ascii")


# =============================================================================
# DEEPSEEK CLIENT
# =============================================================================

class DeepSeekClient:
    def __init__(self, user_token: str, cookie: str = ""):
        self.tokens = TokenManager(user_token, cookie)
        self.cookie = cookie
        self.device_id = DEVICE_ID
        self.browser_headers: Dict[str, str] = {}
        self._client: Optional[httpx.AsyncClient] = None

    def _build_headers(self):
        headers: Dict[str, str] = dict(_FALLBACK_HEADERS)
        for k, v in (self.browser_headers or {}).items():
            if not isinstance(v, str):
                continue
            if k.lower() in _NEVER_REPLAY:
                continue
            headers[k] = v
        headers["content-type"] = "application/json"
        headers["origin"] = "https://chat.deepseek.com"
        if self.device_id:
            headers["x-device-id"] = self.device_id
        browser_cookie = None
        for k, v in (self.browser_headers or {}).items():
            if k.lower() == "cookie" and isinstance(v, str) and v:
                browser_cookie = v
                break
        if browser_cookie:
            headers["cookie"] = browser_cookie
        elif self.cookie:
            headers["cookie"] = self.cookie
        return headers

    async def _get_client(self):
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=BASE_URL,
                headers=self._build_headers(),
                timeout=httpx.Timeout(300.0, connect=30.0))
        return self._client

    def rebuild_client(self):
        old = self._client
        self._client = None
        if old is None or old.is_closed:
            return
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.run_coroutine_threadsafe(old.aclose(), loop)
            else:
                loop.run_until_complete(old.aclose())
        except Exception:
            pass

    async def close(self):
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    async def _auth_headers(self, force_refresh=False):
        token = await self.tokens.acquire_token(force_refresh=force_refresh)
        return {"authorization": f"Bearer {token}"}

    @staticmethod
    def _unwrap(data):
        code = data.get("code")
        if code not in (0, None):
            raise DeepSeekAPIError(int(code),
                data.get("msg") or data.get("message") or "", raw=data)
        inner = data.get("data")
        biz = None
        if isinstance(inner, dict):
            ib = inner.get("biz_data")
            if ib is not None:
                biz = ib
        if biz is None:
            ob = data.get("biz_data")
            if ob is not None:
                biz = ob
        if biz is None:
            raise RuntimeError(f"Unexpected shape: {data}")
        return biz

    async def _post_with_auth(self, path, json_body, extra_headers=None):
        client = await self._get_client()
        headers = await self._auth_headers()
        if extra_headers:
            headers.update(extra_headers)
        resp = await client.post(path, json=json_body, headers=headers)
        if resp.status_code == 401:
            headers = await self._auth_headers(force_refresh=True)
            if extra_headers:
                headers.update(extra_headers)
            resp = await client.post(path, json=json_body, headers=headers)
            if resp.status_code == 401:
                raise DeepSeekAPIError(40101, "userToken invalid.")
        return resp

    async def create_session(self):
        resp = await self._post_with_auth("/chat_session/create", {})
        resp.raise_for_status()
        return self._unwrap(resp.json())["chat_session"]["id"]

    async def get_pow_challenge(self, target_path="/api/v0/chat/completion"):
        resp = await self._post_with_auth(
            "/chat/create_pow_challenge", {"target_path": target_path})
        resp.raise_for_status()
        return self._unwrap(resp.json())["challenge"]

    async def stream_completion(self, prompt, chat_session_id=None,
                                parent_message_id=None,
                                thinking_enabled=False,
                                search_enabled=False, model_type="default",
                                usage_out=None, sys_fp=None,
                                stream_state=None):
        client = await self._get_client()
        if chat_session_id is None or chat_session_id == "":
            print("[session] creating chat session ...")
            chat_session_id = await self.create_session()
            print(f"[session] id={chat_session_id}")
            _session.set_session(chat_session_id, sys_fp)
        elif parent_message_id is not None:
            print(f"[session] continuing {chat_session_id[:8]}... "
                  f"parent={parent_message_id}")

        challenge = await self.get_pow_challenge()
        pow_header = await asyncio.to_thread(
            PoWSolver.make_header, challenge)

        body = {
            "action": None,
            "chat_session_id": chat_session_id,
            "model_type": model_type,
            "parent_message_id": parent_message_id,
            "preempt": False,
            "prompt": prompt,
            "ref_file_ids": [],
            "search_enabled": search_enabled,
            "thinking_enabled": thinking_enabled}
        auth = await self._auth_headers()
        headers = {"x-ds-pow-response": pow_header, **auth}

        async with client.stream("POST", "/chat/completion",
                                 json=body, headers=headers) as resp:
            if resp.status_code == 401:
                auth = await self._auth_headers(force_refresh=True)
                headers = {"x-ds-pow-response": pow_header, **auth}
                async for item in self._stream_once(
                        body, headers, usage_out, stream_state):
                    yield item
                return
            async for item in self._consume_stream(
                    resp, usage_out, stream_state):
                yield item

    async def _stream_once(self, body, headers, usage_out,
                           stream_state=None):
        client = await self._get_client()
        async with client.stream("POST", "/chat/completion",
                                 json=body, headers=headers) as resp:
            if resp.status_code == 401:
                raise DeepSeekAPIError(40101, "userToken invalid.")
            resp.raise_for_status()
            async for item in self._consume_stream(
                    resp, usage_out, stream_state):
                yield item

    async def continue_stream(self, chat_session_id, message_id,
                              fallback_to_resume=True,
                              usage_out=None, stream_state=None):
        client = await self._get_client()
        body = {
            "chat_session_id": chat_session_id,
            "message_id": message_id,
            "fallback_to_resume": fallback_to_resume}
        print(f"[continue] POST /chat/continue session="
              f"{chat_session_id[:8]}... msg_id={message_id}")
        auth = await self._auth_headers()
        headers = {**auth}

        async with client.stream("POST", "/chat/continue",
                                 json=body, headers=headers) as resp:
            if resp.status_code == 401:
                auth = await self._auth_headers(force_refresh=True)
                headers = {**auth}
                async with client.stream(
                        "POST", "/chat/continue",
                        json=body, headers=headers) as r2:
                    if r2.status_code == 401:
                        raise DeepSeekAPIError(40101, "userToken invalid.")
                    r2.raise_for_status()
                    async for item in self._consume_stream(
                            r2, usage_out, stream_state):
                        yield item
                return
            resp.raise_for_status()
            async for item in self._consume_stream(
                    resp, usage_out, stream_state):
                yield item

    async def _consume_stream(self, resp, usage_out, stream_state=None):
        if stream_state is None:
            stream_state = {}
        resp.raise_for_status()
        active_fragment_type = None
        async for raw_line in resp.aiter_lines():
            line = raw_line.strip()
            if not line or line.startswith("event:") or \
                    not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if not payload or payload == "[DONE]":
                continue
            if DEBUG:
                print(f"[sse] {payload}")
            try:
                obj = json.loads(payload)
            except json.JSONDecodeError:
                continue

            if isinstance(obj, dict):
                if obj.get("auto_continue") is True:
                    stream_state["incomplete"] = True
                    continue
                if obj.get("incomplete_message"):
                    stream_state["incomplete"] = True
                    continue
                inner_v = obj.get("v")
                if isinstance(inner_v, dict):
                    inner_resp = inner_v.get("response")
                    if isinstance(inner_resp, dict):
                        if inner_resp.get("auto_continue") is True:
                            stream_state["incomplete"] = True
                            continue
                        if inner_resp.get("incomplete_message"):
                            stream_state["incomplete"] = True
                            continue

            if "code" in obj and obj.get("code") not in (0, None):
                raise DeepSeekAPIError(int(obj["code"]),
                                       obj.get("msg") or "", raw=obj)
            if "biz_code" in obj and obj["biz_code"] not in (0, None):
                raise DeepSeekAPIError(int(obj["biz_code"]),
                                       obj.get("biz_msg") or "", raw=obj)
            if "response_message_id" in obj:
                rid = obj["response_message_id"]
                if isinstance(rid, int):
                    _session.set_last_response(rid)
                continue

            v = obj.get("v")
            if "p" not in obj and isinstance(v, dict) and "response" in v:
                for frag in ((v.get("response") or {}).get("fragments")
                             or []):
                    if not isinstance(frag, dict):
                        continue
                    ftype = frag.get("type")
                    if ftype:
                        active_fragment_type = ftype
                    c = frag.get("content")
                    if not (isinstance(c, str) and c):
                        continue
                    if ftype == "RESPONSE":
                        yield ("content", c)
                    elif ftype in ("THINK", "THINKING", "REASONING"):
                        yield ("reasoning", c)
                continue

            p = obj.get("p", "")
            o = obj.get("o", "APPEND")
            if p == "response/status":
                if v == "FINISHED":
                    break
                continue
            if p == "response" and o == "BATCH":
                if isinstance(v, list) and usage_out is not None:
                    for entry in v:
                        if isinstance(entry, dict) and \
                                entry.get("p") == "accumulated_token_usage":
                            usage_out["total_tokens"] = int(
                                entry.get("v") or 0)
                continue
            if "/elapsed_secs" in p or "/stage_id" in p or \
                    p.endswith("/references"):
                continue

            if p == "response/fragments" and isinstance(v, list):
                for frag in v:
                    if not isinstance(frag, dict):
                        continue
                    ftype = frag.get("type")
                    if ftype:
                        active_fragment_type = ftype
                    c = frag.get("content")
                    if not (isinstance(c, str) and c):
                        continue
                    if ftype == "RESPONSE":
                        yield ("content", c)
                    elif ftype in ("THINK", "THINKING", "REASONING"):
                        yield ("reasoning", c)
                continue

            if "/content" in p and isinstance(v, str) and v:
                if active_fragment_type == "RESPONSE":
                    yield ("content", v)
                elif active_fragment_type in ("THINK", "THINKING",
                                              "REASONING"):
                    yield ("reasoning", v)
                continue

            if "p" not in obj and isinstance(v, str) and v:
                if active_fragment_type == "RESPONSE":
                    yield ("content", v)
                elif active_fragment_type in ("THINK", "THINKING",
                                              "REASONING"):
                    yield ("reasoning", v)
                continue


# =============================================================================
# PROMPT BUILDERS
# =============================================================================

def _system_fingerprint(messages, tools=None):
    base = tuple((m.role, m.content or "")
                 for m in messages if m.role == "system")
    extra = ""
    if tools:
        try:
            extra = json.dumps(tools, sort_keys=True, ensure_ascii=False)
        except Exception:
            extra = str(tools)
    return hash((base, extra))


def _extract_client_system(messages):
    parts = [m.content.strip() for m in messages
             if m.role == "system" and m.content]
    return "\n\n".join(parts)


def _format_tool_result(msg):
    return f"[Tool result from {msg.name or 'tool'}]\n{msg.content or ''}"


def _format_assistant_with_tools(msg) -> str:
    text = (msg.content or "").strip()
    calls = msg.tool_calls if isinstance(msg.tool_calls, list) else []
    blocks = []
    for tc in calls:
        if not isinstance(tc, dict):
            continue
        fn = tc.get("function") or {}
        name = fn.get("name") or "?"
        args = fn.get("arguments") or ""
        blocks.append(f'[Tool call: {name}({args})]')
    combined = text
    if blocks:
        if combined:
            combined += "\n"
        combined += "\n".join(blocks)
    return combined.strip()


def messages_to_prompt(messages, wrap_system=True, tools=None,
                       tool_choice=None):
    client_system = _extract_client_system(messages)
    tools_block = _format_tools_for_prompt(tools, tool_choice)
    rest = [m for m in messages if m.role != "system"]
    lines = []
    if client_system or tools_block:
        if wrap_system:
            lines.append(_wrap_system_prompt(client_system, tools_block))
        else:
            lines.append(client_system)
            if tools_block:
                lines.append("")
                lines.append(tools_block)
        lines.append("")
    for m in rest:
        if m.role == "tool":
            lines.append(_format_tool_result(m))
            continue
        if m.role == "assistant" and m.tool_calls:
            lines.append(f"Assistant: {_format_assistant_with_tools(m)}")
            continue
        lines.append(f"{m.role.capitalize()}: {(m.content or '').strip()}")
    if rest and rest[-1].role == "user":
        lines.append("Assistant:")
    return "\n".join(lines)


def _messages_since_last_assistant(messages):
    last_asst = -1
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].role == "assistant":
            last_asst = i
            break
    if last_asst < 0:
        return list(messages)
    return list(messages[last_asst + 1:])


def _build_continuation_prompt(messages, mode, client_system,
                               tools=None, tool_choice=None):
    last_asst_idx = -1
    last_asst_had_tools = False
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].role == "assistant":
            last_asst_idx = i
            tc = messages[i].tool_calls
            last_asst_had_tools = bool(tc)
            break

    if last_asst_idx < 0:
        new_msgs = list(messages)
    else:
        new_msgs = list(messages[last_asst_idx + 1:])

    new_msgs = [m for m in new_msgs if m.role != "system"]
    if not new_msgs:
        return None

    tool_parts = []
    user_parts = []
    for m in new_msgs:
        if m.role == "tool":
            tool_parts.append(_format_tool_result(m))
        elif m.role == "user":
            user_parts.append((m.content or "").strip())

    if not tool_parts and not user_parts:
        return None

    if last_asst_had_tools and tool_parts:
        body = (
            "The tool calls you made in the previous turn have returned. "
            "Here are the results:\n\n"
            + "\n\n".join(tool_parts)
            + "\n\nNow continue the task. Rules:\n"
            "  - Do NOT repeat a tool call with the same arguments.\n"
            "  - If the task is finished, reply with the final answer.\n"
            "  - If you need another tool call, emit the "
            "<tool_calls>[...]</tool_calls> JSON block exactly as "
            "instructed.\n"
        )
        if user_parts:
            body += ("\n\nAdditional user message:\n"
                     + "\n\n".join(user_parts))
        if mode == "agent" and REINFORCE_SYSTEM and client_system:
            body = _REINFORCE + body
    else:
        parts = []
        if tool_parts:
            parts.append("Context from the environment:\n\n"
                         + "\n\n".join(tool_parts))
        if user_parts:
            parts.append("\n\n".join(user_parts))
        body = "\n\n".join(parts)
        if mode == "agent" and REINFORCE_SYSTEM and client_system:
            body = _REINFORCE + body

    # Compact tool reminder
    if tools:
        reminder = _tool_names_summary(tools)
        if reminder:
            body = reminder + "\n\n" + body
    return body


# =============================================================================
# RESPONSE BUILDERS
# =============================================================================

def _new_id():
    return f"chatcmpl-{uuid.uuid4().hex[:24]}"


def _new_call_id():
    return f"call_{uuid.uuid4().hex[:24]}"


def _sse(payload):
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def build_completion(model, content, reasoning, pt, ct, tool_calls=None):
    message = {"role": "assistant",
               "content": content if content else None}
    if reasoning:
        message["reasoning_content"] = reasoning
    finish = "stop"
    if tool_calls:
        message["tool_calls"] = [
            {"id": _new_call_id(), "type": "function",
             "function": {"name": c["name"], "arguments": c["arguments"]}}
            for c in tool_calls]
        finish = "tool_calls"
    if ct <= 0:
        ct = max(1, len(content or "") // 4)
    return {
        "id": _new_id(), "object": "chat.completion",
        "created": int(time.time()), "model": model,
        "choices": [{"index": 0, "message": message,
                     "logprobs": None, "finish_reason": finish}],
        "usage": {"prompt_tokens": pt, "completion_tokens": ct,
                  "total_tokens": pt + ct}}


def build_chunk(model, chunk_id, created, delta, finish_reason=None):
    return {"id": chunk_id, "object": "chat.completion.chunk",
            "created": created, "model": model,
            "choices": [{"index": 0, "delta": delta,
                         "logprobs": None,
                         "finish_reason": finish_reason}]}


def build_tool_call_chunk(model, chunk_id, created, calls):
    return {"id": chunk_id, "object": "chat.completion.chunk",
            "created": created, "model": model,
            "choices": [{"index": 0,
                         "delta": {"tool_calls": [
                             {"index": i, "id": _new_call_id(),
                              "type": "function",
                              "function": {"name": c["name"],
                                           "arguments": c["arguments"]}}
                             for i, c in enumerate(calls)]},
                         "logprobs": None, "finish_reason": None}]}


def build_usage_chunk(model, chunk_id, created, pt, ct):
    return {"id": chunk_id, "object": "chat.completion.chunk",
            "created": created, "model": model, "choices": [],
            "usage": {"prompt_tokens": pt, "completion_tokens": ct,
                      "total_tokens": pt + ct}}


# =============================================================================
# FASTAPI
# =============================================================================

class ChatMessage(BaseModel):
    role: str
    content: Optional[Union[str, List[Any]]] = None
    tool_calls: Optional[Any] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None


class StreamOptions(BaseModel):
    include_usage: bool = False


class ChatCompletionRequest(BaseModel):
    model: str = "deepseek-chat"
    messages: List[ChatMessage]
    stream: bool = False
    stream_options: Optional[StreamOptions] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    max_completion_tokens: Optional[int] = None
    stop: Optional[Any] = None
    user: Optional[str] = None
    tools: Optional[Any] = None
    tool_choice: Optional[Any] = None
    parallel_tool_calls: Optional[bool] = None
    response_format: Optional[Any] = None
    seed: Optional[int] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    logit_bias: Optional[Any] = None
    logprobs: Optional[bool] = None
    top_logprobs: Optional[int] = None
    n: Optional[int] = 1


_client = DeepSeekClient(DEEPSEEK_USER_TOKEN, DEEPSEEK_COOKIE)
_lock = asyncio.Lock()


async def _require_auth(request: Request):
    if not BRIDGE_API_KEY:
        return
    auth = request.headers.get("authorization", "")
    if auth != f"Bearer {BRIDGE_API_KEY}":
        raise HTTPException(status_code=401, detail="invalid bridge api key")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global MODE
    MODE = _ask_startup_mode()

    initial = DEEPSEEK_USER_TOKEN or _load_token_from_disk() or ""
    if initial:
        _client.tokens.user_token = initial
        print(f"[token] initial userToken ({len(initial)} chars)")
    else:
        print("[token] waiting for browser extension")

    cached = _load_headers_from_disk()
    if cached:
        _client.browser_headers = cached
        ua = cached.get("user-agent", "?")
        print(f"[headers] loaded cached bundle ({len(cached)} headers)")
        print(f"[headers]   user-agent: "
              f"{ua[:70]}{'...' if len(ua) > 70 else ''}")

    if _client.device_id:
        print(f"[device] using device_id: {_client.device_id}")

    _bridge.start()
    threading.Thread(target=_watch_bridge, daemon=True).start()

    try:
        get_wasm_solver()
        print("[wasm] loaded sha3_wasm_bg.wasm")
    except Exception as exc:
        print(f"[wasm] FATAL: {exc}")
        sys.exit(1)

    if _client.tokens.user_token:
        try:
            await _client.tokens.acquire_token(force_refresh=True)
        except Exception as exc:
            print(f"[token] refresh failed: {exc}")

    threading.Thread(target=_stdin_listener, daemon=True).start()

    print()
    print("=" * 64)
    print(f"  OpenAI bridge -> http://{BIND_HOST}:{BIND_PORT}/v1")
    print(f"  Token bridge  -> http://127.0.0.1:{TOKEN_BRIDGE_PORT}")
    print("-" * 64)
    print(f"  [mode] {'AGENT' if MODE == 'agent' else 'CHAT'}")
    print("  Full OpenAI tools forwarding + auto-continue enabled")
    if TOOLS_STRIP_PREFIXES:
        print(f"  tool strip prefixes: {', '.join(TOOLS_STRIP_PREFIXES)}")
    if TOOLS_KEEP_ONLY:
        print(f"  tool keep-only: {len(TOOLS_KEEP_ONLY)} names")
    print("  Runtime commands:  agent|a  chat|c  status|s  ENTER")
    print("=" * 64)
    print()

    yield
    _bridge.stop()
    await _client.close()


app = FastAPI(title="DeepSeek Web -> OpenAI Bridge", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])


@app.get("/health")
async def health():
    sid, parent, _ = _session.snapshot()
    age = _bridge.age()
    hdrs = _client.browser_headers or {}
    return {
        "status": "ok", "mode": MODE,
        "session_id": sid, "parent_message_id": parent,
        "turn_count": _session.turn_count,
        "device_id": _client.device_id or None,
        "user_agent": hdrs.get("user-agent"),
        "browser_headers_count": len(hdrs),
        "access_token_expires_in": (
            max(0, int(_client.tokens._expires_at - time.time()))
            if _client.tokens._expires_at else None),
        "user_token_source": (
            f"extension({_bridge.source})" if _bridge.get_token()
            else "disk" if _load_token_from_disk()
            else "env" if DEEPSEEK_USER_TOKEN else "none"),
        "user_token_age_seconds": (
            None if age == float("inf") else int(age)),
    }


@app.get("/v1/models", dependencies=[Depends(_require_auth)])
async def list_models():
    now = int(time.time())
    return {"object": "list", "data": [
        {"id": "deepseek-chat", "object": "model",
         "created": now, "owned_by": "deepseek"},
        {"id": "deepseek-reasoner", "object": "model",
         "created": now, "owned_by": "deepseek"},
        {"id": "auto", "object": "model",
         "created": now, "owned_by": "deepseek"}]}


@app.post("/api/fresh", dependencies=[Depends(_require_auth)])
async def fresh_session():
    old_sid, _, _ = _session.snapshot()
    _session.reset()
    print(f"[api] /api/fresh - session reset "
          f"(was {old_sid[:8] if old_sid else 'none'})")
    return {
        "status": "ok",
        "message": "session reset - next request starts a new chat",
        "previous_session_id": old_sid,
    }


@app.post("/v1/chat/completions", dependencies=[Depends(_require_auth)])
async def chat_completions(request: Request):
    global MODE
    raw = await request.json()
    try:
        body = ChatCompletionRequest(**raw)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    if not body.messages:
        raise HTTPException(status_code=400,
                            detail="messages must not be empty")

    if not _client.tokens.user_token:
        raise HTTPException(status_code=503,
            detail=("Bridge waiting for browser extension. Load the "
                    "extension, disable uBlock on chat.deepseek.com, "
                    "and open the site once."))

    model = body.model
    is_reasoner = model in ("deepseek-reasoner", "deepseek-r1")
    model_type = "expert" if is_reasoner else "default"
    thinking = is_reasoner
    want_usage = bool(body.stream_options and
                      body.stream_options.include_usage)

    tools = body.tools if isinstance(body.tools, list) else None
    tool_choice = body.tool_choice

    # tool_choice="none" disables tool use entirely
    tools_disabled = (
        isinstance(tool_choice, str) and tool_choice == "none"
    )

    sys_fp = _system_fingerprint(body.messages, tools)
    session_id, parent_id, stored_fp = _session.snapshot()

    if (RESET_ON_SYSTEM_CHANGE and session_id and stored_fp is not None
            and sys_fp != stored_fp):
        print("[session] system prompt changed - resetting")
        _session.reset()
        session_id, parent_id = None, None

    client_system = _extract_client_system(body.messages)

    if session_id:
        cont = _build_continuation_prompt(
            body.messages, MODE, client_system,
            tools=None if tools_disabled else tools,
            tool_choice=tool_choice)
        if cont is None:
            print("[session] nothing new - folding full history")
            prompt = messages_to_prompt(
                body.messages, wrap_system=False,
                tools=None if tools_disabled else tools,
                tool_choice=tool_choice)
            _session.reset()
            session_id = None
            parent_id = None
        elif len(cont) > MAX_CONTINUATION_CHARS:
            print(f"[session] continuation too large ({len(cont)} chars "
                  f"> {MAX_CONTINUATION_CHARS}) - folding full history "
                  f"in a fresh session")
            _session.reset()
            session_id = None
            parent_id = None
            wrap = bool(client_system)
            prompt = messages_to_prompt(
                body.messages, wrap_system=wrap,
                tools=None if tools_disabled else tools,
                tool_choice=tool_choice)
        else:
            prompt = cont
            sig = "|".join(
                f"{m.name}:{hash(m.content or '')}"
                for m in _messages_since_last_assistant(body.messages)
                if m.role == "tool")
            if sig:
                if _session.tool_signature_seen(sig):
                    print("[session] repeated tool result - adding "
                          "no-repeat warning")
                    prompt = ("[WARNING: You already invoked this exact "
                              "tool with these arguments and already saw "
                              "this result. Do NOT call it again. Use the "
                              "information you have and produce the final "
                              "answer.]\n\n" + prompt)
                else:
                    _session.remember_tool_signature(sig)
            print(f"[req] {model} CONTINUATION ({MODE}) "
                  f"session={session_id[:8]}... parent={parent_id} "
                  f"prompt_len={len(prompt)}")
    else:
        wrap = bool(client_system)
        prompt = messages_to_prompt(
            body.messages, wrap_system=wrap,
            tools=None if tools_disabled else tools,
            tool_choice=tool_choice)
        parent_id = None
        src = "client" if client_system else "none"
        print(f"[req] {model} NEW ({MODE}) system={src} "
              f"model_type={model_type} thinking={thinking} "
              f"tools={len(tools) if tools else 0} "
              f"tool_choice={tool_choice!r} "
              f"prompt_len={len(prompt)}")

    prompt_tokens = max(1, len(prompt) // 4)

    if body.stream:
        return StreamingResponse(
            _stream_response(prompt, model, prompt_tokens, thinking,
                             model_type, session_id, parent_id, sys_fp,
                             want_usage),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache",
                     "X-Accel-Buffering": "no",
                     "Connection": "keep-alive"})

    usage_out: Dict[str, int] = {}
    content_parts: List[str] = []
    reasoning_parts: List[str] = []
    stream_state: Dict[str, bool] = {}

    async with _lock:
        try:
            async for kind, text in _client.stream_completion(
                    prompt, chat_session_id=session_id,
                    parent_message_id=parent_id,
                    thinking_enabled=thinking, model_type=model_type,
                    usage_out=usage_out, sys_fp=sys_fp,
                    stream_state=stream_state):
                (content_parts if kind == "content"
                 else reasoning_parts).append(text)
        except DeepSeekAPIError as exc:
            raise HTTPException(status_code=502, detail=str(exc))

    raw_content = "".join(content_parts)
    reasoning = "".join(reasoning_parts)
    ct = max(0, usage_out.get("total_tokens", 0) - prompt_tokens)

    if MODE == "agent" and not tools_disabled:
        content, tool_calls = _split_text_and_tool_calls(raw_content)
    else:
        content, tool_calls = raw_content, []

    if MODE == "agent" and not tools_disabled and not tool_calls \
            and _detect_truncation(raw_content):
        print("[warn] response looks truncated - the model may have "
              "exceeded the token limit")

    # AUTO-CONTINUE
    if (MODE == "agent" and not tool_calls and content
            and MAX_CONTINUE_ROUNDS > 0
            and (_looks_incomplete(content)
                 or stream_state.get("incomplete"))):
        rounds = 0
        while rounds < MAX_CONTINUE_ROUNDS:
            if not _looks_incomplete(content):
                break
            sid, parent, _ = _session.snapshot()
            if not sid or not parent:
                print("[continue] no session/message_id - cannot continue")
                break

            rounds += 1
            print(f"[continue] round {rounds}: "
                  f"{len(content)} chars so far - calling /chat/continue")

            cont_parts: List[str] = []
            async with _lock:
                try:
                    async for kind, text in _client.continue_stream(
                            chat_session_id=sid,
                            message_id=parent,
                            fallback_to_resume=True,
                            usage_out=usage_out):
                        if kind == "content":
                            cont_parts.append(text)
                except Exception as exc:
                    print(f"[continue] failed: {exc}")
                    break

            cont_text = "".join(cont_parts)
            if not cont_text:
                print(f"[continue] round {rounds}: empty - stopping")
                break

            if cont_text.startswith(content):
                added = len(cont_text) - len(content)
                print(f"[continue] round {rounds}: +{added} chars "
                      f"(full={len(cont_text)})")
                content = cont_text
            elif cont_text in content:
                print(f"[continue] round {rounds}: no new content - stop")
                break
            else:
                print(f"[continue] round {rounds}: +{len(cont_text)} "
                      f"chars (appended)")
                content = content + cont_text

            c2, tc2 = _split_text_and_tool_calls(content)
            if tc2:
                content = c2
                tool_calls = tc2
                print(f"[continue] final tool_calls={len(tc2)}")
                break

        if rounds > 0:
            print(f"[continue] completed {rounds} round(s), "
                  f"final {len(content)} chars")

    # Retry 1: promise without tool call
    if (MODE == "agent" and not tool_calls and content
            and not tools_disabled
            and _looks_like_promise(content)):
        print("[retry] model promised an action without a tool call - "
              "nudging once")
        sid, parent, _ = _session.snapshot()
        nudge = (
            "[You said you would do something, but you did not emit a "
            "tool call. Do NOT repeat that text. Emit the tool call NOW "
            "using the <tool_calls>[...]</tool_calls> format, or, if no "
            "tool is needed, provide the final answer in plain prose.]"
        )
        retry_parts: List[str] = []
        async with _lock:
            try:
                async for kind, text in _client.stream_completion(
                        nudge,
                        chat_session_id=sid,
                        parent_message_id=parent,
                        thinking_enabled=False,
                        model_type=model_type,
                        usage_out=usage_out, sys_fp=sys_fp):
                    if kind == "content":
                        retry_parts.append(text)
            except Exception as exc:
                print(f"[retry] failed: {exc}")

        retry_content = "".join(retry_parts)
        if retry_content:
            c2, tc2 = _split_text_and_tool_calls(retry_content)
            if tc2:
                content = c2 if c2 else ""
                tool_calls = tc2
                print(f"[retry] got tool_calls={len(tc2)}")
            elif c2:
                content = c2
                print(f"[retry] got content only ({len(c2)} chars)")

    # Retry 2: empty response
    if not content and not tool_calls:
        print("[retry] empty response - asking for final answer")
        sid, parent, _ = _session.snapshot()
        nudge = (
            "[You returned no output. Provide the FINAL ANSWER to the "
            "user now, in plain prose. Do NOT call any more tools. "
            "Summarize what you accomplished.]"
        )
        retry_parts: List[str] = []
        async with _lock:
            try:
                async for kind, text in _client.stream_completion(
                        nudge,
                        chat_session_id=sid,
                        parent_message_id=parent,
                        thinking_enabled=False,
                        model_type=model_type,
                        usage_out=usage_out, sys_fp=sys_fp):
                    if kind == "content":
                        retry_parts.append(text)
            except Exception as exc:
                print(f"[retry] failed: {exc}")

        retry_content = "".join(retry_parts)
        if retry_content:
            c2, tc2 = _split_text_and_tool_calls(retry_content)
            if c2:
                content = c2
                print(f"[retry] got final answer ({len(c2)} chars)")
            if tc2:
                tool_calls = tc2
        else:
            print("[retry] still empty - resetting session")
            _session.reset()

    print(f"[resp] content={len(content)} tool_calls={len(tool_calls)} "
          f"reasoning={len(reasoning)} ct={ct}")
    return build_completion(model, content, reasoning, prompt_tokens, ct,
                            tool_calls=tool_calls)


@app.post("/v1/completions", dependencies=[Depends(_require_auth)])
async def legacy_completions(request: Request):
    """
    Legacy /v1/completions endpoint. Maps prompt -> messages and
    returns the same structure as the chat endpoint but with the
    legacy text format.
    """
    raw = await request.json()
    prompt = raw.get("prompt", "")
    if isinstance(prompt, list):
        prompt = prompt[0] if prompt else ""
    if not isinstance(prompt, str):
        prompt = str(prompt)

    body = ChatCompletionRequest(
        model=raw.get("model", "deepseek-chat"),
        messages=[ChatMessage(role="user", content=prompt)],
        stream=bool(raw.get("stream", False)),
        temperature=raw.get("temperature"),
        top_p=raw.get("top_p"),
        max_tokens=raw.get("max_tokens"),
        stop=raw.get("stop"),
        user=raw.get("user"),
        seed=raw.get("seed"),
        presence_penalty=raw.get("presence_penalty"),
        frequency_penalty=raw.get("frequency_penalty"),
        n=raw.get("n", 1))

    # Reuse the chat handler by synthesizing a request
    new_raw = {
        "model": body.model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": body.stream,
        "temperature": body.temperature,
        "top_p": body.top_p,
        "max_tokens": body.max_tokens,
        "stop": body.stop,
        "user": body.user,
        "seed": body.seed,
        "presence_penalty": body.presence_penalty,
        "frequency_penalty": body.frequency_penalty,
        "n": body.n,
    }

    class _FakeRequest:
        def __init__(self, data):
            self._data = data
        async def json(self):
            return self._data

    result = await chat_completions(_FakeRequest(new_raw))

    # If streaming, return as-is; the content format works for
    # /v1/completions clients too.
    if body.stream:
        return result

    # Non-streaming: convert chat response to legacy text format
    if isinstance(result, dict) and result.get("choices"):
        content = result["choices"][0].get("message", {}).get("content") or ""
        return {
            "id": result["id"],
            "object": "text_completion",
            "created": result["created"],
            "model": result["model"],
            "choices": [{
                "text": content,
                "index": 0,
                "logprobs": None,
                "finish_reason": result["choices"][0].get("finish_reason", "stop"),
            }],
            "usage": result.get("usage", {}),
        }
    return result


async def _stream_response(prompt, model, prompt_tokens, thinking, model_type,
                           session_id, parent_id, sys_fp, want_usage):
    chunk_id = _new_id()
    created = int(time.time())
    usage_out: Dict[str, int] = {}
    completion_tokens = 0
    parser = StreamingToolParser() if MODE == "agent" else None
    emitted_tool_calls = False
    parser_buffer_parts: List[str] = []
    stream_state: Dict[str, bool] = {}

    yield _sse(build_chunk(model, chunk_id, created,
                           {"role": "assistant", "content": ""}))

    async with _lock:
        try:
            async for kind, text in _client.stream_completion(
                    prompt, chat_session_id=session_id,
                    parent_message_id=parent_id,
                    thinking_enabled=thinking, model_type=model_type,
                    usage_out=usage_out, sys_fp=sys_fp,
                    stream_state=stream_state):
                if kind == "reasoning":
                    yield _sse(build_chunk(model, chunk_id, created,
                                           {"reasoning_content": text}))
                    continue
                completion_tokens += 1
                if parser is None:
                    yield _sse(build_chunk(model, chunk_id, created,
                                           {"content": text}))
                    continue
                for ev_type, ev_data in parser.feed(text):
                    if ev_type == "content":
                        parser_buffer_parts.append(ev_data)
                        yield _sse(build_chunk(model, chunk_id, created,
                                               {"content": ev_data}))
                    elif ev_type == "tool_calls":
                        emitted_tool_calls = True
                        yield _sse(build_tool_call_chunk(
                            model, chunk_id, created, ev_data))

            if parser is not None:
                for ev_type, ev_data in parser.flush():
                    if ev_type == "content":
                        parser_buffer_parts.append(ev_data)
                        yield _sse(build_chunk(model, chunk_id, created,
                                               {"content": ev_data}))
                    elif ev_type == "tool_calls":
                        emitted_tool_calls = True
                        yield _sse(build_tool_call_chunk(
                            model, chunk_id, created, ev_data))
        except DeepSeekAPIError as exc:
            yield _sse({"error": {"message": str(exc),
                                  "type": exc.label}})
            yield "data: [DONE]\n\n"
            return
        except Exception as exc:
            yield _sse({"error": {"message": str(exc),
                                  "type": "backend_error"}})
            yield "data: [DONE]\n\n"
            return

    # Auto-continue (streaming)
    if (parser is not None and not emitted_tool_calls
            and MAX_CONTINUE_ROUNDS > 0):
        accumulated = "".join(parser_buffer_parts)
        if accumulated and (_looks_incomplete(accumulated)
                            or stream_state.get("incomplete")):
            rounds = 0
            while rounds < MAX_CONTINUE_ROUNDS:
                if not _looks_incomplete(accumulated):
                    break
                sid, parent, _ = _session.snapshot()
                if not sid or not parent:
                    break

                rounds += 1
                print(f"[continue] (stream) round {rounds}: "
                      f"{len(accumulated)} chars so far")

                parser2 = StreamingToolParser()
                cont_parts: List[str] = []
                async with _lock:
                    try:
                        async for kind, text in _client.continue_stream(
                                chat_session_id=sid,
                                message_id=parent,
                                fallback_to_resume=True,
                                usage_out=usage_out):
                            if kind == "content":
                                cont_parts.append(text)
                                for ev_type, ev_data in parser2.feed(text):
                                    if ev_type == "content":
                                        yield _sse(build_chunk(
                                            model, chunk_id, created,
                                            {"content": ev_data}))
                                    elif ev_type == "tool_calls":
                                        emitted_tool_calls = True
                                        yield _sse(
                                            build_tool_call_chunk(
                                                model, chunk_id, created,
                                                ev_data))
                    except Exception as exc:
                        print(f"[continue] (stream) failed: {exc}")
                        break

                for ev_type, ev_data in parser2.flush():
                    if ev_type == "content":
                        yield _sse(build_chunk(model, chunk_id, created,
                                               {"content": ev_data}))
                    elif ev_type == "tool_calls":
                        emitted_tool_calls = True
                        yield _sse(build_tool_call_chunk(
                            model, chunk_id, created, ev_data))

                cont_text = "".join(cont_parts)
                if not cont_text:
                    break
                if cont_text.startswith(accumulated):
                    accumulated = cont_text
                else:
                    accumulated = accumulated + cont_text
                if emitted_tool_calls:
                    break
                if not _looks_incomplete(accumulated):
                    print("[continue] (stream) response now complete")
                    break

            if rounds > 0:
                print(f"[continue] (stream) completed {rounds} round(s)")

    # Retries in streaming mode
    if parser is not None and not emitted_tool_calls:
        accumulated = "".join(parser_buffer_parts)
        should_retry = False
        nudge = ""
        if accumulated and _looks_like_promise(accumulated):
            should_retry = True
            nudge = (
                "[You said you would do something, but you did not emit "
                "a tool call. Do NOT repeat that text. Emit the tool "
                "call NOW using the <tool_calls>[...]</tool_calls> "
                "format, or provide the final answer in plain prose.]"
            )
            print("[retry] (stream) model promised without a tool call - "
                  "nudging once")
        elif not accumulated:
            should_retry = True
            nudge = (
                "[You returned no output. Provide the FINAL ANSWER to "
                "the user now, in plain prose. Do NOT call any more "
                "tools. Summarize what you accomplished.]"
            )
            print("[retry] (stream) empty response - asking for final "
                  "answer")

        if should_retry:
            sid, parent, _ = _session.snapshot()
            parser2 = StreamingToolParser()
            got_something = False
            async with _lock:
                try:
                    async for kind, text in _client.stream_completion(
                            nudge,
                            chat_session_id=sid,
                            parent_message_id=parent,
                            thinking_enabled=False,
                            model_type=model_type,
                            usage_out=usage_out, sys_fp=sys_fp):
                        if kind != "content":
                            continue
                        for ev_type, ev_data in parser2.feed(text):
                            if ev_type == "content":
                                got_something = True
                                yield _sse(build_chunk(
                                    model, chunk_id, created,
                                    {"content": ev_data}))
                            elif ev_type == "tool_calls":
                                emitted_tool_calls = True
                                got_something = True
                                yield _sse(build_tool_call_chunk(
                                    model, chunk_id, created, ev_data))
                    for ev_type, ev_data in parser2.flush():
                        if ev_type == "content":
                            got_something = True
                            yield _sse(build_chunk(
                                model, chunk_id, created,
                                {"content": ev_data}))
                        elif ev_type == "tool_calls":
                            emitted_tool_calls = True
                            got_something = True
                            yield _sse(build_tool_call_chunk(
                                model, chunk_id, created, ev_data))
                except Exception as exc:
                    print(f"[retry] (stream) failed: {exc}")

            if not got_something:
                print("[retry] (stream) still empty - resetting session")
                _session.reset()

    finish = "tool_calls" if emitted_tool_calls else "stop"
    yield _sse(build_chunk(model, chunk_id, created, {},
                           finish_reason=finish))
    if want_usage:
        reported = usage_out.get("total_tokens", 0)
        real_ct = max(0, reported - prompt_tokens) or completion_tokens
        yield _sse(build_usage_chunk(model, chunk_id, created,
                                     prompt_tokens, real_ct))
    yield "data: [DONE]\n\n"


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=BIND_HOST, port=BIND_PORT)
