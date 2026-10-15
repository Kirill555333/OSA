"""Check that a local llama.cpp server really supports tool calling.

Usage:
    python scripts/check_tool_calling.py [--url http://127.0.0.1:8080]

The script only talks to the server over HTTP and needs no OSA imports. It prints the
chat template capabilities and runs a few realistic requests, reporting PASS or FAIL
for each. Exit code 0 means every check passed.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request

OPEN_URL = {
    "type": "function",
    "function": {
        "name": "open_url",
        "description": "Open a web page in the default browser.",
        "parameters": {
            "type": "object",
            "properties": {"url": {"type": "string", "description": "Full http(s) address."}},
            "required": ["url"],
        },
    },
}
LAUNCH_APPLICATION = {
    "type": "function",
    "function": {
        "name": "launch_application",
        "description": "Launch an installed desktop application by its real name.",
        "parameters": {
            "type": "object",
            "properties": {
                "application_name": {"type": "string", "description": "Application name."},
            },
            "required": ["application_name"],
        },
    },
}
SYSTEM = (
    "Ты — OSA, ассистент с доступом к компьютеру. Если просьба требует действия на "
    "компьютере, вызови подходящий инструмент. Если действие не нужно, ответь текстом."
)

# (prompt, tools, expected tool name or None when no call is expected)
CASES = (
    ("Открой сайт github.com", [OPEN_URL, LAUNCH_APPLICATION], "open_url"),
    ("Найди на ютубе обзоры на ноутбуки", [OPEN_URL, LAUNCH_APPLICATION], "open_url"),
    ("Открой калькулятор", [OPEN_URL, LAUNCH_APPLICATION], "launch_application"),
    ("Привет, как дела?", [OPEN_URL, LAUNCH_APPLICATION], None),
)


def _request(base_url: str, path: str, payload: dict | None = None) -> dict:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"} if payload is not None else {}
    req = urllib.request.Request(base_url.rstrip("/") + path, data=data, headers=headers)

    with urllib.request.urlopen(req, timeout=180) as response:
        return json.load(response)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", default="http://127.0.0.1:8080")
    args = parser.parse_args()

    try:
        props = _request(args.url, "/props")
    except (urllib.error.URLError, OSError) as exc:
        print(f"[FAIL] Server is not reachable at {args.url}: {exc}")
        return 2

    caps = props.get("chat_template_caps") or {}
    supports_tools = bool(caps.get("supports_tools"))
    print(f"chat template supports tools: {supports_tools}")

    if not supports_tools:
        print(
            "[FAIL] The chat template ignores tools. Start llama-server with --jinja and a "
            "tool-capable --chat-template-file (see config/chat_templates)."
        )
        return 1

    failures = 0

    for prompt, tools, expected in CASES:
        payload = {
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": prompt},
            ],
            "tools": tools,
            "temperature": 0.1,
            "max_tokens": 200,
        }

        try:
            data = _request(args.url, "/v1/chat/completions", payload)
            message = data["choices"][0]["message"]
        except Exception as exc:
            print(f"[FAIL] {prompt!r}: request failed: {exc}")
            failures += 1
            continue

        calls = message.get("tool_calls") or []
        names = [call.get("function", {}).get("name") for call in calls]
        arguments = [call.get("function", {}).get("arguments") for call in calls]

        if expected is None:
            ok = not calls
        else:
            ok = bool(names) and names[0] == expected

        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {prompt!r} -> calls={names} args={arguments}")

        if not ok:
            failures += 1
            print(f"       content: {message.get('content')!r}")

    total = len(CASES)
    print(f"\n{total - failures}/{total} checks passed")

    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
