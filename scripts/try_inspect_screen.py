"""Diagnose the "look at the screen" feature step by step.

Usage:
    python scripts/try_inspect_screen.py [--url http://127.0.0.1:8080] [--open]
                                         [--query "Что видно на экране?"] [--image FILE]

It repeats what the inspect_screen tool does (take a screenshot, send it to the vision
model) and prints what happens at every step, so a missing screen-recording permission,
a timeout or an empty answer can be told apart. It needs no OSA imports on macOS.
"""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
import subprocess
import sys
import time
import urllib.error
import urllib.request

MIN_PLAUSIBLE_BYTES = 30_000


def capture(target: Path) -> None:
    """Take a screenshot into ``target`` the same way the tool does."""
    target.parent.mkdir(parents=True, exist_ok=True)

    if sys.platform == "darwin":
        subprocess.run(["/usr/sbin/screencapture", "-x", "-t", "png", str(target)], check=True)
        subprocess.run(
            ["/usr/bin/sips", "-Z", "1024", str(target)],
            capture_output=True,
            check=True,
        )
        return

    from osa.desktop.factory import create_desktop_automation

    target.write_bytes(create_desktop_automation().screenshot())


def ask(url: str, image: Path, query: str, *, max_tokens: int, with_system: bool) -> dict:
    encoded = base64.b64encode(image.read_bytes()).decode("utf-8")
    messages: list[dict] = []

    if with_system:
        messages.append({"role": "system", "content": "You are a helpful assistant."})

    messages.append(
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"Ответь на русском кратко и по существу: {query}"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}},
            ],
        }
    )
    payload = {"messages": messages, "max_tokens": max_tokens, "temperature": 0.1}
    request = urllib.request.Request(
        url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(request, timeout=120) as response:
        return json.load(response)


def show(label: str, started: float, data: dict) -> str:
    choice = data["choices"][0]
    content = choice["message"].get("content")
    elapsed = time.monotonic() - started
    print(f"[{label}] {elapsed:.1f} с, finish_reason={choice.get('finish_reason')}")
    print(f"[{label}] usage={data.get('usage')}")
    print(f"[{label}] ответ: {content!r}")

    return (content or "").strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", default="http://127.0.0.1:8080")
    parser.add_argument("--query", default="Что видно на экране?")
    parser.add_argument("--image", help="Use this image instead of taking a screenshot")
    parser.add_argument("--open", action="store_true", help="Open the screenshot to look at it")
    args = parser.parse_args()

    if args.image:
        path = Path(args.image)
    else:
        path = Path("data") / "screenshots" / "diagnostic.png"
        print("1) Снимок экрана...")

        try:
            capture(path)
        except Exception as exc:
            print(f"[FAIL] Не удалось сделать снимок: {exc}")
            return 1

    size = path.stat().st_size
    print(f"   файл: {path} ({size // 1024} КБ)")

    if args.open:
        opener = {"darwin": ["open"], "win32": ["cmd", "/c", "start", ""]}.get(
            sys.platform, ["xdg-open"]
        )
        subprocess.run([*opener, str(path)], check=False)

    if size < MIN_PLAUSIBLE_BYTES:
        print("   [!] Файл подозрительно маленький: возможно, снимок пустой (нет разрешения).")

    print("2) Запрос к модели, как делает инструмент...")
    started = time.monotonic()

    try:
        data = ask(args.url, path, args.query, max_tokens=150, with_system=False)
        first = show("как в инструменте", started, data)
    except (urllib.error.URLError, OSError, KeyError, ValueError) as exc:
        print(f"[FAIL] Запрос не удался: {exc!r}")
        print("   Сервер запущен? Модель с --mmproj? Время ответа больше 30 секунд?")
        return 2

    print("3) Тот же запрос с системным сообщением и большим лимитом...")
    started = time.monotonic()

    try:
        data = ask(args.url, path, args.query, max_tokens=400, with_system=True)
        second = show("вариант", started, data)
    except (urllib.error.URLError, OSError, KeyError, ValueError) as exc:
        print(f"[FAIL] Запрос не удался: {exc!r}")
        return 2

    print()

    if not first and not second:
        print("ИТОГ: модель вернула пустой ответ в обоих случаях (проблема в сервере или шаблоне).")
        return 3

    if not first:
        print("ИТОГ: в режиме инструмента ответ пустой, а в варианте нет:")
        print("      нужно менять запрос инструмента.")
        return 3

    print("ИТОГ: модель отвечает. Если описание не соответствует экрану, откройте снимок")
    print("      (флаг --open): пустой фон без окон значит, что нет права «Запись экрана».")

    return 0


if __name__ == "__main__":
    sys.exit(main())
