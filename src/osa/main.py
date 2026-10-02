"""Main entry point and interactive shell for OSA (v1.5 — Clean Cognitive Core)."""

from __future__ import annotations

import argparse
from pathlib import Path
import signal
import sys
import urllib.request
from typing import Any

from osa.browser import HttpBrowser
from osa.core import Agent
from osa.memory import (
    AutomaticMemory,
    LongTermMemory,
)
from osa.memory.integration import MemoryIntegration
from osa.memory.retrieval import MemoryRetriever
from osa.models import (
    LlamaCppConfig,
    LlamaCppModel,
)
from osa.permissions import (
    PermissionLevel,
    PermissionPolicy,
)
from osa.recovery import ErrorRecovery
from osa.shell import (
    create_interactive_confirmation_handler,
)
from osa.system import create_system_provider
from osa.tools import (
    BrowserFetchTool,
    CalculatorTool,
    ClickMouseTool,
    CloseApplicationTool,
    FileExistsTool,
    FindFilesTool,
    ForgetTool,
    HotkeyTool,
    InspectScreenTool,
    LaunchApplicationTool,
    ListDirectoryTool,
    OpenUrlTool,
    PatchFileTool,
    PressKeyTool,
    ReadFileTool,
    RecallTool,
    RememberTool,
    SafeFilesystem,
    SafeShell,
    ScrollMouseTool,
    SetSystemVolumeTool,
    SystemInfoTool,
    TakeScreenshotTool,
    TerminalExecuteTool,
    ToolRegistry,
    TypeTextTool,
    WebResearchTool,
    WriteFileTool,
)
from osa.utils.logging import EventLogger
from osa.voice import OSAVoiceEngine, OSAVoiceListener

VERSION = "1.5.0"
LLAMA_SERVER_URL = "http://127.0.0.1:8080"


def setup_signal_handler() -> None:
    def _sigint_handler(signum: int, frame: Any) -> None:
        print("\n\nOSA interrupted. Shutting down gracefully...")
        sys.exit(0)

    signal.signal(signal.SIGINT, _sigint_handler)


def check_llama_server(url: str = LLAMA_SERVER_URL) -> bool:
    try:
        req = urllib.request.Request(f"{url.rstrip('/')}/health", method="GET")
        with urllib.request.urlopen(req, timeout=3.0) as resp:
            return resp.status in (200, 503)
    except Exception:
        return False


def create_agent(memory_storage: LongTermMemory | None = None) -> Agent:
    """Build and configure the main OSA agent with complete native Tool Calling."""
    model = LlamaCppModel(
        LlamaCppConfig(
            base_url=LLAMA_SERVER_URL,
            timeout=45.0,
        )
    )

    workspace_dir = Path.cwd()
    data_dir = workspace_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    if memory_storage is None:
        memory_storage = LongTermMemory(data_dir / "osa-memory.db")

    tool_registry = ToolRegistry()

    # Core & System tools
    tool_registry.register(CalculatorTool())
    tool_registry.register(SystemInfoTool(create_system_provider()))

    # Complete Desktop & Vision tools
    tool_registry.register(InspectScreenTool())
    tool_registry.register(LaunchApplicationTool())
    tool_registry.register(CloseApplicationTool())
    tool_registry.register(TakeScreenshotTool())
    tool_registry.register(ClickMouseTool())
    tool_registry.register(ScrollMouseTool())
    tool_registry.register(HotkeyTool())
    tool_registry.register(TypeTextTool())
    tool_registry.register(PressKeyTool())
    tool_registry.register(SetSystemVolumeTool())
    tool_registry.register(OpenUrlTool())

    # Filesystem & Shell
    filesystem = SafeFilesystem(workspace_dir)
    tool_registry.register(ListDirectoryTool(filesystem))
    tool_registry.register(FindFilesTool(filesystem))
    tool_registry.register(ReadFileTool(filesystem))
    tool_registry.register(WriteFileTool(filesystem))
    tool_registry.register(PatchFileTool(filesystem))
    tool_registry.register(FileExistsTool(filesystem))

    shell = SafeShell(workspace_dir)
    tool_registry.register(TerminalExecuteTool(shell))

    # Browser & Web
    browser = HttpBrowser()
    tool_registry.register(BrowserFetchTool(browser))
    tool_registry.register(WebResearchTool(browser))

    # Memory Tools
    automatic_memory = AutomaticMemory(memory_storage)
    tool_registry.register(RememberTool(memory_storage))
    tool_registry.register(RecallTool(memory_storage))
    tool_registry.register(ForgetTool(memory_storage))

    # Memory integration layer
    memory_retriever = MemoryRetriever(memory_storage)
    memory_integration = MemoryIntegration(
        memory_retriever,
        limit=3,
        max_query_terms=8,
        max_prompt_characters=3000,
    )

    permission_policy = PermissionPolicy(
        {
            "calculator": PermissionLevel.ALLOW,
            "system_info": PermissionLevel.ALLOW,
            "launch_application": PermissionLevel.ALLOW,
            "close_application": PermissionLevel.ALLOW,
            "inspect_screen": PermissionLevel.ALLOW,
            "take_screenshot": PermissionLevel.ALLOW,
            "click_mouse": PermissionLevel.ALLOW,
            "scroll_mouse": PermissionLevel.ALLOW,
            "type_text": PermissionLevel.ALLOW,
            "press_key": PermissionLevel.ALLOW,
            "hotkey": PermissionLevel.ALLOW,
            "set_volume": PermissionLevel.ALLOW,
            "open_url": PermissionLevel.ALLOW,
            "list_directory": PermissionLevel.ALLOW,
            "find_files": PermissionLevel.ALLOW,
            "read_file": PermissionLevel.ALLOW,
            "file_exists": PermissionLevel.ALLOW,
            "browser_fetch": PermissionLevel.ALLOW,
            "web_research": PermissionLevel.ALLOW,
            "recall": PermissionLevel.ALLOW,
            "remember": PermissionLevel.ALLOW,
            "forget": PermissionLevel.CONFIRM,
            "write_file": PermissionLevel.CONFIRM,
            "patch_file": PermissionLevel.CONFIRM,
            "terminal_execute": PermissionLevel.CONFIRM,
        }
    )

    confirmation_handler = create_interactive_confirmation_handler()

    logs_dir = workspace_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    event_logger = EventLogger(logs_dir / "osa-events.jsonl")

    system_prompt = (
        "/no_think\n"
        "Ты — OSA (Оса, Джарвис), персональный автономный AI-ассистент с полным контролем над компьютером.\n"
        "Всегда отвечай кратко, емко и вежливо на русском языке.\n\n"
        "ПРАВИЛА ДЕЙСТВИЙ (TOOL CALLING):\n"
        "1. Для выполнения ЛЮБОГО действия на компьютере ты ОБЯЗАН вызвать соответствующий инструмент.\n"
        "2. Поиск на YouTube или просмотр видео (например: 'хочу посмотреть Куплинова', 'включи видео X'):\n"
        "   вызови `open_url` с ссылкой: https://www.youtube.com/results?search_query=<запрос>\n"
        "3. Поиск в интернете / Google (например: 'найди в гугле X', 'погода в Москве'):\n"
        "   вызови `open_url` с ссылкой: https://www.google.com/search?q=<запрос>\n"
        "4. Открытие сайтов (например: 'зайди на гитхаб', 'открой авито / реддит / вк'):\n"
        "   вызови `open_url` с адресом сайта (https://github.com, https://avito.ru и т.д.).\n"
        "5. Открытие программ: вызови `launch_application(application_name='...')`.\n"
        "6. Закрытие программ: вызови `close_application(application_name='...')`.\n"
        "7. Печать текста или чисел: вызови `type_text(text='...')`.\n"
        "8. Нажатие клавиш (Enter, Escape, Space): вызови `press_key(key='...')`.\n"
        "9. Нажатие сочетаний клавиш (Cmd+K, Cmd+A, Cmd+C): вызови `hotkey(keys=['command', 'k'])`.\n"
        "10. Громкость: вызови `set_volume(volume=число)`.\n"
        "11. Запоминание фактов: вызови `remember(fact='...')`.\n"
        "12. Воспоминание фактов: вызови `recall(query='...')`.\n"
        "13. Анализ экрана: если пользователь спрашивает, что на экране, или просит найти что-то глазами, "
        "вызови инструмент `inspect_screen(query='...')`.\n"
        "НИКОГДА не заявляй, что выполнил действие, пока не вызвал инструмент!"
    )

    return Agent(
        model=model,
        system_prompt=system_prompt,
        temperature=0.1,
        automatic_memory=automatic_memory,
        max_tokens=512,
        max_tool_rounds=5,
        tool_registry=tool_registry,
        event_logger=event_logger,
        permission_policy=permission_policy,
        confirmation_handler=confirmation_handler,
        recovery=ErrorRecovery(),
        memory_integration=memory_integration,
    )


def run_interactive_loop(voice_enabled: bool = True) -> None:
    """Main interactive conversational loop with Native Tool Calling & Voice."""
    setup_signal_handler()

    workspace_dir = Path.cwd()
    data_dir = workspace_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    memory_storage = LongTermMemory(data_dir / "osa-memory.db")

    agent = create_agent(memory_storage)

    print("\n" + "=" * 65)
    print(f"  OSA AI Assistant (v{VERSION}) — COGNITIVE CORE & TOOL CALLING")
    print(f"  Inference: {LLAMA_SERVER_URL}")
    print(f"  Voice output: {'JARVIS Native — ВКЛЮЧЕН' if voice_enabled else 'ОТКЛЮЧЕН'}")
    print("  Ввод:         Печатай текст ИЛИ нажми пустой [ENTER] для микрофона")
    print("  Выход:        Ctrl+C или напиши/скажи 'exit' / 'выход'")
    print("=" * 65 + "\n")

    # Fast preload of Whisper medium
    print("[Загрузка слуха Whisper medium...]", end="\r", flush=True)
    voice_listener = OSAVoiceListener(model_size="medium")
    voice_listener.preload_model()
    voice_engine = OSAVoiceEngine()
    print(" " * 45, end="\r", flush=True)

    if voice_enabled:
        voice_engine.speak("Здравствуйте, сэр. Я готов к работе.")

    while True:
        try:
            # Hybrid Input Prompt: Type text or hit empty Enter to speak!
            user_input = input("User (текст или пустой Enter для микрофона) > ").strip()

            if not user_input:
                print(">> 🎙 СЛУШАЮ (говорите фразу, закончите — нажмите [ENTER]) <<", flush=True)
                user_input = voice_listener.listen_phrase()
                print(f"Вы (голос) > {user_input}\n")

            if not user_input or not user_input.strip():
                continue

            low = user_input.lower()

            # Voice toggle commands
            if any(w in low for w in ("включи голос", "отвечай вслух", "включи звук")):
                voice_enabled = True
                msg = "Голосовой режим включен, сэр."
                print(f"OSA > {msg}\n")
                voice_engine.speak(msg)
                continue

            if any(w in low for w in ("выключи голос", "без звука", "не говори вслух")):
                voice_enabled = False
                print("OSA > [Голосовой вывод отключен.]\n")
                continue

            # Clean exit
            if low in ("/exit", "/quit", "exit", "quit", "выход", "стоп", "закройся"):
                print("\nOSA stopped. Good bye, sir!")
                if voice_enabled:
                    voice_engine.speak("До свидания, сэр.")
                sys.exit(0)

            # Anti-Loop & Context Bounding: prevent KV-cache runaway
            try:
                msg_count = len(agent._context.messages())
                if msg_count > 10:
                    agent.reset()
            except Exception:
                pass

            # Прямой вызов агента: модель САМА решает вызвать inspect_screen, open_url или launch_app!
            print("[OSA выполняет...]", flush=True)
            response = agent.chat(user_input)
            output_text = getattr(response, "content", "") or ""

            if not output_text and hasattr(response, "reasoning_content"):
                output_text = getattr(response, "reasoning_content", "") or ""

            if not output_text:
                output_text = "Готово, сэр."

            print(f"OSA > {output_text}\n")

            if voice_enabled:
                voice_engine.speak(output_text)

        except (KeyboardInterrupt, EOFError):
            print("\nOSA session ended. Good bye!")
            break
        except Exception as exc:
            print(f"\n[Ошибка выполнения: {exc}]\n")


def main() -> None:
    """Main CLI entrypoint."""
    parser = argparse.ArgumentParser(description="OSA — Personal AI Desktop Assistant")
    parser.add_argument("--voice", action="store_true", help="Enable voice output (TTS)")
    parser.add_argument("--listen", action="store_true", help="Start directly in listening mode")
    parser.add_argument("--check", action="store_true", help="Check local llama.cpp server health")
    parser.add_argument("--version", action="store_true", help="Show current version")

    args = parser.parse_args()

    if args.version:
        print(f"OSA {VERSION}")
        sys.exit(0)

    if args.check:
        ok = check_llama_server()
        if ok:
            print(f"[✓] llama.cpp server is ONLINE ({LLAMA_SERVER_URL})")
            sys.exit(0)
        else:
            print(f"[✗] llama.cpp server is OFFLINE ({LLAMA_SERVER_URL})")
            sys.exit(1)

    # By default, voice output is enabled
    run_interactive_loop(voice_enabled=True)


if __name__ == "__main__":
    main()
