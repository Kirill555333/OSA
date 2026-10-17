"""Main entry point and interactive shell for OSA (v1.5 — Clean Cognitive Core)."""

from __future__ import annotations

import argparse
import os
import signal
import sys
import urllib.request
from typing import Any

from osa.actions.desktop_tool_safety import DesktopToolSafetyPolicy, ForegroundProbe
from osa.browser import HttpBrowser
from osa.context import ContextBudget
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
    ConfirmationHandler,
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
    FocusWindowTool,
    ForgetTool,
    HotkeyTool,
    InspectScreenTool,
    LaunchApplicationTool,
    ListApplicationsTool,
    ListDirectoryTool,
    ListWindowsTool,
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
from osa.utils.paths import resolve_osa_home, resolve_workspace
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


def create_agent(
    memory_storage: LongTermMemory | None = None,
    confirmation_handler: ConfirmationHandler | None = None,
    wake_names: Any | None = None,
) -> Agent:
    """Build and configure the main OSA agent with complete native Tool Calling."""
    model = LlamaCppModel(
        LlamaCppConfig(
            base_url=LLAMA_SERVER_URL,
            timeout=45.0,
        )
    )

    server_n_ctx = model.context_size()
    if not isinstance(server_n_ctx, int) or server_n_ctx <= 0:
        server_n_ctx = 8192

    context_budget = ContextBudget.create(
        total_context_limit=server_n_ctx,
        reserved_output_tokens=512,
    )

    osa_home = resolve_osa_home()
    data_dir = osa_home / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    workspace_dir = resolve_workspace()

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
    tool_registry.register(ListApplicationsTool())
    tool_registry.register(ListWindowsTool())
    tool_registry.register(FocusWindowTool())

    if wake_names is not None:
        from osa.tools.wake_name import ClearWakeNameTool, SetWakeNameTool

        tool_registry.register(SetWakeNameTool(wake_names))
        tool_registry.register(ClearWakeNameTool(wake_names))
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
            "close_application": PermissionLevel.CONFIRM,
            "list_applications": PermissionLevel.ALLOW,
            "list_windows": PermissionLevel.ALLOW,
            "focus_window": PermissionLevel.ALLOW,
            "set_wake_name": PermissionLevel.ALLOW,
            "clear_wake_name": PermissionLevel.ALLOW,
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

    if confirmation_handler is None:
        confirmation_handler = create_interactive_confirmation_handler()

    logs_dir = osa_home / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    event_logger = EventLogger(logs_dir / "osa-events.jsonl")

    system_prompt = (
        "Ты — OSA (Оса), персональный ассистент с доступом к компьютеру пользователя. "
        "Отвечай кратко, вежливо и по-русски.\n\n"
        "Правила:\n"
        "1. Если просьба требует действия на компьютере (открыть или закрыть программу "
        "или сайт, найти что-то в интернете, переключить окно, напечатать текст, нажать "
        "клавишу, изменить громкость, посмотреть на экран, запомнить что-то), вызови "
        "подходящий инструмент через механизм вызова инструментов. Не записывай вызов "
        "текстом и не описывай, что собираешься сделать.\n"
        "2. Не сообщай, что действие выполнено, пока инструмент не вернул результат. "
        "Если инструмент вернул ошибку, честно скажи об этом и предложи вариант.\n"
        "3. Не уверен в названии программы или окна — сначала вызови list_applications "
        "или list_windows и выбери из реального списка.\n"
        "4. Для поиска в интернете или видео вызови open_url с готовой ссылкой поиска "
        "(например, https://www.youtube.com/results?search_query=... или "
        "https://www.google.com/search?q=...), подставив запрос пользователя в "
        "именительном падеже, без выдуманных слов.\n"
        "5. Если действие не нужно, просто ответь."
    )

    # Desktop input tools go through the desktop safety rules: confirmation for
    # protected applications, destructive hotkeys and blocked combinations.
    # close_application is already confirmed by the permission policy.
    action_safety_policy = DesktopToolSafetyPolicy(
        foreground=ForegroundProbe(),
        skip_tools=frozenset({"close_application"}),
    )

    return Agent(
        model=model,
        system_prompt=system_prompt,
        temperature=0.1,
        automatic_memory=automatic_memory,
        max_tokens=512,
        tool_registry=tool_registry,
        event_logger=event_logger,
        permission_policy=permission_policy,
        confirmation_handler=confirmation_handler,
        recovery=ErrorRecovery(),
        memory_integration=memory_integration,
        action_safety_policy=action_safety_policy,
        context_budget=context_budget,
    )


def run_interactive_loop(voice_enabled: bool = True) -> None:
    """Main interactive conversational loop with Native Tool Calling & Voice."""
    setup_signal_handler()

    osa_home = resolve_osa_home()
    data_dir = osa_home / "data"
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

            # Прямой вызов агента: модель САМА выбирает инструмент.
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


def run_listening_loop(listen_mode: str, voice_enabled: bool = True) -> None:
    """Interactive loop with background voice listening and keyboard input together."""
    from osa.shell.input_hub import InputHub
    from osa.shell.session import (
        OutputSpeaker,
        create_confirmation_callback,
        create_debug_sink,
        debug_settings_from_env,
        describe_names,
        format_heard,
        handle_listen_command,
        handle_name_command,
        listener_problem,
        wake_config_from_env,
    )
    from osa.voice.background import BackgroundListener
    from osa.voice.background_audio import FasterWhisperTranscriber, SoundDeviceFrameSource
    from osa.voice.names import WakeNameService
    from osa.voice.wake import ListenMode, WakeWordGate

    setup_signal_handler()

    osa_home = resolve_osa_home()
    data_dir = osa_home / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    memory_storage = LongTermMemory(data_dir / "osa-memory.db")

    hub = InputHub()
    wake_names = WakeNameService(data_dir / "osa-settings.json")
    agent = create_agent(
        memory_storage,
        ConfirmationHandler(callback=create_confirmation_callback(hub)),
        wake_names=wake_names,
    )
    voice_engine = OSAVoiceEngine()
    model_size = os.environ.get("OSA_STT_MODEL", "small")

    print("\n" + "=" * 65)
    print(f"  OSA AI Assistant (v{VERSION}) — фоновое прослушивание")
    print(f"  Inference: {LLAMA_SERVER_URL}")
    print(f"  Микрофон: режим {listen_mode}, Whisper {model_size}")
    print(f"  {describe_names(wake_names)}")
    print("  Скажите «Оса, ...» или печатайте текст. Команды: /listen, /name, /heard")
    print("  Подтверждения опасных действий — только с клавиатуры.")
    print("  Выход: Ctrl+C или напишите 'exit' / 'выход'")
    print("=" * 65 + "\n")

    print(f"[Загрузка слуха Whisper {model_size}...]", end="\r", flush=True)
    transcriber = FasterWhisperTranscriber(model_size)
    transcriber.preload()
    print(" " * 45, end="\r", flush=True)

    debug_enabled, log_transcripts = debug_settings_from_env(os.environ)
    debug_sink = None

    if debug_enabled:
        debug_sink = create_debug_sink(osa_home / "logs" / "voice-debug.log")

    gate = WakeWordGate(
        wake_config_from_env(os.environ),
        mode=ListenMode.parse(listen_mode),
        extra_names=wake_names.alias_names(),
    )
    listener: BackgroundListener | None = BackgroundListener(
        SoundDeviceFrameSource(),
        transcriber,
        gate,
        hub.post_voice,
        debug=debug_sink,
        log_transcripts=log_transcripts,
    )

    try:
        listener.start()
    except Exception as exc:
        print(f"[Микрофон недоступен: {exc}. Продолжаю только с клавиатурой.]\n")
        listener = None

    if listener is not None:
        wake_names.subscribe(listener.set_wake_names)

    speaker = OutputSpeaker(voice_engine, listener, hub)
    hub.start()

    prompt = "User > "
    problem_reported = False

    try:
        if voice_enabled:
            speaker.say("Здравствуйте, сэр. Я готов к работе.")

        print(prompt, end="", flush=True)

        while True:
            event = hub.next_event(timeout=0.5)

            if event is None:
                problem = None if problem_reported else listener_problem(listener)

                if problem:
                    problem_reported = True
                    print(f"\n[{problem}]\n{prompt}", end="", flush=True)

                continue

            if event.kind == "eof":
                print("\nOSA session ended. Good bye!")
                break

            from_voice = event.kind == "voice"
            user_input = event.text.strip()

            try:
                if from_voice and event.reason == "wake_only":
                    print("\n 🎙 Слушаю.")

                    if voice_enabled:
                        speaker.say("Слушаю.")

                    if listener is not None:
                        listener.open_follow_up()

                    print(prompt, end="", flush=True)
                    continue

                if not user_input:
                    print(prompt, end="", flush=True)
                    continue

                if from_voice:
                    print(f"\nВы (голос) > {user_input}\n")

                low = user_input.lower()

                if not from_voice and low.startswith("/listen"):
                    print(f"OSA > {handle_listen_command(user_input, listener)}\n")
                    print(prompt, end="", flush=True)
                    continue

                if not from_voice and low.startswith("/name"):
                    print(f"OSA > {handle_name_command(user_input, wake_names)}\n")
                    print(prompt, end="", flush=True)
                    continue

                if not from_voice and low.startswith("/heard"):
                    print(f"OSA > {format_heard(listener)}\n")
                    print(prompt, end="", flush=True)
                    continue

                if any(w in low for w in ("включи голос", "отвечай вслух", "включи звук")):
                    voice_enabled = True
                    msg = "Голосовой режим включен, сэр."
                    print(f"OSA > {msg}\n")
                    speaker.say(msg)
                    print(prompt, end="", flush=True)
                    continue

                if any(w in low for w in ("выключи голос", "без звука", "не говори вслух")):
                    voice_enabled = False
                    print("OSA > [Голосовой вывод отключен.]\n")
                    print(prompt, end="", flush=True)
                    continue

                exit_words = ("выход", "закройся")

                if not from_voice:
                    exit_words += ("/exit", "/quit", "exit", "quit", "стоп")

                if low in exit_words:
                    print("\nOSA stopped. Good bye, sir!")

                    if voice_enabled:
                        speaker.say("До свидания, сэр.")

                    break

                print("[OSA выполняет...]", flush=True)
                response = agent.chat(user_input)
                output_text = getattr(response, "content", "") or ""

                if not output_text and hasattr(response, "reasoning_content"):
                    output_text = getattr(response, "reasoning_content", "") or ""

                if not output_text:
                    output_text = "Готово, сэр."

                print(f"OSA > {output_text}\n")

                if voice_enabled:
                    speaker.say(output_text)

                if from_voice and listener is not None:
                    listener.open_follow_up()
            except Exception as exc:
                print(f"\n[Ошибка выполнения: {exc}]\n")

            print(prompt, end="", flush=True)
    except (KeyboardInterrupt, EOFError):
        print("\nOSA session ended. Good bye!")
    finally:
        voice_engine.stop()

        if listener is not None:
            listener.stop()


def main() -> None:
    """Main CLI entrypoint."""
    parser = argparse.ArgumentParser(description="OSA — Personal AI Desktop Assistant")
    parser.add_argument("--voice", action="store_true", help="Enable voice output (TTS)")
    parser.add_argument(
        "--listen",
        nargs="?",
        const="wake",
        choices=["wake", "always"],
        default=None,
        help="Background microphone mode (default: wake, say 'Оса, ...'; always: everything)",
    )
    parser.add_argument(
        "--push-to-talk",
        action="store_true",
        help="Legacy mode: press Enter to record a phrase instead of background listening",
    )
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

    if not args.push_to_talk:
        # Default: background listening (say 'Оса, ...' or type); see --listen
        run_listening_loop(args.listen or "wake", voice_enabled=True)
        return

    # Legacy mode: type text or press Enter to record a phrase
    run_interactive_loop(voice_enabled=True)


if __name__ == "__main__":
    main()
