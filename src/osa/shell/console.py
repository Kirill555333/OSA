"""Interactive JARVIS-style console shell for OSA."""

from __future__ import annotations

import os
import sys
from typing import Any, TextIO

from osa.core import Agent, AgentError
from osa.core.modes import AgentMode
from osa.permissions import ConfirmationHandler


class ConsoleColors:
    """ANSI color codes for terminal formatting."""

    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    MAGENTA = "\033[95m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"

    @classmethod
    def strip_if_unsupported(cls, text: str) -> str:
        """Strip ANSI codes if running in a non-tty environment."""
        if not sys.stdout.isatty():
            for code in (
                cls.CYAN,
                cls.GREEN,
                cls.YELLOW,
                cls.RED,
                cls.MAGENTA,
                cls.BOLD,
                cls.DIM,
                cls.RESET,
            ):
                text = text.replace(code, "")
        return text


def terminal_confirmation_prompt(
    description: str,
    *,
    input_stream: TextIO | None = None,
    output_stream: TextIO | None = None,
) -> bool:
    """Prompt the user interactively in the terminal for operation confirmation."""
    inp = input_stream or sys.stdin
    out = output_stream or sys.stdout

    prompt = (
        f"\n{ConsoleColors.YELLOW}{ConsoleColors.BOLD}"
        f"[ACTION CONFIRMATION REQUIRED]{ConsoleColors.RESET}\n"
        f"{description}\n"
        f"{ConsoleColors.BOLD}Allow this operation? [y/N]: {ConsoleColors.RESET}"
    )

    out.write(ConsoleColors.strip_if_unsupported(prompt))
    out.flush()

    try:
        response = inp.readline()
        if not response:
            return False
        return response.strip().lower() in ("y", "yes")
    except (EOFError, KeyboardInterrupt):
        return False


def create_interactive_confirmation_handler() -> ConfirmationHandler:
    """Create a ConfirmationHandler that prompts the user interactively in the terminal."""
    return ConfirmationHandler(callback=terminal_confirmation_prompt)


class InteractiveShell:
    """Interactive command shell managing session interaction, voice, and commands."""

    def __init__(
        self,
        agent: Agent,
        *,
        voice_runtime: Any | None = None,
        input_stream: TextIO | None = None,
        output_stream: TextIO | None = None,
    ) -> None:
        self._agent = agent
        self._voice_runtime = voice_runtime
        self._input = input_stream or sys.stdin
        self._output = output_stream or sys.stdout

    def print_banner(self) -> None:
        """Display the JARVIS-style startup banner."""
        model_name = self._agent.model.model_name
        mode_val = self._agent.mode.value if hasattr(self._agent.mode, "value") else str(self._agent.mode)
        ctx_limit = self._agent.context_manager.budget.total_context_limit
        voice_status = "Available" if self._voice_runtime is not None else "Standby"

        banner = (
            f"{ConsoleColors.CYAN}{ConsoleColors.BOLD}\n"
            "╔═════════════════════════════════════════════════════════════════════╗\n"
            "║                       OSA — PERSONAL AI AGENT                       ║\n"
            f"║  Model: {model_name:<16} Mode: {mode_val:<10} Context Limit: {ctx_limit:<6} ║\n"
            f"║  Voice Runtime: {voice_status:<16}                                    ║\n"
            "╚═════════════════════════════════════════════════════════════════════╝\n"
            f"{ConsoleColors.RESET}"
            f"{ConsoleColors.DIM}Type /help for command list, /voice for speech mode, or 'exit' to quit.{ConsoleColors.RESET}\n"
        )
        self._write(banner)

    def handle_command(self, command_line: str) -> bool:
        """Handle a slash command. Returns True if handled, False otherwise."""
        parts = command_line.strip().split()
        if not parts:
            return True

        cmd = parts[0].lower()
        args = parts[1:]

        if cmd in ("/exit", "/quit"):
            self._write(f"{ConsoleColors.GREEN}OSA session terminated.{ConsoleColors.RESET}\n")
            return False

        if cmd == "/help":
            self._cmd_help()
        elif cmd == "/status":
            self._cmd_status()
        elif cmd == "/mode":
            self._cmd_mode(args)
        elif cmd == "/new":
            self._cmd_new()
        elif cmd == "/tools":
            self._cmd_tools()
        elif cmd == "/voice":
            self._cmd_voice()
        elif cmd == "/clear":
            self._cmd_clear()
        else:
            self._write(f"{ConsoleColors.RED}Unknown command '{cmd}'. Type /help for available commands.{ConsoleColors.RESET}\n")

        return True

    def run(self) -> None:
        """Run the main interactive command loop."""
        self.print_banner()

        while True:
            prompt_str = f"{ConsoleColors.GREEN}{ConsoleColors.BOLD}You > {ConsoleColors.RESET}"
            self._write(prompt_str)

            try:
                line = self._input.readline()
                if not line:
                    self._write("\n")
                    break
                user_input = line.strip()
            except (EOFError, KeyboardInterrupt):
                self._write("\nOSA session terminated.\n")
                break

            if not user_input:
                continue

            if user_input.lower() in ("exit", "quit"):
                self._write(f"{ConsoleColors.GREEN}OSA session terminated.{ConsoleColors.RESET}\n")
                break

            if user_input.startswith("/"):
                should_continue = self.handle_command(user_input)
                if not should_continue:
                    break
                continue

            self._execute_chat(user_input)

    def _execute_chat(self, user_input: str) -> None:
        """Send message to agent and stream chunks to terminal."""
        self._write(f"{ConsoleColors.CYAN}{ConsoleColors.BOLD}OSA > {ConsoleColors.RESET}")

        try:
            for chunk in self._agent.chat_stream(user_input):
                self._write(chunk)
            self._write("\n\n")

        except AgentError as exc:
            self._write(f"\n{ConsoleColors.RED}Agent Error: {exc}{ConsoleColors.RESET}\n\n")
        except Exception as exc:
            self._write(f"\n{ConsoleColors.RED}Unexpected Error: {exc}{ConsoleColors.RESET}\n\n")

    def _cmd_help(self) -> None:
        help_text = (
            f"\n{ConsoleColors.BOLD}Available Shell Commands:{ConsoleColors.RESET}\n"
            f"  {ConsoleColors.CYAN}/help{ConsoleColors.RESET}             Show this help menu\n"
            f"  {ConsoleColors.CYAN}/status{ConsoleColors.RESET}           Display model, mode, token budget, and voice statistics\n"
            f"  {ConsoleColors.CYAN}/voice{ConsoleColors.RESET}            Start interactive voice conversation mode\n"
            f"  {ConsoleColors.CYAN}/mode [name]{ConsoleColors.RESET}      Get or set agent mode (chat, task, autonomous)\n"
            f"  {ConsoleColors.CYAN}/new{ConsoleColors.RESET}              Start a fresh conversation session\n"
            f"  {ConsoleColors.CYAN}/tools{ConsoleColors.RESET}            List all available tools and descriptions\n"
            f"  {ConsoleColors.CYAN}/clear{ConsoleColors.RESET}            Clear terminal screen\n"
            f"  {ConsoleColors.CYAN}/exit, /quit{ConsoleColors.RESET}      Exit the interactive console\n\n"
        )
        self._write(help_text)

    def _cmd_status(self) -> None:
        budget = self._agent.context_manager.budget
        summary = self._agent.context_manager.current_summary
        mode_val = self._agent.mode.value if hasattr(self._agent.mode, "value") else str(self._agent.mode)
        is_healthy = self._agent.model.health_check()
        health_str = f"{ConsoleColors.GREEN}Healthy{ConsoleColors.RESET}" if is_healthy else f"{ConsoleColors.RED}Offline{ConsoleColors.RESET}"
        voice_str = f"{ConsoleColors.GREEN}Connected{ConsoleColors.RESET}" if self._voice_runtime is not None else f"{ConsoleColors.DIM}Standby / Not Initialized{ConsoleColors.RESET}"

        status_text = (
            f"\n{ConsoleColors.BOLD}OSA System Status:{ConsoleColors.RESET}\n"
            f"  Model:           {self._agent.model.model_name} [{health_str}]\n"
            f"  Voice Runtime:   {voice_str}\n"
            f"  Agent Mode:      {ConsoleColors.YELLOW}{mode_val}{ConsoleColors.RESET}\n"
            f"  Active Session:  {len(self._agent.context.messages())} messages in history\n"
            f"  Summary State:   {'Active (' + str(summary.messages_summarized_count) + ' summarized)' if not summary.is_empty else 'None (recent chat)'}\n"
            f"  Context Budget:  {budget.total_context_limit} total tokens\n"
            f"    - Input Max:   {budget.max_input_tokens} tokens\n"
            f"    - Reserved:    {budget.reserved_output_tokens} tokens\n"
            f"    - History:     {budget.history_budget} tokens\n"
            f"    - Memory:      {budget.memory_budget} tokens\n"
            f"    - Summary:     {budget.summary_budget} tokens\n"
            f"    - Tools:       {budget.tool_budget} tokens\n\n"
        )
        self._write(status_text)

    def _cmd_mode(self, args: list[str]) -> None:
        if not args:
            current = self._agent.mode.value if hasattr(self._agent.mode, "value") else str(self._agent.mode)
            self._write(f"Current mode: {ConsoleColors.YELLOW}{current}{ConsoleColors.RESET}\n\n")
            return

        target_mode = args[0].lower()
        try:
            mode_enum = AgentMode(target_mode)
            self._agent.set_mode(mode_enum)
            self._write(f"{ConsoleColors.GREEN}Agent mode set to '{target_mode}'.{ConsoleColors.RESET}\n\n")
        except ValueError:
            valid_modes = ", ".join(m.value for m in AgentMode)
            self._write(f"{ConsoleColors.RED}Invalid mode '{target_mode}'. Valid modes: {valid_modes}{ConsoleColors.RESET}\n\n")

    def _cmd_new(self) -> None:
        self._agent.reset()
        self._write(f"{ConsoleColors.GREEN}Conversation reset. Started a new clean session.{ConsoleColors.RESET}\n\n")

    def _cmd_tools(self) -> None:
        self._write(f"\n{ConsoleColors.BOLD}Registered Tools ({len(self._agent.tools)}):{ConsoleColors.RESET}\n")
        for tool in self._agent.tools.describe():
            self._write(f"  {ConsoleColors.CYAN}{tool['name']:<22}{ConsoleColors.RESET} {tool['description']}\n")
        self._write("\n")

    def _cmd_voice(self) -> None:
        """Start interactive voice dialogue mode."""
        if self._voice_runtime is None:
            self._write(
                f"\n{ConsoleColors.YELLOW}[Voice Runtime Standby]{ConsoleColors.RESET}\n"
                "Live microphone/speaker runtime is not configured or in standby.\n"
                "To enable voice, ensure audio drivers (sounddevice, whisper, piper) are initialized.\n\n"
            )
            return

        self._write(
            f"\n{ConsoleColors.MAGENTA}{ConsoleColors.BOLD}"
            "=== OSA LIVE VOICE MODE ACTIVE ===\n"
            f"{ConsoleColors.RESET}"
            f"{ConsoleColors.DIM}Speak into your microphone. Say 'exit' or press Ctrl+C to return to text.{ConsoleColors.RESET}\n\n"
        )

        try:
            self._voice_runtime.start()
            while True:
                self._write(f"{ConsoleColors.MAGENTA}[Listening...]{ConsoleColors.RESET}\r")
                session_result = self._voice_runtime.run_once()

                user_text = getattr(session_result, "transcribed_text", "").strip()
                if not user_text:
                    continue

                self._write(f"{ConsoleColors.GREEN}You (Voice) > {ConsoleColors.RESET}{user_text}\n")

                if user_text.lower() in ("exit", "quit", "стоп", "выход", "отмена"):
                    self._write(f"{ConsoleColors.MAGENTA}Voice session ended.{ConsoleColors.RESET}\n\n")
                    break

                # Let agent process and get text response
                self._write(f"{ConsoleColors.CYAN}OSA (Voice) > {ConsoleColors.RESET}")
                resp = self._agent.chat(user_text)
                reply_text = resp.content.strip()
                self._write(f"{reply_text}\n\n")

                # Speak response through runtime
                try:
                    self._voice_runtime.speak(reply_text)
                except Exception as exc:
                    self._write(f"{ConsoleColors.DIM}[Voice playback error: {exc}]{ConsoleColors.RESET}\n")

        except (KeyboardInterrupt, EOFError):
            self._write(f"\n{ConsoleColors.MAGENTA}Voice session interrupted. Returned to text console.{ConsoleColors.RESET}\n\n")
        except Exception as exc:
            self._write(f"\n{ConsoleColors.RED}Voice session error: {exc}{ConsoleColors.RESET}\n\n")
        finally:
            try:
                self._voice_runtime.stop()
            except Exception:
                pass

    def _cmd_clear(self) -> None:
        os.system("cls" if os.name == "nt" else "clear")
        self.print_banner()

    def _write(self, text: str) -> None:
        self._output.write(ConsoleColors.strip_if_unsupported(text))
        self._output.flush()
