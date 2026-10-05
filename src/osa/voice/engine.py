"""Multi-engine Text-to-Speech (TTS) architecture for OSA with timeout protection.

Playback runs in a child process (``afplay`` or ``say``). The engine remembers that
process, so callers can check whether OSA is still speaking, wait for the end of the
speech, or interrupt it. Temporary audio files are removed when playback finishes.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
from typing import Any, Literal

VoiceMode = Literal["jarvis", "system", "clone"]


class OSAVoiceEngine:
    """Manages playback and live switching of TTS voice profiles."""

    def __init__(self, mode: VoiceMode = "jarvis") -> None:
        self.mode: VoiceMode = mode
        self.jarvis_voice: str = "ru-RU-DmitryNeural"
        self.system_voice: str = "Milena" if sys.platform == "darwin" else ""
        self.clone_sample_path: Path | None = None
        self._has_say = shutil.which("say") is not None and sys.platform == "darwin"
        # Test hook; when None, subprocess.Popen is resolved at call time.
        self._popen: Any = None
        self._lock = threading.Lock()
        self._process: Any = None
        self._cleanup_path: Path | None = None

    def set_mode(self, mode: VoiceMode, clone_path: str | Path | None = None) -> None:
        """Switch voice provider on the fly."""
        self.mode = mode
        if clone_path:
            self.clone_sample_path = Path(clone_path)

    def speak(self, text: str) -> bool:
        """Start speaking without ever freezing the main thread.

        Returns as soon as playback has started; use ``wait_until_done`` to wait for it.
        A new utterance interrupts the previous one instead of overlapping with it.
        """
        clean_text = self._sanitize_text(text)
        if not clean_text:
            return False

        self.stop()

        if self.mode == "jarvis":
            success = self._speak_edge(clean_text)
            if not success:
                return self._speak_system(clean_text)
            return True
        else:
            return self._speak_system(clean_text)

    def is_speaking(self) -> bool:
        """Return True while the playback process is still running."""
        with self._lock:
            process = self._process

        if process is None:
            return False

        if process.poll() is None:
            return True

        self._finish(process)
        return False

    def wait_until_done(self, timeout: float | None = None) -> bool:
        """Block until playback ends; return False if the timeout expired first."""
        with self._lock:
            process = self._process

        if process is None:
            return True

        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            return False

        self._finish(process)
        return True

    def stop(self) -> None:
        """Interrupt the current speech, if any."""
        with self._lock:
            process = self._process

        if process is None:
            return

        if process.poll() is None:
            try:
                process.terminate()
                process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                try:
                    process.kill()
                except Exception:
                    pass
            except Exception:
                pass

        self._finish(process)

    def _start_process(self, command: list[str], **kwargs: Any) -> Any:
        """Start a child process, resolving ``subprocess.Popen`` at call time."""
        factory = self._popen if self._popen is not None else subprocess.Popen
        return factory(command, **kwargs)

    def _track(self, process: Any, cleanup_path: Path | None = None) -> None:
        with self._lock:
            self._process = process
            self._cleanup_path = cleanup_path

    def _finish(self, process: Any) -> None:
        with self._lock:
            if self._process is not process:
                return

            path = self._cleanup_path
            self._process = None
            self._cleanup_path = None

        if path is not None:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass

    def _sanitize_text(self, text: str) -> str:
        """Strip markdown, quotes and system markers before speech."""
        t = text
        for marker in ("```", "**", "*", "#", "[⚡", "[image_path:", "«", "»", '"', "'"):
            t = t.replace(marker, "")
        return t.strip()

    def _speak_edge(self, text: str) -> bool:
        """Synthesize with studio-quality neural voice protected by timeout."""
        tmp_path: Path | None = None
        tracked = False

        try:
            import edge_tts

            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
                tmp_path = Path(tmp.name)

            async def _generate_with_timeout() -> None:
                communicate = edge_tts.Communicate(text, self.jarvis_voice)
                # Max 4 seconds for generation, never hang indefinitely
                await asyncio.wait_for(communicate.save(str(tmp_path)), timeout=4.0)

            asyncio.run(_generate_with_timeout())

            if sys.platform == "darwin":
                process = self._start_process(["/usr/bin/afplay", str(tmp_path)])
                self._track(process, tmp_path)
                tracked = True
            elif sys.platform == "win32":
                self._start_process(["start", str(tmp_path)], shell=True)
                tracked = True
            return True
        except Exception:
            # Fallback to instant local say if edge-tts stalls
            return self._speak_system(text)
        finally:
            if tmp_path is not None and not tracked:
                try:
                    tmp_path.unlink(missing_ok=True)
                except OSError:
                    pass

    def _speak_system(self, text: str) -> bool:
        """Offline speech fallback using native OS speech."""
        if self._has_say:
            # A leading dash would be parsed by ``say`` as an option.
            safe_text = text.lstrip("-–— ")
            if not safe_text:
                return False

            try:
                process = self._start_process(["say", safe_text])
                self._track(process)
                return True
            except Exception:
                pass
        return False
