"""Multi-engine Text-to-Speech (TTS) architecture for OSA with timeout protection."""

from __future__ import annotations

import asyncio
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Literal

VoiceMode = Literal["jarvis", "system", "clone"]


class OSAVoiceEngine:
    """Manages playback and live switching of TTS voice profiles."""

    def __init__(self, mode: VoiceMode = "jarvis") -> None:
        self.mode: VoiceMode = mode
        self.jarvis_voice: str = "ru-RU-DmitryNeural"
        self.system_voice: str = "Milena" if sys.platform == "darwin" else ""
        self.clone_sample_path: Path | None = None
        self._has_say = shutil.which("say") is not None and sys.platform == "darwin"

    def set_mode(self, mode: VoiceMode, clone_path: str | Path | None = None) -> None:
        """Switch voice provider on the fly."""
        self.mode = mode
        if clone_path:
            self.clone_sample_path = Path(clone_path)

    def speak(self, text: str) -> bool:
        """Speak text without ever freezing the main thread."""
        clean_text = self._sanitize_text(text)
        if not clean_text:
            return False

        if self.mode == "jarvis":
            success = self._speak_edge(clean_text)
            if not success:
                return self._speak_system(clean_text)
            return True
        else:
            return self._speak_system(clean_text)

    def _sanitize_text(self, text: str) -> str:
        """Strip markdown, quotes and system markers before speech."""
        t = text
        for marker in ("```", "**", "*", "#", "[⚡", "[image_path:", "«", "»", '"', "'"):
            t = t.replace(marker, "")
        return t.strip()

    def _speak_edge(self, text: str) -> bool:
        """Synthesize with studio-quality neural voice protected by timeout."""
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
                subprocess.Popen(["/usr/bin/afplay", str(tmp_path)])
            elif sys.platform == "win32":
                subprocess.Popen(["start", str(tmp_path)], shell=True)
            return True
        except Exception:
            # Fallback to instant local say if edge-tts stalls
            return self._speak_system(text)

    def _speak_system(self, text: str) -> bool:
        """Offline speech fallback using native OS speech."""
        if self._has_say:
            try:
                subprocess.Popen(["say", text])
                return True
            except Exception:
                pass
        return False
