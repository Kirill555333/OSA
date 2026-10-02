"""Live audio voice runtime integration for OSA."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile

from osa.core import Agent
from osa.voice.composition import create_default_voice_runtime
from osa.voice.contracts import VoiceOutput
from osa.voice.runtime import VoiceRuntime
from osa.voice.session import VoiceSession
from osa.voice.tts import TextToSpeech
from osa.voice.vad import EnergyVoiceActivityDetector


class MacOSNativeTTS:
    """Zero-dependency macOS TextToSpeech using native /usr/bin/say."""

    def __init__(self, voice: str | None = None) -> None:
        self._voice = voice

    def synthesize(self, output: VoiceOutput) -> bytes:
        """Synthesize text into WAV audio bytes using macOS say utility."""
        text = output.text.strip()
        if not text:
            return b""

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            cmd = ["/usr/bin/say", "-o", str(tmp_path), "--data-format=LEI16@16000"]
            if self._voice:
                cmd.extend(["-v", self._voice])
            cmd.append(text)

            subprocess.run(cmd, check=True, capture_output=True, timeout=15)
            if tmp_path.exists():
                return tmp_path.read_bytes()
            return b""
        except Exception:
            return b""
        finally:
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass

    def stop(self) -> None:
        """Interrupt any running say process."""
        try:
            subprocess.run(["killall", "say"], check=False, capture_output=True)
        except Exception:
            pass


def is_live_audio_available() -> bool:
    """Return True if sounddevice, numpy and faster_whisper are installed."""
    has_sd = importlib.util.find_spec("sounddevice") is not None
    has_np = importlib.util.find_spec("numpy") is not None
    has_stt = importlib.util.find_spec("faster_whisper") is not None
    return bool(has_sd and has_np and has_stt)


def create_live_voice_runtime(agent: Agent) -> VoiceRuntime | None:
    """Create live audio VoiceRuntime if drivers are installed, else return None."""
    if not is_live_audio_available():
        return None

    try:
        from osa.voice.faster_whisper_stt import FasterWhisperConfig, FasterWhisperSpeechToText

        vad = EnergyVoiceActivityDetector()
        stt = FasterWhisperSpeechToText(FasterWhisperConfig(model="base", language="ru"))

        # On macOS prefer native TTS if piper is not installed
        if sys.platform == "darwin":
            tts: TextToSpeech = MacOSNativeTTS()
        else:
            from osa.voice.piper_tts import create_piper_tts

            tts = create_piper_tts()

        session = VoiceSession(
            agent=agent,
            vad=vad,
            stt=stt,
            tts=tts,
        )

        return create_default_voice_runtime(session)

    except Exception:
        # Fall back to standby gracefully if audio devices are unavailable
        return None
