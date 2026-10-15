"""Try background listening on this computer, without starting the whole OSA agent.

Usage:
    python scripts/try_background_listening.py [--mode wake|always] [--model small] [--debug]

Say "Оса, <something>". The script repeats what it understood out loud using the system
voice, so it also exercises the microphone-while-speaking path. Press Ctrl+C to stop.
"""

from __future__ import annotations

import argparse
import sys
import time

from osa.voice.background import BackgroundListener, VoiceCommand
from osa.voice.background_audio import FasterWhisperTranscriber, SoundDeviceFrameSource
from osa.voice.engine import OSAVoiceEngine
from osa.voice.wake import ListenMode, WakeWordGate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mode", default="wake", choices=["wake", "always"])
    parser.add_argument("--model", default="small", help="Whisper size: tiny, base, small, medium")
    parser.add_argument("--debug", action="store_true", help="Print what the listener is doing")
    parser.add_argument("--show-text", action="store_true", help="Also print recognized text")
    parser.add_argument("--device", default=None, help="Input device index or name")
    args = parser.parse_args()

    device = int(args.device) if args.device and args.device.isdigit() else args.device
    engine = OSAVoiceEngine(mode="system")
    transcriber = FasterWhisperTranscriber(args.model)
    debug = (lambda message: print(f"   [debug {time.strftime('%H:%M:%S')}] {message}", flush=True))

    print(f"Загружаю модель Whisper '{args.model}'...", flush=True)
    transcriber.preload()

    gate = WakeWordGate(mode=ListenMode.parse(args.mode))
    pending: list[VoiceCommand] = []
    listener = BackgroundListener(
        SoundDeviceFrameSource(device=device),
        transcriber,
        gate,
        pending.append,
        debug=debug if (args.debug or args.show_text) else None,
        log_transcripts=args.show_text,
    )

    try:
        listener.start()
    except Exception as exc:
        print(f"Не удалось открыть микрофон: {exc}")
        return 1

    print(f"Слушаю (режим: {args.mode}). Скажите «Оса, привет». Остановить: Ctrl+C.", flush=True)

    try:
        while True:
            if listener.error is not None:
                print(f"Слушатель остановился из-за ошибки: {listener.error!r}")
                return 1

            if not pending:
                time.sleep(0.1)
                continue

            command = pending.pop(0)
            reply = "Слушаю." if command.reason == "wake_only" else f"Вы сказали: {command.text}"
            print(f"🎙 {command.reason}: {command.text or '(только обращение)'}", flush=True)

            with listener.suppressed():
                engine.speak(reply)
                engine.wait_until_done(timeout=20.0)

            listener.open_follow_up()
    except KeyboardInterrupt:
        print("\nОстанавливаю...")
    finally:
        engine.stop()
        listener.stop()

    return 0


if __name__ == "__main__":
    sys.exit(main())
