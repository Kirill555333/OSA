"""Observability wrapper for the OSA voice session."""

from __future__ import annotations

from typing import Any

from osa.observability.logger import ObservabilityLogger
from osa.voice.session import (
    VoiceSession,
    VoiceSessionResult,
)


class ObservableVoiceSessionError(
    RuntimeError
):
    """Raised when voice observability integration is invalid."""


class ObservableVoiceSession:
    """
    Observe VoiceSession lifecycle without changing voice semantics.

    Logger failures are deliberately swallowed. Observability is never
    authoritative and cannot turn a voice operation into success or failure.
    """

    def __init__(
        self,
        session: VoiceSession,
        logger: ObservabilityLogger,
        *,
        session_id: str | None = None,
        run_id: str | None = None,
        task_id: str | None = None,
        request_id: str | None = None,
    ) -> None:
        if session is None:
            raise ValueError(
                "session is required."
            )

        if logger is None:
            raise ValueError(
                "logger is required."
            )

        self._session = session
        self._logger = logger
        self._session_id = self._normalize_identifier(
            session_id,
            "session_id",
        )
        self._run_id = self._normalize_identifier(
            run_id,
            "run_id",
        )
        self._task_id = self._normalize_identifier(
            task_id,
            "task_id",
        )
        self._request_id = self._normalize_identifier(
            request_id,
            "request_id",
        )

    @property
    def session(self) -> VoiceSession:
        """Return the wrapped voice session."""
        return self._session

    @property
    def logger(self) -> ObservabilityLogger:
        """Return the configured observability logger."""
        return self._logger

    @property
    def session_id(self) -> str | None:
        """Return the optional voice session correlation identifier."""
        return self._session_id

    @property
    def state(self):
        """Return the wrapped session state."""
        return self._session.state

    def process_audio(
        self,
        audio: bytes,
        *,
        sample_rate_hz: int,
        channels: int = 1,
    ) -> VoiceSessionResult | None:
        """Observe one voice processing cycle."""
        self._log(
            "voice.session.started",
            audio_present=bool(audio),
            audio_length=len(audio),
        )

        self._log(
            "voice.listening.started",
        )

        try:
            result = self._session.process_audio(
                audio,
                sample_rate_hz=sample_rate_hz,
                channels=channels,
            )
        except Exception as exc:
            self._log(
                "voice.listening.failed",
                error=str(exc),
            )
            self._log(
                "voice.session.failed",
                error=str(exc),
            )
            raise

        if result is None:
            self._log(
                "voice.listening.completed",
                speech_detected=False,
            )
            self._log(
                "voice.session.completed",
                speech_detected=False,
            )
            return None

        self._log(
            "voice.listening.completed",
            speech_detected=True,
        )

        self._log(
            "voice.transcription.completed",
            transcript_present=True,
            transcript_length=len(
                result.transcript.text
            ),
        )

        self._log(
            "voice.response.started",
        )

        self._log(
            "voice.response.completed",
            response_present=True,
            response_length=len(
                result.response.text
            ),
        )

        self._log(
            "voice.tts.completed",
            output_present=bool(result.audio),
            output_length=len(result.audio),
        )

        self._log(
            "voice.session.completed",
            speech_detected=True,
            response_present=True,
        )

        return result

    def complete_speaking(self) -> None:
        """Observe completion of voice playback."""
        try:
            self._session.complete_speaking()
        except Exception as exc:
            self._log(
                "voice.session.failed",
                error=str(exc),
            )
            raise

        self._log(
            "voice.response.playback.completed",
        )

    def interrupt(self) -> None:
        """Observe and safely forward a TTS interruption."""
        try:
            self._session.interrupt()
        except Exception as exc:
            self._log(
                "voice.tts.failed",
                error=str(exc),
            )
            self._log(
                "voice.session.failed",
                error=str(exc),
            )
            raise

        self._log(
            "voice.response.interrupted",
        )

    def reset(self) -> None:
        """Reset the wrapped session."""
        try:
            self._session.reset()
        except Exception as exc:
            self._log(
                "voice.session.failed",
                error=str(exc),
            )
            raise

        self._log(
            "voice.session.reset",
        )

    def stop(self) -> None:
        """Stop the wrapped session."""
        try:
            self._session.stop()
        except Exception as exc:
            self._log(
                "voice.session.failed",
                error=str(exc),
            )
            raise

        self._log(
            "voice.session.stopped",
        )

    def _log(
        self,
        event: str,
        **data: Any,
    ) -> None:
        metadata = {
            "session_id": self._session_id,
            "run_id": self._run_id,
            "task_id": self._task_id,
            "request_id": self._request_id,
            **data,
        }

        metadata = {
            key: value
            for key, value in metadata.items()
            if value is not None
        }

        try:
            self._logger.log(
                event,
                **metadata,
            )
        except Exception:
            return

    @staticmethod
    def _normalize_identifier(
        value: str | None,
        field_name: str,
    ) -> str | None:
        if value is None:
            return None

        if not isinstance(
            value,
            str,
        ):
            raise TypeError(
                f"{field_name} must be a string or None."
            )

        normalized = value.strip()

        if not normalized:
            raise ValueError(
                f"{field_name} cannot be empty."
            )

        return normalized
