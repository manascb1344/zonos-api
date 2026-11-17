import io
import logging
import subprocess
import wave
from typing import Callable, Optional

import numpy as np
from fastapi import APIRouter
from fastapi.responses import Response
from pydantic import BaseModel, Field

from .config import MAX_CHARACTERS

logger = logging.getLogger("zonos-tts-api")

OPENAI_MODEL_ALIASES = {
    "tts-1": "Zyphra/Zonos-v0.1-transformer",
    "tts-1-hd": "Zyphra/Zonos-v0.1-transformer",
    "gpt-4o-mini-tts": "Zyphra/Zonos-v0.1-transformer",
}
SUPPORTED_FORMATS = {
    "wav": "audio/wav",
    "pcm": "audio/pcm",
    "mp3": "audio/mpeg",
    "opus": "audio/ogg",
    "aac": "audio/aac",
    "flac": "audio/flac",
}
FFMPEG_CODECS = {
    "mp3": "libmp3lame",
    "opus": "libopus",
    "aac": "aac",
    "flac": "flac",
}
DEFAULT_SPEAKING_RATE = 15.0


def encode_audio(audio_data, sample_rate: int, response_format: str) -> bytes:
    """Encode mono float audio into an OpenAI response format."""
    pcm = (audio_data * 32767).astype(np.int16).tobytes()
    if response_format == "pcm":
        return pcm

    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm)
    wav_bytes = wav_buffer.getvalue()
    if response_format == "wav":
        return wav_bytes

    try:
        result = subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-f",
                "wav",
                "-i",
                "-",
                "-c:a",
                FFMPEG_CODECS[response_format],
                "-f",
                response_format,
                "-",
            ],
            input=wav_bytes,
            capture_output=True,
            check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        logger.exception("ffmpeg encoding failed for %s", response_format)
        raise OpenAIError(
            500, "Audio encoding failed", error_type="server_error"
        ) from exc
    return result.stdout


class OpenAIError(Exception):
    def __init__(
        self,
        status_code: int,
        message: str,
        error_type: str = "invalid_request_error",
        param: Optional[str] = None,
        code: Optional[str] = None,
    ):
        self.status_code = status_code
        self.payload = {
            "message": message,
            "type": error_type,
            "param": param,
            "code": code,
        }
        super().__init__(message)


class SpeechRequest(BaseModel):
    model: str = Field(description="Model ID or OpenAI TTS model alias")
    input: str = Field(description="Text to convert to speech")
    voice: str = Field(default="alloy", description="Accepted but not used by Zonos")
    response_format: str = Field(default="wav", description="wav or pcm")
    speed: float = Field(
        default=1.0, description="Speed multiplier between 0.25 and 4.0"
    )


class ModelObject(BaseModel):
    id: str
    object: str = "model"
    created: int = 0
    owned_by: str = "zyphra"


class ModelList(BaseModel):
    object: str = "list"
    data: list[ModelObject]


def create_router(get_service: Callable):
    router = APIRouter(prefix="/v1")

    @router.get("/models", response_model=ModelList)
    def list_models():
        service = get_service()
        ids = list(service.get_model_names()) + list(OPENAI_MODEL_ALIASES)
        return ModelList(data=[ModelObject(id=model_id) for model_id in ids])

    @router.get("/models/{model_id}", response_model=ModelObject)
    def retrieve_model(model_id: str):
        service = get_service()
        if (
            model_id not in service.get_model_names()
            and model_id not in OPENAI_MODEL_ALIASES
        ):
            raise OpenAIError(404, f"Unknown model '{model_id}'", param="model")
        return ModelObject(id=model_id)

    @router.post("/audio/speech")
    def create_speech(request: SpeechRequest):
        service = get_service()
        model_choice = OPENAI_MODEL_ALIASES.get(request.model, request.model)
        if model_choice not in service.get_model_names():
            raise OpenAIError(404, f"Unknown model '{request.model}'", param="model")
        if not request.input.strip():
            raise OpenAIError(400, "input must not be empty", param="input")
        if len(request.input) > MAX_CHARACTERS:
            raise OpenAIError(
                400, f"input exceeds {MAX_CHARACTERS} characters", param="input"
            )
        if request.response_format not in SUPPORTED_FORMATS:
            raise OpenAIError(
                400,
                f"response_format must be one of {', '.join(sorted(SUPPORTED_FORMATS))}",
                param="response_format",
            )
        if not 0.25 <= request.speed <= 4.0:
            raise OpenAIError(400, "speed must be between 0.25 and 4.0", param="speed")

        try:
            (sample_rate, audio_data), seed = service.generate_audio(
                model_choice=model_choice,
                text=request.input,
                speaking_rate=DEFAULT_SPEAKING_RATE * request.speed,
            )
        except Exception:
            logger.exception("OpenAI speech request failed")
            raise OpenAIError(500, "Internal server error", error_type="server_error")

        content = encode_audio(audio_data, sample_rate, request.response_format)

        return Response(
            content=content,
            media_type=SUPPORTED_FORMATS[request.response_format],
            headers={"x-seed": str(seed)},
        )

    return router
