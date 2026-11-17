import io
import logging
import sys
import threading
import wave
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse

from .config import LOG_LEVEL, MAX_CHARACTERS
from .models import TTSRequest
from .openai_api import OpenAIError, create_router
from .services.tts import TTSService

# Configure logging
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("zonos-tts-api")

# Initialize FastAPI app
app = FastAPI(
    title="Zonos Text-to-Speech API",
    description="API for generating high-quality speech using Zonos models",
    version="0.1.0",
)

# TTS service instance
_tts_service: Optional[TTSService] = None
_service_lock = threading.Lock()


def get_tts_service() -> TTSService:
    """Get or initialize TTS service singleton."""
    global _tts_service
    if _tts_service is None:
        with _service_lock:
            if _tts_service is None:
                logger.info("Initializing TTS service")
                _tts_service = TTSService()
    return _tts_service


@app.exception_handler(OpenAIError)
async def openai_error_handler(request, exc: OpenAIError):
    return JSONResponse(status_code=exc.status_code, content={"error": exc.payload})


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request, exc: RequestValidationError):
    errors = jsonable_encoder(exc.errors())
    if request.url.path.startswith("/v1"):
        first = errors[0] if errors else {}
        location = first.get("loc", [])
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "message": first.get("msg", "Invalid request"),
                    "type": "invalid_request_error",
                    "param": str(location[-1]) if location else None,
                    "code": None,
                }
            },
        )
    return JSONResponse(status_code=422, content={"detail": errors})


app.include_router(create_router(lambda: get_tts_service()))


@app.get("/")
async def root():
    """Root endpoint."""
    logger.debug("Root endpoint accessed")
    return {"message": "Zonos Text-to-Speech API"}


@app.get("/health")
async def health():
    """Health endpoint. Does not load the model."""
    return {"status": "ok"}


@app.get("/models")
def get_models():
    """Get available TTS models."""
    logger.debug("Fetching available models")
    service = get_tts_service()
    models = service.get_model_names()
    logger.info(f"Retrieved {len(models)} models")
    return {"models": models}


@app.get("/languages")
def get_languages():
    """Get supported languages."""
    logger.debug("Fetching supported languages")
    service = get_tts_service()
    languages = service.get_supported_languages()
    logger.info(f"Retrieved {len(languages)} supported languages")
    return {"languages": languages}


@app.get("/model/{model_name:path}/conditioners")
def get_model_conditioners(model_name: str):
    """Get available conditioners for a specific model."""
    logger.debug(f"Fetching conditioners for model: {model_name}")
    service = get_tts_service()
    logger.info(f"Requested model: {model_name}")
    logger.debug(f"Available models: {service.get_model_names()}")
    if model_name not in service.get_model_names():
        logger.warning(f"Model not found: {model_name}")
        raise HTTPException(status_code=404, detail="Model not found")
    conditioners = service.get_model_conditioners(model_name)
    logger.info(f"Retrieved {len(conditioners)} conditioners for model {model_name}")
    return {"conditioners": conditioners}


@app.post("/synthesize")
def synthesize_speech(request: TTSRequest):
    """Generate speech from text."""
    logger.info(
        f"Speech synthesis requested for text of length {len(request.text)} chars using model {request.model_choice}"
    )

    if len(request.text) > MAX_CHARACTERS:
        logger.warning(
            f"Text length ({len(request.text)}) exceeds maximum of {MAX_CHARACTERS} characters"
        )
        raise HTTPException(
            status_code=400,
            detail=f"Text length exceeds maximum of {MAX_CHARACTERS} characters",
        )

    service = get_tts_service()
    if request.model_choice not in service.get_model_names():
        logger.warning(f"Model not found: {request.model_choice}")
        raise HTTPException(status_code=404, detail="Model not found")

    try:
        logger.debug(
            f"Starting audio generation with params: language={request.language}, "
            f"speaking_rate={request.speaking_rate}, seed={request.seed}"
        )

        # Generate audio using TTS service
        (sample_rate, audio_data), seed = service.generate_audio(
            model_choice=request.model_choice,
            text=request.text,
            language=request.language,
            speaker_audio=request.speaker_audio,
            prefix_audio=request.prefix_audio,
            emotion_values=request.emotion_values,
            vq_score=request.vq_score,
            fmax=request.fmax,
            pitch_std=request.pitch_std,
            speaking_rate=request.speaking_rate,
            dnsmos_ovrl=request.dnsmos_ovrl,
            speaker_noised=request.speaker_noised,
            cfg_scale=request.cfg_scale,
            min_p=request.min_p,
            seed=request.seed,
            randomize_seed=request.randomize_seed,
            unconditional_keys=request.unconditional_keys,
            top_p=request.top_p,
            top_k=request.top_k,
            linear=request.linear,
            confidence=request.confidence,
            quadratic=request.quadratic,
        )

        logger.info(
            f"Successfully generated audio with sample rate {sample_rate}Hz, seed {seed}"
        )

        # Convert to WAV format
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)  # 16-bit audio
            wav_file.setframerate(sample_rate)
            wav_file.writeframes((audio_data * 32767).astype(np.int16).tobytes())

        # Prepare response
        wav_buffer.seek(0)
        response = StreamingResponse(
            wav_buffer, media_type="audio/wav", headers={"x-seed": str(seed)}
        )

        logger.debug("Returning WAV audio stream response")
        return response

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        logger.exception("Error during speech synthesis")
        raise HTTPException(status_code=500, detail="Internal server error")
