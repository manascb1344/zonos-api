import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
import numpy as np

from app.main import app
from app.models import TTSRequest
from app.services.tts import TTSService, resolve_audio_path

client = TestClient(app)

MODEL = "Zyphra/Zonos-v0.1-transformer"


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["message"] == "Zonos Text-to-Speech API"


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_get_models():
    with patch("app.main.get_tts_service") as mock_get_service:
        mock_service = Mock(spec=TTSService)
        mock_service.get_model_names.return_value = [
            "Zyphra/Zonos-v0.1-transformer",
            "Zyphra/Zonos-v0.1-hybrid",
        ]
        mock_get_service.return_value = mock_service

        response = client.get("/models")
        assert response.status_code == 200
        assert "models" in response.json()
        assert len(response.json()["models"]) == 2


def test_get_languages():
    with patch("app.main.get_tts_service") as mock_get_service:
        mock_service = Mock(spec=TTSService)
        mock_service.get_supported_languages.return_value = ["en-us", "es-es"]
        mock_get_service.return_value = mock_service

        response = client.get("/languages")
        assert response.status_code == 200
        assert "languages" in response.json()
        assert len(response.json()["languages"]) == 2


def test_get_model_conditioners():
    with patch("app.main.get_tts_service") as mock_get_service:
        mock_service = Mock(spec=TTSService)
        mock_service.get_model_names.return_value = [MODEL]
        mock_service.get_model_conditioners.return_value = ["espeak", "speaker"]
        mock_get_service.return_value = mock_service

        response = client.get(f"/model/{MODEL}/conditioners")
        assert response.status_code == 200
        assert response.json() == {"conditioners": ["espeak", "speaker"]}


def test_get_model_conditioners_not_found():
    with patch("app.main.get_tts_service") as mock_get_service:
        mock_service = Mock(spec=TTSService)
        mock_service.get_model_names.return_value = [MODEL]
        mock_get_service.return_value = mock_service

        response = client.get("/model/unknown-model/conditioners")
        assert response.status_code == 404
        assert response.json()["detail"] == "Model not found"


def test_synthesize_speech():
    with patch("app.main.get_tts_service") as mock_get_service:
        mock_service = Mock(spec=TTSService)
        mock_service.get_model_names.return_value = [MODEL]
        mock_service.generate_audio.return_value = (
            (44100, np.zeros(44100).astype(np.float32)),  # 1 second of silence
            42,  # seed
        )
        mock_get_service.return_value = mock_service

        test_request = {
            "model_choice": MODEL,
            "text": "Test speech",
            "language": "en-us",
            "emotion_values": [1.0, 0.05, 0.05, 0.05, 0.05, 0.05, 0.1, 0.2],
            "vq_score": 0.78,
            "cfg_scale": 2.0,
            "min_p": 0.15,
            "randomize_seed": False,
            "seed": 42,
        }

        response = client.post("/synthesize", json=test_request)
        assert response.status_code == 200
        assert response.headers["content-type"] == "audio/wav"
        assert response.headers["x-seed"] == "42"


def test_synthesize_text_too_long():
    response = client.post(
        "/synthesize",
        json={
            "model_choice": MODEL,
            "text": "x" * 2001,
        },
    )
    assert response.status_code == 400
    assert "exceeds maximum" in response.json()["detail"]


def test_synthesize_unknown_model():
    with patch("app.main.get_tts_service") as mock_get_service:
        mock_service = Mock(spec=TTSService)
        mock_service.get_model_names.return_value = [MODEL]
        mock_get_service.return_value = mock_service

        response = client.post(
            "/synthesize",
            json={
                "model_choice": "unknown-model",
                "text": "Test speech",
            },
        )
        assert response.status_code == 404
        assert response.json()["detail"] == "Model not found"


def test_audio_path_is_restricted_to_audio_root():
    with pytest.raises(ValueError):
        resolve_audio_path("/etc/passwd")


def test_request_mutable_defaults_are_independent():
    first = TTSRequest(model_choice=MODEL, text="First")
    second = TTSRequest(model_choice=MODEL, text="Second")
    first.emotion_values.append(9.9)
    first.unconditional_keys.append("speaker")
    assert len(second.emotion_values) == 8
    assert second.unconditional_keys == ["emotion"]


def test_openai_models_list():
    with patch("app.main.get_tts_service") as mock_get_service:
        mock_service = Mock(spec=TTSService)
        mock_service.get_model_names.return_value = [MODEL]
        mock_get_service.return_value = mock_service

        response = client.get("/v1/models")
        assert response.status_code == 200
        body = response.json()
        assert body["object"] == "list"
        ids = [item["id"] for item in body["data"]]
        assert MODEL in ids
        assert "tts-1" in ids
        assert all(item["object"] == "model" for item in body["data"])


def test_openai_models_retrieve():
    with patch("app.main.get_tts_service") as mock_get_service:
        mock_service = Mock(spec=TTSService)
        mock_service.get_model_names.return_value = [MODEL]
        mock_get_service.return_value = mock_service

        response = client.get("/v1/models/tts-1")
        assert response.status_code == 200
        assert response.json()["id"] == "tts-1"

        full_id = client.get(f"/v1/models/{MODEL}")
        assert full_id.status_code == 200
        assert full_id.json()["id"] == MODEL

        missing = client.get("/v1/models/unknown-model")
        assert missing.status_code == 404
        assert missing.json()["error"]["type"] == "invalid_request_error"


def test_openai_speech_wav():
    with patch("app.main.get_tts_service") as mock_get_service:
        mock_service = Mock(spec=TTSService)
        mock_service.get_model_names.return_value = [MODEL]
        mock_service.generate_audio.return_value = (
            (44100, np.zeros(44100).astype(np.float32)),
            42,
        )
        mock_get_service.return_value = mock_service

        response = client.post(
            "/v1/audio/speech",
            json={
                "model": "tts-1",
                "input": "Test speech",
                "response_format": "wav",
            },
        )
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("audio/wav")
        assert response.headers["x-seed"] == "42"
        assert response.content[:4] == b"RIFF"


def test_openai_speech_pcm():
    with patch("app.main.get_tts_service") as mock_get_service:
        mock_service = Mock(spec=TTSService)
        mock_service.get_model_names.return_value = [MODEL]
        mock_service.generate_audio.return_value = (
            (44100, np.zeros(100).astype(np.float32)),
            7,
        )
        mock_get_service.return_value = mock_service

        response = client.post(
            "/v1/audio/speech",
            json={
                "model": MODEL,
                "input": "Test speech",
                "response_format": "pcm",
            },
        )
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("audio/pcm")
        assert len(response.content) % 2 == 0
        assert abs(len(response.content) - 108) <= 2


def test_openai_speech_mp3_uses_ffmpeg():
    with (
        patch("app.main.get_tts_service") as mock_get_service,
        patch("app.openai_api.subprocess.run") as mock_run,
    ):
        mock_service = Mock(spec=TTSService)
        mock_service.get_model_names.return_value = [MODEL]
        mock_service.generate_audio.return_value = (
            (44100, np.zeros(100).astype(np.float32)),
            7,
        )
        mock_get_service.return_value = mock_service
        mock_run.return_value.stdout = b"mp3bytes"

        response = client.post(
            "/v1/audio/speech",
            json={
                "model": "tts-1",
                "input": "Test speech",
                "response_format": "mp3",
            },
        )
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("audio/mpeg")
        assert response.content == b"mp3bytes"
        assert "libmp3lame" in mock_run.call_args.args[0]


def test_openai_validation_errors_use_error_envelope():
    with patch("app.main.get_tts_service") as mock_get_service:
        mock_service = Mock(spec=TTSService)
        mock_service.get_model_names.return_value = [MODEL]
        mock_get_service.return_value = mock_service

        missing_input = client.post("/v1/audio/speech", json={"model": "tts-1"})
        assert missing_input.status_code == 422
        assert missing_input.json()["error"]["type"] == "invalid_request_error"
        assert missing_input.json()["error"]["param"] == "input"

        legacy = client.post("/synthesize", json={"model_choice": MODEL})
        assert legacy.status_code == 422
        assert "detail" in legacy.json()


def test_openai_speech_validation():
    with patch("app.main.get_tts_service") as mock_get_service:
        mock_service = Mock(spec=TTSService)
        mock_service.get_model_names.return_value = [MODEL]
        mock_get_service.return_value = mock_service

        bad_format = client.post(
            "/v1/audio/speech",
            json={
                "model": "tts-1",
                "input": "Hi",
                "response_format": "wma",
            },
        )
        assert bad_format.status_code == 400
        assert bad_format.json()["error"]["param"] == "response_format"

        bad_model = client.post(
            "/v1/audio/speech",
            json={
                "model": "unknown-model",
                "input": "Hi",
            },
        )
        assert bad_model.status_code == 404
        assert bad_model.json()["error"]["param"] == "model"

        bad_speed = client.post(
            "/v1/audio/speech",
            json={
                "model": "tts-1",
                "input": "Hi",
                "speed": 9.0,
            },
        )
        assert bad_speed.status_code == 400
        assert bad_speed.json()["error"]["param"] == "speed"
