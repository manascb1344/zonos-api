# Zonos API

> Warning: this API is at version 0.1.0 and is unstable. Breaking changes may occur. Use it in production at your own risk.

A FastAPI service for Zonos text-to-speech with an OpenAI-compatible speech endpoint.

## Credits

This service is built on the Zonos models created by [Zyphra](https://huggingface.co/Zyphra). The models provide zero-shot voice cloning, more than 100 languages through eSpeak NG, 44 kHz audio output, and control over speaking rate, pitch, and emotion. This release serves the [Zonos-v0.1-transformer](https://huggingface.co/Zyphra/Zonos-v0.1-transformer) variant. The [hybrid variant](https://huggingface.co/Zyphra/Zonos-v0.1-hybrid) is not served because it needs a mamba backbone. See the [model collection](https://huggingface.co/collections/Zyphra/zonos-v01-67ac661c85e1898670823b4f) and the [release blog post](https://www.zyphra.com/post/beta-release-of-zonos-v0-1) for details.

## Features

- FastAPI REST API for the Zonos transformer model
- Docker and docker-compose support with NVIDIA GPU acceleration
- Health check and structured logging
- Swagger documentation at `/docs`
- Voice cloning and audio continuation through `speaker_audio` and `prefix_audio`
- OpenAI-compatible `POST /v1/audio/speech` and model listing endpoints

## Quick Start

Use the published image.

```bash
docker pull ghcr.io/manascb1344/zonos-api:latest
docker run -d \
  --name zonos-api \
  --gpus all \
  -p 8000:8000 \
  -e CUDA_VISIBLE_DEVICES=0 \
  ghcr.io/manascb1344/zonos-api:latest
```

CI publishes this image on pushes to `main`.

## Manual Installation

Clone this repository and the Zonos model repository, then install both.

```bash
git clone https://github.com/manascb1344/zonos-api
cd zonos-api
git clone https://github.com/Zyphra/Zonos.git
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e Zonos
pip install -e .
```

Start the server.

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Running with Docker

Build and run the image locally.

```bash
docker build -t zonos-api .
docker run -d \
  --name zonos-api \
  --gpus all \
  -p 8000:8000 \
  -e CUDA_VISIBLE_DEVICES=0 \
  zonos-api
```

Or use docker-compose.

```bash
docker-compose up --build
```

## Environment Variables

- `CUDA_VISIBLE_DEVICES`: GPU selection (default: 0)
- `PYTHONPATH`: set to `/app:/app/Zonos` inside the container
- `TTS_DEVICE`: torch device for inference (default: cuda)
- `LOG_LEVEL`: Python log level (default: INFO)
- `MAX_CHARACTERS`: maximum input length (default: 2000)
- `AUDIO_ROOT`: directory that `speaker_audio` and `prefix_audio` paths must stay inside (default: /app/audio)
- `HF_HOME`: Hugging Face cache directory (default: /app/.cache/huggingface in the container)

## Requirements

- Docker with NVIDIA Container Toolkit
- NVIDIA GPU with CUDA support
- At least 8 GB of GPU memory

## Verifying the Installation

```bash
curl http://localhost:8000/health
```

The response is `{"status":"ok"}`.

## API Endpoints

- `GET /`: basic API information
- `GET /health`: liveness check that does not load the model
- `GET /models`: available TTS models
- `GET /languages`: supported language codes
- `GET /model/{model_name}/conditioners`: conditioners for one model
- `POST /synthesize`: generate speech from text
- `GET /v1/models`: OpenAI-compatible model list
- `GET /v1/models/{model_id}`: OpenAI-compatible model details
- `POST /v1/audio/speech`: OpenAI-compatible speech generation

### POST /v1/audio/speech

This endpoint accepts the standard OpenAI request fields. `model` accepts a Zonos model ID or the aliases `tts-1`, `tts-1-hd`, and `gpt-4o-mini-tts`. `input` is the text to speak. `speed` ranges from 0.25 to 4.0 and scales the Zonos speaking rate. `voice` and `instructions` are accepted for compatibility but do not change the Zonos voice. `response_format` accepts `wav`, `pcm`, `mp3`, `opus`, `aac`, and `flac`. The compressed formats require `ffmpeg` on the server. Errors use the OpenAI `{"error": {...}}` shape.

```json
{
  "model": "tts-1",
  "input": "Hello, this is a test.",
  "voice": "alloy",
  "response_format": "wav",
  "speed": 1.0
}
```

### POST /synthesize

```json
{
  "model_choice": "Zyphra/Zonos-v0.1-transformer",
  "text": "Hello, this is a test.",
  "language": "en-us",
  "emotion_values": [1.0, 0.05, 0.05, 0.05, 0.05, 0.05, 0.1, 0.2],
  "vq_score": 0.78,
  "cfg_scale": 2.0,
  "min_p": 0.15
}
```

The response is a WAV stream with an `x-seed` header containing the seed used for generation. `model_choice` must match an entry from `/models`. Text longer than 2000 characters returns HTTP 400.

## Development

Install the test dependencies from `requirements.txt`, then run the tests.

```bash
pytest
```

Start the server in development mode.

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for environment setup, coding conventions, and the pull request process.

## License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE).
