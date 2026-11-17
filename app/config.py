import os

MAX_CHARACTERS = int(os.getenv("MAX_CHARACTERS", "2000"))
TTS_DEVICE = os.getenv("TTS_DEVICE", "cuda")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
AUDIO_ROOT = os.getenv("AUDIO_ROOT", "/app/audio")
