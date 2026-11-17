import logging
import threading
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import torch
import torchaudio
from zonos.model import Zonos
from zonos.conditioning import make_cond_dict, supported_language_codes

from ..config import AUDIO_ROOT, TTS_DEVICE


def resolve_audio_path(path: str) -> str:
    """Return an absolute audio path inside AUDIO_ROOT."""
    candidate = Path(path).expanduser().resolve()
    root = Path(AUDIO_ROOT).expanduser().resolve()
    if not candidate.is_relative_to(root):
        raise ValueError(f"Audio path must be inside {root}")
    return str(candidate)


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TTSService:
    def __init__(self, device: str = TTS_DEVICE):
        self.device = device
        self.model_names = ["Zyphra/Zonos-v0.1-transformer"]
        self.models = {
            name: Zonos.from_pretrained(name, device=device, backbone="torch")
            for name in self.model_names
        }
        self._lock = threading.Lock()

        logger.info("Loaded models: %s", list(self.models))

        # Set models to eval mode
        for model in self.models.values():
            model.requires_grad_(False).eval()

    def get_model_names(self) -> List[str]:
        logger.info("Retrieving model names")
        return self.model_names

    def get_supported_languages(self) -> List[str]:
        logger.info("Retrieving supported languages")
        return supported_language_codes

    def get_model_conditioners(self, model_choice: str) -> List[str]:
        """Get list of conditioner names for a model"""
        logger.info(f"Retrieving conditioners for model: {model_choice}")
        model = self.models[model_choice]
        return [c.name for c in model.prefix_conditioner.conditioners]

    def generate_audio(
        self,
        model_choice: str,
        text: str,
        language: str = "en-us",
        speaker_audio: Optional[str] = None,
        prefix_audio: Optional[str] = None,
        emotion_values: Optional[List[float]] = None,
        vq_score: float = 0.78,
        fmax: float = 24000,
        pitch_std: float = 45.0,
        speaking_rate: float = 15.0,
        dnsmos_ovrl: float = 4.0,
        speaker_noised: bool = False,
        cfg_scale: float = 2.0,
        min_p: float = 0.15,
        seed: int = 420,
        randomize_seed: bool = True,
        unconditional_keys: Optional[List[str]] = None,
        top_p: float = 0.95,
        top_k: int = 50,
        linear: float = 1.0,
        confidence: float = 0.1,
        quadratic: float = 1.0,
    ) -> Tuple[Tuple[int, np.ndarray], int]:
        """
        Generate audio using the specified model and parameters.
        Returns a tuple of ((sample_rate, audio_data), seed).
        """
        if emotion_values is None:
            emotion_values = [1.0, 0.05, 0.05, 0.05, 0.05, 0.05, 0.1, 0.2]
        if unconditional_keys is None:
            unconditional_keys = ["emotion"]

        with self._lock:
            logger.info(f"Generating audio for model: {model_choice}")
            selected_model = self.models[model_choice]

            if randomize_seed:
                seed = torch.randint(0, 2**32 - 1, (1,)).item()
            torch.manual_seed(seed)

            # Process speaker audio if provided
            speaker_embedding = None
            if speaker_audio is not None and "speaker" not in unconditional_keys:
                logger.info(f"Processing speaker audio: {speaker_audio}")
                wav, sr = torchaudio.load(resolve_audio_path(speaker_audio))
                speaker_embedding = selected_model.make_speaker_embedding(wav, sr)
                speaker_embedding = speaker_embedding.to(
                    self.device, dtype=torch.bfloat16
                )

            # Process prefix audio if provided
            audio_prefix_codes = None
            if prefix_audio is not None:
                logger.info(f"Processing prefix audio: {prefix_audio}")
                wav_prefix, sr_prefix = torchaudio.load(
                    resolve_audio_path(prefix_audio)
                )
                wav_prefix = wav_prefix.mean(0, keepdim=True)
                wav_prefix = torchaudio.functional.resample(
                    wav_prefix, sr_prefix, selected_model.autoencoder.sampling_rate
                )
                wav_prefix = wav_prefix.to(self.device, dtype=torch.float32)
                with torch.autocast(self.device, dtype=torch.float32):
                    audio_prefix_codes = selected_model.autoencoder.encode(
                        wav_prefix.unsqueeze(0)
                    )

            # Prepare emotion tensor
            emotion_tensor = torch.tensor(emotion_values, device=self.device)

            # Prepare VQ score tensor
            vq_tensor = torch.tensor([vq_score] * 8, device=self.device).unsqueeze(0)

            # Create conditioning dictionary
            logger.info("Creating conditioning dictionary")
            cond_dict = make_cond_dict(
                text=text,
                language=language,
                speaker=speaker_embedding,
                emotion=emotion_tensor,
                vqscore_8=vq_tensor,
                fmax=float(fmax),
                pitch_std=float(pitch_std),
                speaking_rate=float(speaking_rate),
                dnsmos_ovrl=float(dnsmos_ovrl),
                speaker_noised=bool(speaker_noised),
                device=self.device,
                unconditional_keys=unconditional_keys,
            )
            conditioning = selected_model.prepare_conditioning(cond_dict)

            # sampling parameters
            sampling_params = {
                "top_p": float(top_p),
                "top_k": int(top_k),
                "min_p": float(min_p),
                "linear": float(linear),
                "conf": float(confidence),
                "quad": float(quadratic),
            }

            # Generate audio
            logger.info("Generating audio")
            max_new_tokens = 86 * 60  # ~30 seconds of audio
            codes = selected_model.generate(
                prefix_conditioning=conditioning,
                audio_prefix_codes=audio_prefix_codes,
                max_new_tokens=max_new_tokens,
                cfg_scale=float(cfg_scale),
                batch_size=1,
                sampling_params=sampling_params,
            )

            # Decode generated codes to waveform
            logger.info("Decoding generated audio to waveform")
            wav_out = selected_model.autoencoder.decode(codes).cpu().detach()
            sr_out = selected_model.autoencoder.sampling_rate
            if wav_out.dim() == 2 and wav_out.size(0) > 1:
                wav_out = wav_out[0:1, :]

            logger.info("Audio generation complete")
            return (sr_out, wav_out.squeeze().numpy()), seed
