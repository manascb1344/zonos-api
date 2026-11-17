from typing import List, Optional
from pydantic import BaseModel, Field


class TTSRequest(BaseModel):
    model_choice: str = Field(description="Model variant to use")
    text: str = Field(description="Text to convert to speech")
    language: str = Field(description="Language code", default="en-us")
    speaker_audio: Optional[str] = Field(
        default=None, description="Path to speaker audio file"
    )
    prefix_audio: Optional[str] = Field(
        default=None, description="Path to prefix audio file"
    )
    emotion_values: List[float] = Field(
        default_factory=lambda: [1.0, 0.05, 0.05, 0.05, 0.05, 0.05, 0.1, 0.2],
        description="List of 8 emotion values: happiness, sadness, disgust, fear, surprise, anger, other, neutral",
    )
    vq_score: float = Field(default=0.78, description="VQ Score value")
    fmax: float = Field(default=24000, description="Maximum frequency")
    pitch_std: float = Field(default=45.0, description="Pitch standard deviation")
    speaking_rate: float = Field(default=15.0, description="Speaking rate")
    dnsmos_ovrl: float = Field(default=4.0, description="DNSMOS overall score")
    speaker_noised: bool = Field(
        default=False, description="Whether to denoise speaker audio"
    )
    cfg_scale: float = Field(default=2.0, description="CFG scale value")
    min_p: float = Field(default=0.15, description="Minimum probability")
    seed: int = Field(default=420, description="Random seed")
    randomize_seed: bool = Field(default=True, description="Whether to randomize seed")
    unconditional_keys: List[str] = Field(
        default_factory=lambda: ["emotion"],
        description="List of conditioning keys to make unconditional",
    )
    top_p: float = Field(default=0.95, description="Top-p sampling value")
    top_k: int = Field(default=50, description="Top-k sampling value")
    linear: float = Field(default=1.0, description="Linear scaling factor")
    confidence: float = Field(default=0.1, description="Confidence scaling factor")
    quadratic: float = Field(default=1.0, description="Quadratic scaling factor")
