from pydantic import BaseModel, Field


class AudioConfig(BaseModel):
    sample_rate: int = Field(default=16000, description="Input sample rate for VAD/STT")
    tts_sample_rate: int = Field(default=24000, description="Output sample rate from Kokoro")
    chunk_ms: int = Field(default=32, description="Buffer size in milliseconds (Silero VAD needs 32ms)")
    channels: int = Field(default=1, description="Mono audio required")

    @property
    def chunk_samples(self) -> int:
        return int(self.sample_rate * self.chunk_ms / 1000)


class VADConfig(BaseModel):
    threshold: float = Field(default=0.5, ge=0.0, le=1.0, description="Speech probability threshold")
    silence_timeout_ms: int = Field(default=650, description="Consecutive silence before cutting utterance")
    min_speech_ms: int = Field(default=250, description="Minimum duration to be considered valid speech")


class ModelConfig(BaseModel):
    stt_model: str = Field(default="moonshine/base")
    tts_voice: str = Field(default="af_heart")
    tts_speed: float = Field(default=1.0)
    llm_model: str = Field(default="openai/gpt-oss-20b", description="Groq model id")
    llm_system_prompt: str = Field(
        default=(
            "You are a helpful voice assistant. Keep responses concise, "
            "conversational, and under 3 sentences when possible. Do not use "
            "markdown, emojis, or stage directions like *smiles*."
        )
    )
    llm_max_tokens: int = Field(default=256)
    llm_max_history_turns: int = Field(
        default=10,
        description=(
            "Max user+assistant turn pairs kept in conversation_history. Older "
            "turns are dropped once exceeded — otherwise the full history is "
            "resent to Groq every request and grows unboundedly for a "
            "long-running conversation, eventually hitting the model's context "
            "window and increasing latency/cost along the way."
        ),
    )


class AudioProcessingConfig(BaseModel):
    highpass_cutoff_hz: int = Field(default=80)
    enable_noise_reduction: bool = Field(default=True, description="Applied once per utterance, not per-chunk")
    target_lufs: float = Field(default=-16.0)
    tts_apply_effects: bool = Field(
        default=False,
        description=(
            "De-ess/compress/LUFS-normalize Kokoro's output. Off by default: Kokoro's "
            "raw audio is already clean, and this chain runs per streamed chunk (not "
            "once per utterance), so filter/loudness state resets each call and "
            "produces audible artifacts (clicks, pumping) rather than improving it."
        ),
    )


class AppConfig(BaseModel):
    audio: AudioConfig = AudioConfig()
    vad: VADConfig = VADConfig()
    models: ModelConfig = ModelConfig()
    processing: AudioProcessingConfig = AudioProcessingConfig()
