from pydantic_settings import BaseSettings, SettingsConfigDict
import torch

class Settings(BaseSettings):
    # Model configuration
    model_name: str = "Qwen/Qwen2.5-0.5B"
    model_version: str = "jev-lite-0.1"

    # Device management: auto-detect cCUDA, fallback to CPU
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # calibration parameters
    default_temperature: float = 1.0

    # parallel batch chunk size for GPU VRAM safety
    batch_chunk_size: int = 32

    # scheduler settings
    scheduler_max_batch_size : int = 16  #maximum request to bundle per batch
    scheduler_max_delay_ms : float = 10.0 # maximum collection window (10ms)
    scheduler_max_queue_size: int = 1024  # backpressure limit ( rejects 429 if full)

    # API coonfiguration
    api_title: str = "jev-Lite System one Engine"
    api_version: str = "v1"

    # Allow reading from a .env file if present
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding = "utf-8")

settings = Settings()

