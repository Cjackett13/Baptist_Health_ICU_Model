from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # Local path to model files directory.
    # GCP: set to a /tmp staging dir and download from gcs_bucket at startup.
    model_dir: str = "./model_files"

    # GCS bucket holding model .pkl files (blank = local-only mode).
    gcs_bucket: str = ""

    port: int = 8080
    log_level: str = "info"


settings = Settings()
