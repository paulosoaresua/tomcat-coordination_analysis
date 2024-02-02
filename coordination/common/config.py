from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    run_dir: str = ".run"
    data_dir: str = "data"
    inferences_dir: str = ".run/inferences"


settings = Settings()
