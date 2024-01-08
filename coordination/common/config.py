from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    data_dir: str = "data"
    inferences_dir: str = ".run/inferences"


settings = Settings()
