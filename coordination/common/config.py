from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    run_dir: str = ".run"
    data_dir: str = "data"
    inferences_dir: str = ".run/inferences"
    evaluations_dir: str = ".run/evaluations"
    webapp_run_dir: str = ".webapp"
    pytensor_comp_dir = ".pytensor_compiles"


settings = Settings()
