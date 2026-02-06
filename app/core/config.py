from pydantic import BaseModel


class Settings(BaseModel):
    app_name: str = "English Handwritten and Printed OCR System"
    device: str = "cuda"  # auto-resolved in model loader


settings = Settings()
