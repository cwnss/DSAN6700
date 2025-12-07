from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    ENVIRONMENT: str = "development"

    # Database
    DATABASE_URL: str
    DB_PORT: str
    DB_USER: str
    DB_PASSWORD: str
    DB_NAME:str

    # AWS
    AWS_REGION: str
    AWS_ACCESS_KEY_ID: str
    AWS_SECRET_ACCESS_KEY: str

    #Buckets
    S3_BUCKET_DOCUMENTS: str
    S3_BUCKET_IMAGES: str

    # Model files
    EMBEDDING_PATH: str
    LABEL_PATH: str

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
print("DEBUG SETTINGS:", settings.AWS_ACCESS_KEY_ID, settings.AWS_SECRET_ACCESS_KEY)
