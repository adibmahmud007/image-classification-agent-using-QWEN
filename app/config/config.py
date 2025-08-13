from pydantic_settings import BaseSettings
from pydantic import Field, validator
from typing import List
import os


class Settings(BaseSettings):
    """Application settings and configuration"""
    
    # OpenRouter API Configuration
    OPENROUTER_API_KEY: str = Field(
        ..., 
        description="OpenRouter API key for accessing vision models"
    )
    OPENROUTER_URL: str = Field(
        default="https://openrouter.ai/api/v1/chat/completions",
        description="OpenRouter API endpoint URL"
    )
    
    # Model Configuration
    DEFAULT_MODEL: str = Field(
        default="qwen/qwen2.5-vl-32b-instruct:free",
        description="Default vision model to use for classification"
    )
    
    # Application Configuration
    APP_NAME: str = Field(
        default="Image Classification API",
        description="Application name"
    )
    APP_VERSION: str = Field(
        default="1.0.0",
        description="Application version"
    )
    DEBUG: bool = Field(
        default=False,
        description="Debug mode"
    )
    
    # Server Configuration
    HOST: str = Field(
        default="0.0.0.0",
        description="Server host"
    )
    PORT: int = Field(
        default=8000,
        description="Server port"
    )
    RELOAD: bool = Field(
        default=False,
        description="Auto-reload on code changes (development only)"
    )
    
    # Request Configuration
    REQUEST_TIMEOUT: float = Field(
        default=30.0,
        description="HTTP request timeout in seconds"
    )
    MAX_FILE_SIZE: int = Field(
        default=5 * 1024 * 1024,  # 5MB
        description="Maximum file size for uploaded images"
    )
    MAX_BATCH_FILES: int = Field(
        default=10,
        description="Maximum number of files in batch processing"
    )
    
    # CORS Configuration
    CORS_ORIGINS: List[str] = Field(
        default=[
            "http://localhost:3000",
            "http://localhost:8080",
            "http://127.0.0.1:3000",
            "http://127.0.0.1:8080"
        ],
        description="Allowed CORS origins"
    )
    CORS_ALLOW_CREDENTIALS: bool = Field(
        default=True,
        description="Allow credentials in CORS requests"
    )
    CORS_ALLOW_METHODS: List[str] = Field(
        default=["GET", "POST", "OPTIONS"],
        description="Allowed HTTP methods for CORS"
    )
    CORS_ALLOW_HEADERS: List[str] = Field(
        default=["*"],
        description="Allowed headers for CORS"
    )
    
    # Logging Configuration
    LOG_LEVEL: str = Field(
        default="INFO",
        description="Logging level"
    )
    
    # Rate Limiting Configuration
    RATE_LIMIT_REQUESTS: int = Field(
        default=100,
        description="Number of requests allowed per period"
    )
    RATE_LIMIT_PERIOD: int = Field(
        default=3600,  # 1 hour
        description="Rate limit period in seconds"
    )
    
    @validator("OPENROUTER_API_KEY")
    def validate_api_key(cls, v):
        """Validate that API key is provided"""
        if not v or v == "your-openrouter-api-key-here":
            raise ValueError(
                "OpenRouter API key must be provided. "
                "Get one from https://openrouter.ai and set it in .env file"
            )
        return v
    
    @validator("MAX_FILE_SIZE")
    def validate_file_size(cls, v):
        """Validate file size limit"""
        if v > 10 * 1024 * 1024:  # 10MB
            raise ValueError("Maximum file size should not exceed 10MB")
        return v
    
    @validator("LOG_LEVEL")
    def validate_log_level(cls, v):
        """Validate log level"""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of: {', '.join(valid_levels)}")
        return v.upper()
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get application settings (dependency injection)"""
    return settings