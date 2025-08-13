from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class ImageClassificationRequest(BaseModel):
    """Request schema for image classification"""
    max_tokens: Optional[int] = Field(
        default=50, 
        ge=10, 
        le=500, 
        description="Maximum tokens for response"
    )
    temperature: Optional[float] = Field(
        default=0.1, 
        ge=0.0, 
        le=1.0, 
        description="Temperature for model response"
    )


class ImageClassificationResponse(BaseModel):
    """Response schema for image classification"""
    success: bool = Field(description="Whether the classification was successful")
    classification: str = Field(description="The classification result")
    filename: str = Field(description="Original filename")
    file_size: int = Field(description="File size in bytes")
    processing_time: float = Field(description="Time taken for processing")
    timestamp: datetime = Field(description="When the classification was done")


class ErrorResponse(BaseModel):
    """Error response schema"""
    success: bool = False
    error: str = Field(description="Error message")
    error_code: Optional[str] = Field(default=None, description="Error code")
    timestamp: datetime = Field(description="When the error occurred")


class HealthCheckResponse(BaseModel):
    """Health check response schema"""
    status: str = Field(description="API status")
    message: str = Field(description="Status message")
    version: str = Field(description="API version")
    models_available: bool = Field(description="Whether models are accessible")
    timestamp: datetime = Field(description="Current timestamp")