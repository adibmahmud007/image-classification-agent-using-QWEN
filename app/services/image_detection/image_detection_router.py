from fastapi import APIRouter, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from datetime import datetime
import time
from typing import Optional

from app.services.image_detection.image_detection_schema import (
    ImageClassificationResponse,
    ErrorResponse,
    HealthCheckResponse
)
from app.services.image_detection.image_detection import ImageDetectionService
from app.config.config import settings

router = APIRouter(
    prefix="/api/v1/image-detection",
    tags=["Image Detection"],
    responses={
        500: {"model": ErrorResponse, "description": "Internal server error"},
        400: {"model": ErrorResponse, "description": "Bad request"}
    }
)

# Initialize service
image_service = ImageDetectionService()


@router.get(
    "/health",
    response_model=HealthCheckResponse,
    summary="Health Check",
    description="Check if the image detection service is running properly"
)
async def health_check():
    """Health check endpoint"""
    try:
        # Test OpenRouter connection
        models_available = await image_service.test_connection()
        
        return HealthCheckResponse(
            status="healthy" if models_available else "degraded",
            message="Image Detection Service is running" if models_available else "Service running but API connection issue",
            version="1.0.0",
            models_available=models_available,
            timestamp=datetime.now()
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Health check failed: {str(e)}"
        )


@router.post(
    "/classify",
    response_model=ImageClassificationResponse,
    summary="Classify Image",
    description="Upload an image and get classification result"
)
async def classify_image(
    file: UploadFile = File(..., description="Image file to classify"),
    max_tokens: Optional[int] = Form(default=50, description="Max tokens for response"),
    temperature: Optional[float] = Form(default=0.1, description="Temperature for model")
):
    """
    Classify an uploaded image using Qwen 2.5 VL 32B model
    
    - **file**: Image file (JPEG, PNG, GIF, WebP)
    - **max_tokens**: Maximum tokens for response (10-500)
    - **temperature**: Model temperature (0.0-1.0)
    """
    
    start_time = time.time()
    
    # Validate file
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400, 
            detail="File must be an image (JPEG, PNG, GIF, WebP)"
        )
    
    # Check file size (5MB limit)
    if file.size and file.size > 5 * 1024 * 1024:
        raise HTTPException(
            status_code=400,
            detail="Image file too large. Maximum size is 5MB"
        )
    
    try:
        # Read file
        image_data = await file.read()
        file_size = len(image_data)
        
        # Validate parameters
        if max_tokens and (max_tokens < 10 or max_tokens > 500):
            max_tokens = 50
            
        if temperature and (temperature < 0.0 or temperature > 1.0):
            temperature = 0.1
        
        # Classify image
        classification_result = await image_service.classify_image(
            image_data=image_data,
            max_tokens=max_tokens or 50,
            temperature=temperature or 0.1
        )
        
        processing_time = time.time() - start_time
        
        return ImageClassificationResponse(
            success=True,
            classification=classification_result,
            filename=file.filename or "unknown",
            file_size=file_size,
            processing_time=round(processing_time, 2),
            timestamp=datetime.now()
        )
        
    except Exception as e:
        processing_time = time.time() - start_time
        
        # Log error (in production, use proper logging)
        print(f"Classification error: {str(e)}")
        
        raise HTTPException(
            status_code=500,
            detail=f"Failed to classify image: {str(e)}"
        )


@router.post(
    "/classify-batch",
    summary="Classify Multiple Images",
    description="Upload multiple images for batch classification"
)
async def classify_batch_images(
    files: list[UploadFile] = File(..., description="Multiple image files")
):
    """
    Classify multiple images in batch using Qwen 2.5 VL 32B model
    
    - **files**: List of image files (max 10 files)
    """
    
    if len(files) > 10:
        raise HTTPException(
            status_code=400,
            detail="Maximum 10 files allowed in batch processing"
        )
    
    results = []
    
    for file in files:
        try:
            start_time = time.time()
            
            # Validate file
            if not file.content_type or not file.content_type.startswith("image/"):
                results.append({
                    "filename": file.filename,
                    "success": False,
                    "error": "Not a valid image file"
                })
                continue
            
            # Read and classify
            image_data = await file.read()
            classification = await image_service.classify_image(image_data=image_data)
            
            processing_time = time.time() - start_time
            
            results.append({
                "filename": file.filename,
                "success": True,
                "classification": classification,
                "file_size": len(image_data),
                "processing_time": round(processing_time, 2)
            })
            
        except Exception as e:
            results.append({
                "filename": file.filename,
                "success": False,
                "error": str(e)
            })
    
    return {
        "batch_results": results,
        "total_files": len(files),
        "successful": len([r for r in results if r["success"]]),
        "failed": len([r for r in results if not r["success"]]),
        "timestamp": datetime.now()
    }