import sys
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager

from app.config.config import settings
from app.services.image_detection.image_detection_router import router as image_router

# Setup logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    description="AI-powered image classification API using OpenRouter vision models",
    version=settings.APP_VERSION,
    debug=settings.DEBUG,
    lifespan=None,  # will assign below
    docs_url="/docs" if settings.DEBUG else None,
    redoc_url="/redoc" if settings.DEBUG else None,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=settings.CORS_ALLOW_CREDENTIALS,
    allow_methods=settings.CORS_ALLOW_METHODS,
    allow_headers=settings.CORS_ALLOW_HEADERS,
)

# Include routers
app.include_router(image_router)

# Root endpoint
@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information"""
    return {
        "message": f"Welcome to {settings.APP_NAME}",
        "version": settings.APP_VERSION,
        "status": "running",
        "docs_url": "/docs" if settings.DEBUG else "disabled",
        "endpoints": {
            "health": "/api/v1/image-detection/health",
            "classify": "/api/v1/image-detection/classify",
            "classify_batch": "/api/v1/image-detection/classify-batch"
        }
    }

# Global error handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Global exception: {str(exc)}")
    
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Internal server error",
            "detail": str(exc) if settings.DEBUG else "Something went wrong"
        }
    )

# Health check endpoint
@app.get("/health", tags=["Health"])
async def health():
    """Simple health check"""
    return {
        "status": "healthy",
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION
    }

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events"""
    # Startup
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    logger.info(f"Default model: {settings.DEFAULT_MODEL}")
    logger.info(f"Debug mode: {settings.DEBUG}")
    logger.info(f"Application running on {settings.HOST}:{settings.PORT}")
    
    # Test API connection on startup
    from app.services.image_detection.image_detection import ImageDetectionService
    service = ImageDetectionService()
    
    try:
        connection_ok = await service.test_connection()
        if connection_ok:
            logger.info("✅ OpenRouter API connection successful")
        else:
            logger.warning("⚠️  OpenRouter API connection failed - check your API key")
    except Exception as e:
        logger.error(f"❌ Failed to test API connection: {str(e)}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down application")

app.router.lifespan_context = lifespan

if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting server on {settings.HOST}:{settings.PORT}")
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.RELOAD and settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower(),
        access_log=settings.DEBUG
    )
