import base64
import httpx
import asyncio
from typing import Optional, Dict, Any
import logging

from app.config.config import settings

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImageDetectionService:
    """Service for image classification using OpenRouter API"""
    
    def __init__(self):
        self.api_url = settings.OPENROUTER_URL
        self.api_key = settings.OPENROUTER_API_KEY
        self.timeout = settings.REQUEST_TIMEOUT
        
        if not self.api_key:
            logger.warning("OpenRouter API key not found in environment variables")
    
    async def test_connection(self) -> bool:
        """Test connection to OpenRouter API"""
        try:
            headers = self._get_headers()
            async with httpx.AsyncClient(timeout=10.0) as client:
                # Try to make a simple request to check if API is accessible
                response = await client.get(
                    "https://openrouter.ai/api/v1/models",
                    headers=headers
                )
                return response.status_code == 200
        except Exception as e:
            logger.error(f"Connection test failed: {str(e)}")
            return False
    
    def _get_headers(self) -> Dict[str, str]:
        """Get request headers for OpenRouter API"""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": settings.APP_NAME,
            "X-Title": settings.APP_NAME
        }
    
    def _prepare_payload(
        self, 
        image_base64: str,
        max_tokens: int = 50,
        temperature: float = 0.1
    ) -> Dict[str, Any]:
        """Prepare API payload for image classification"""
        return {
            "model": settings.DEFAULT_MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "What is this image? Give me a brief, clear identification "
                                "of the main object or subject in this image. "
                                "Just tell me what it is in 1-3 words if possible. "
                                "Be specific and accurate."
                            )
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False
        }
    
    async def classify_image(
        self,
        image_data: bytes,
        max_tokens: int = 50,
        temperature: float = 0.1,
        retry_count: int = 3
    ) -> str:
        """
        Classify image using Qwen 2.5 VL 32B model
        
        Args:
            image_data: Raw image bytes
            max_tokens: Maximum tokens for response
            temperature: Model temperature
            retry_count: Number of retries on failure
            
        Returns:
            Classification result string
            
        Raises:
            Exception: If classification fails after all retries
        """
        
        if not self.api_key:
            raise Exception("OpenRouter API key not configured")
        
        # Encode image to base64
        image_base64 = base64.b64encode(image_data).decode("utf-8")
        
        # Prepare request
        headers = self._get_headers()
        payload = self._prepare_payload(
            image_base64=image_base64,
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        # Try classification with retries
        last_exception = None
        
        for attempt in range(retry_count):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    logger.info(f"Attempting classification with Qwen model (attempt {attempt + 1})")
                    
                    response = await client.post(
                        self.api_url,
                        json=payload,
                        headers=headers
                    )
                    
                    # Check response status
                    if response.status_code != 200:
                        error_msg = f"API error {response.status_code}: {response.text}"
                        logger.error(error_msg)
                        
                        # If it's a client error (4xx), don't retry
                        if 400 <= response.status_code < 500:
                            raise Exception(error_msg)
                        
                        # For server errors (5xx), continue to retry
                        last_exception = Exception(error_msg)
                        continue
                    
                    # Parse response
                    result = response.json()
                    
                    # Extract classification from response
                    if "choices" not in result or not result["choices"]:
                        raise Exception("Invalid response format from API")
                    
                    classification = result["choices"][0]["message"]["content"].strip()
                    
                    if not classification:
                        raise Exception("Empty classification result")
                    
                    logger.info(f"Classification successful: {classification}")
                    return classification
            
            except httpx.TimeoutException:
                last_exception = Exception(f"Request timeout after {self.timeout}s")
                logger.warning(f"Timeout on attempt {attempt + 1}")
            
            except httpx.RequestError as e:
                last_exception = Exception(f"Request error: {str(e)}")
                logger.warning(f"Request error on attempt {attempt + 1}: {str(e)}")
            
            except Exception as e:
                last_exception = e
                logger.error(f"Classification error on attempt {attempt + 1}: {str(e)}")
                # If it's a client error, don't retry
                if "400" in str(e) or "401" in str(e) or "403" in str(e):
                    break
            
            # Wait before retry (exponential backoff)
            if attempt < retry_count - 1:
                wait_time = 2 ** attempt
                logger.info(f"Waiting {wait_time}s before retry...")
                await asyncio.sleep(wait_time)
        # If all retries failed, raise the last exception
        raise last_exception or Exception("Classification failed after all retries")
    
    async def classify_batch(
        self,
        images_data: list[bytes],
        max_concurrent: int = 3
    ) -> list[Dict[str, Any]]:
        """
        Classify multiple images concurrently using Qwen 2.5 VL 32B model
        
        Args:
            images_data: List of raw image bytes
            max_concurrent: Maximum concurrent requests
            
        Returns:
            List of classification results
        """
        
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def classify_single(image_data: bytes, index: int) -> Dict[str, Any]:
            async with semaphore:
                try:
                    result = await self.classify_image(image_data=image_data)
                    return {
                        "index": index,
                        "success": True,
                        "classification": result
                    }
                except Exception as e:
                    return {
                        "index": index,
                        "success": False,
                        "error": str(e)
                    }
        
        # Run all classifications concurrently
        tasks = [
            classify_single(image_data, i)
            for i, image_data in enumerate(images_data)
        ]
        
        results = await asyncio.gather(*tasks)
        return results
