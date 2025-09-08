"""Authentication middleware for FastAPI."""

from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app.core.config import settings


class AuthMiddleware(BaseHTTPMiddleware):
    """Simple token-based authentication middleware."""
    
    def __init__(self, app, excluded_paths: list = None):
        super().__init__(app)
        # Paths that don't require authentication
        self.excluded_paths = excluded_paths or [
            "/docs",
            "/redoc", 
            "/openapi.json",
            "/health",
            "/"
        ]
    
    async def dispatch(self, request: Request, call_next):
        # Check if the path is excluded from authentication
        if request.url.path in self.excluded_paths:
            return await call_next(request)
        
        # Get token from query parameters
        token = request.query_params.get("token")
        
        if not token:
            return JSONResponse(
                status_code=401,
                content={
                    "success": False,
                    "error": "Authentication required",
                    "message": "Missing 'token' parameter in query string"
                }
            )
        
        # Validate token
        if token != settings.api_auth_token:
            return JSONResponse(
                status_code=401,
                content={
                    "success": False,
                    "error": "Authentication failed",
                    "message": "Invalid token"
                }
            )
        
        # Token is valid, proceed with the request
        return await call_next(request)
