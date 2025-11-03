#!/usr/bin/env python3
"""
🚀 FastAPI Middleware for GenOps AI Attribution

Complete working FastAPI middleware that automatically sets up
attribution context for all AI operations in your FastAPI application.

Features:
✅ Async/await support with proper context management
✅ Automatic user/customer/request attribution
✅ JWT token integration and dependency injection
✅ Request tracing and performance monitoring
✅ Custom header support for multi-tenant apps
✅ OpenAPI documentation integration
✅ Error handling and fallback behavior
"""

import time
import uuid
from typing import Any, Callable, Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

import genops

# Optional JWT integration
try:
    import jwt as pyjwt
    HAS_JWT = True
except ImportError:
    HAS_JWT = False


class GenOpsFastAPIMiddleware:
    """
    FastAPI middleware for automatic GenOps AI attribution context management.

    This middleware provides async-compatible context management that works
    with FastAPI's dependency injection system and async request handling.
    """

    def __init__(
        self,
        app: FastAPI,
        customer_header: str = "x-customer-id",
        user_header: str = "x-user-id",
        tenant_header: str = "x-tenant-id",
        trace_header: str = "x-trace-id",
        environment: str = "production",
        enable_performance_tracking: bool = True,
        jwt_secret: Optional[str] = None,
        debug: bool = False,
        **app_defaults
    ):
        self.app = app
        self.customer_header = customer_header
        self.user_header = user_header
        self.tenant_header = tenant_header
        self.trace_header = trace_header
        self.environment = environment
        self.enable_performance_tracking = enable_performance_tracking
        self.jwt_secret = jwt_secret
        self.debug = debug

        # Set up global defaults
        defaults = {
            'service': app.title.lower().replace(' ', '-'),
            'environment': environment,
            'framework': 'fastapi',
            **app_defaults
        }
        genops.set_default_attributes(**defaults)

        # Register middleware
        self._register_middleware()

    def _register_middleware(self):
        """Register the middleware with FastAPI."""

        @self.app.middleware("http")
        async def genops_attribution_middleware(request: Request, call_next: Callable):
            """Main middleware function for attribution context management."""
            start_time = time.time()

            # Set up attribution context
            await self._set_request_context(request, start_time)

            try:
                # Process the request
                response = await call_next(request)

                # Add performance tracking
                if self.enable_performance_tracking:
                    await self._add_performance_metrics(start_time, response)

                return response

            except Exception as e:
                # Add error context
                genops.set_context(
                    error_type=type(e).__name__,
                    error_message=str(e),
                    error_occurred=True
                )
                raise

            finally:
                # Always clear context
                genops.clear_context()

                if self.debug:
                    print("GenOps context cleared for request")

    async def _set_request_context(self, request: Request, start_time: float):
        """Set up attribution context for the current request."""

        # Generate or extract request ID
        request_id = (
            request.headers.get(self.trace_header) or
            request.headers.get("x-request-id") or
            str(uuid.uuid4())
        )

        # Extract attribution information
        user_id = await self._extract_user_id(request)
        customer_id = self._extract_customer_id(request)
        user_info = await self._extract_user_info(request)

        # Build context
        context_attrs = {
            'request_id': request_id,
            'method': request.method,
            'path': request.url.path,
            'user_agent': request.headers.get('user-agent'),
            'client_ip': self._get_client_ip(request),
            'start_time': start_time,
        }

        # Add user information
        if user_id:
            context_attrs['user_id'] = user_id
        if user_info:
            context_attrs.update(user_info)

        # Add customer/tenant information
        if customer_id:
            context_attrs['customer_id'] = customer_id

        tenant_id = request.headers.get(self.tenant_header)
        if tenant_id:
            context_attrs['tenant_id'] = tenant_id

        # Set the context
        genops.set_context(**context_attrs)

        if self.debug:
            print(f"GenOps context set: {context_attrs}")

    async def _extract_user_id(self, request: Request) -> Optional[str]:
        """Extract user ID from headers or JWT token."""

        # Try explicit header first
        user_id = request.headers.get(self.user_header)
        if user_id:
            return user_id

        # Try JWT token
        if HAS_JWT and self.jwt_secret:
            try:
                auth_header = request.headers.get('authorization')
                if auth_header and auth_header.startswith('Bearer '):
                    token = auth_header.split(' ')[1]
                    payload = pyjwt.decode(token, self.jwt_secret, algorithms=['HS256'])
                    return payload.get('sub') or payload.get('user_id')
            except Exception:
                pass

        return None

    def _extract_customer_id(self, request: Request) -> Optional[str]:
        """Extract customer ID from headers."""
        return request.headers.get(self.customer_header)

    async def _extract_user_info(self, request: Request) -> dict[str, Any]:
        """Extract additional user information from JWT or headers."""
        user_info = {}

        # Try JWT token for additional claims
        if HAS_JWT and self.jwt_secret:
            try:
                auth_header = request.headers.get('authorization')
                if auth_header and auth_header.startswith('Bearer '):
                    token = auth_header.split(' ')[1]
                    payload = pyjwt.decode(token, self.jwt_secret, algorithms=['HS256'])

                    # Extract common claims
                    if 'role' in payload:
                        user_info['user_role'] = payload['role']
                    if 'tier' in payload:
                        user_info['user_tier'] = payload['tier']
                    if 'customer_id' in payload:
                        user_info['jwt_customer_id'] = payload['customer_id']
                    if 'email' in payload:
                        user_info['user_email'] = payload['email']

            except Exception:
                pass

        return user_info

    def _get_client_ip(self, request: Request) -> Optional[str]:
        """Extract client IP address."""
        # Check for forwarded headers first
        forwarded_for = request.headers.get('x-forwarded-for')
        if forwarded_for:
            return forwarded_for.split(',')[0].strip()

        real_ip = request.headers.get('x-real-ip')
        if real_ip:
            return real_ip

        # Fallback to client host
        if request.client:
            return request.client.host

        return None

    async def _add_performance_metrics(self, start_time: float, response: Response):
        """Add performance metrics to context."""
        duration = time.time() - start_time

        genops.set_context(
            request_duration_ms=round(duration * 1000, 2),
            response_status=response.status_code,
        )


# Dependency injection functions
async def get_attribution_context() -> dict[str, Any]:
    """Dependency to get current attribution context."""
    return genops.get_context()


async def get_effective_attributes() -> dict[str, Any]:
    """Dependency to get effective attributes for the current operation."""
    return genops.get_effective_attributes()


async def require_customer_id(customer_id: str = Header(..., alias="x-customer-id")) -> str:
    """Dependency to require customer ID header."""
    return customer_id


async def require_user_id(user_id: str = Header(..., alias="x-user-id")) -> str:
    """Dependency to require user ID header."""
    return user_id


class JWTBearer(HTTPBearer):
    """JWT Bearer token authentication with attribution context."""

    def __init__(self, jwt_secret: str, auto_error: bool = True):
        super().__init__(auto_error=auto_error)
        self.jwt_secret = jwt_secret

    async def __call__(self, request: Request) -> dict[str, Any]:
        credentials: HTTPAuthorizationCredentials = await super().__call__(request)

        if not HAS_JWT:
            raise HTTPException(
                status_code=500,
                detail="JWT support not available. Install with: pip install PyJWT"
            )

        try:
            payload = pyjwt.decode(
                credentials.credentials,
                self.jwt_secret,
                algorithms=['HS256']
            )

            # Add JWT claims to attribution context
            jwt_context = {}
            if 'sub' in payload:
                jwt_context['jwt_user_id'] = payload['sub']
            if 'role' in payload:
                jwt_context['jwt_role'] = payload['role']
            if 'customer_id' in payload:
                jwt_context['jwt_customer_id'] = payload['customer_id']

            genops.set_context(**jwt_context)

            return payload

        except pyjwt.InvalidTokenError as e:
            raise HTTPException(
                status_code=401,
                detail=f"Invalid token: {str(e)}"
            )


# Response models
class AttributionResponse(BaseModel):
    """Response model for attribution context."""
    defaults: dict[str, Any]
    context: dict[str, Any]
    effective: dict[str, Any]


class AIOperationRequest(BaseModel):
    """Request model for AI operations."""
    operation_name: str
    input_text: str
    feature: Optional[str] = None
    priority: Optional[str] = "normal"


class AIOperationResponse(BaseModel):
    """Response model for AI operations."""
    result: str
    attribution: dict[str, Any]
    operation_metadata: dict[str, Any]


def create_example_app():
    """Create an example FastAPI app with GenOps middleware."""

    app = FastAPI(
        title="GenOps AI FastAPI Example",
        description="FastAPI application with automatic GenOps AI attribution",
        version="1.0.0"
    )

    # Initialize GenOps middleware
    GenOpsFastAPIMiddleware(
        app,
        environment="development",
        team="backend-engineering",
        project="ai-api-fastapi",
        debug=True,
        jwt_secret="demo-secret-key"  # In production, use secure secret
    )

    # Set up JWT authentication (optional)
    jwt_bearer = JWTBearer(jwt_secret="demo-secret-key")

    @app.get("/")
    async def root():
        """Basic endpoint showing attribution context."""
        context = genops.get_context()
        return {
            "message": "FastAPI + GenOps AI Attribution",
            "attribution_context": context
        }

    @app.get("/attribution", response_model=AttributionResponse)
    async def get_attribution():
        """Get complete attribution information."""
        return AttributionResponse(
            defaults=genops.get_default_attributes(),
            context=genops.get_context(),
            effective=genops.get_effective_attributes()
        )

    @app.get("/protected")
    async def protected_endpoint(
        customer_id: str = Depends(require_customer_id),
        context: dict[str, Any] = Depends(get_attribution_context)
    ):
        """Protected endpoint requiring customer ID header."""
        return {
            "message": "Protected endpoint accessed",
            "customer_id": customer_id,
            "context": context
        }

    @app.post("/ai-operation", response_model=AIOperationResponse)
    async def ai_operation(
        request_data: AIOperationRequest,
        effective_attrs: dict[str, Any] = Depends(get_effective_attributes)
    ):
        """AI operation endpoint with full attribution."""

        # Add operation-specific context
        operation_context = {
            'operation_name': request_data.operation_name,
            'operation_type': 'ai.inference',
            'feature': request_data.feature or 'general',
            'priority': request_data.priority,
            'input_length': len(request_data.input_text)
        }

        # Merge with effective attributes
        final_attrs = {**effective_attrs, **operation_context}

        # Simulate AI processing
        result = f"Processed: {request_data.input_text[:50]}..."

        return AIOperationResponse(
            result=result,
            attribution=final_attrs,
            operation_metadata={
                'processing_time': '45ms',
                'model': 'example-model',
                'tokens_used': 150
            }
        )

    @app.post("/login")
    async def login(user_id: str, customer_id: Optional[str] = None):
        """Generate a demo JWT token with attribution claims."""
        if not HAS_JWT:
            raise HTTPException(
                status_code=500,
                detail="JWT support not available. Install with: pip install PyJWT"
            )

        payload = {
            'sub': user_id,
            'role': 'user',
            'tier': 'premium',
            'exp': int(time.time()) + 3600,  # 1 hour expiry
        }

        if customer_id:
            payload['customer_id'] = customer_id

        token = pyjwt.encode(payload, "demo-secret-key", algorithm='HS256')

        return {
            'access_token': token,
            'token_type': 'bearer',
            'expires_in': 3600
        }

    @app.get("/protected-jwt")
    async def protected_jwt_endpoint(
        jwt_payload: dict[str, Any] = Depends(jwt_bearer),
        context: dict[str, Any] = Depends(get_attribution_context)
    ):
        """JWT protected endpoint with automatic attribution."""
        return {
            "message": "JWT protected endpoint accessed",
            "jwt_payload": jwt_payload,
            "attribution_context": context
        }

    @app.get("/health")
    async def health_check():
        """Health check endpoint (no attribution needed)."""
        return {"status": "healthy", "service": "fastapi-genops-example"}

    return app


if __name__ == "__main__":
    import uvicorn

    app = create_example_app()

    print("🚀 FastAPI + GenOps AI Attribution Example")
    print("=" * 50)
    print("Endpoints:")
    print("  GET  /                    - Basic attribution demo")
    print("  GET  /attribution         - Full attribution info")
    print("  GET  /protected           - Requires X-Customer-ID header")
    print("  POST /ai-operation        - AI operation with attribution")
    print("  POST /login               - Generate JWT token")
    print("  GET  /protected-jwt       - JWT protected endpoint")
    print("  GET  /health              - Health check")
    print("  GET  /docs                - OpenAPI documentation")
    print()
    print("Try these requests:")
    print("  curl http://localhost:8000/")
    print("  curl -H 'X-Customer-ID: enterprise-123' http://localhost:8000/protected")
    print("  curl -X POST http://localhost:8000/login?user_id=demo_user&customer_id=demo_customer")
    print()

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
