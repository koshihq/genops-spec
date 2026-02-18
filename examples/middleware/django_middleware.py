#!/usr/bin/env python3
"""
🎸 Django Middleware for GenOps AI Attribution

Complete working Django middleware that automatically sets up
attribution context for all AI operations in your Django application.

Features:
✅ Django middleware integration with proper setup
✅ User/customer/session attribution from Django models
✅ Django REST Framework authentication support
✅ Session-based and token-based authentication
✅ Multi-tenant support with proper context isolation
✅ Integration with Django's built-in User model
✅ Custom user model support
✅ Request tracing and performance monitoring
"""

import time
import uuid
from typing import Any, Callable, Optional

from django.conf import settings
from django.core.exceptions import ObjectDoesNotExist
from django.http import HttpRequest, HttpResponse
from django.utils.deprecation import MiddlewareMixin

import genops

# Optional Django REST Framework integration
try:
    from rest_framework.authtoken.models import Token
    from rest_framework.request import Request as DRFRequest

    HAS_DRF = True
except ImportError:
    HAS_DRF = False
    Token = None
    DRFRequest = None


class GenOpsDjangoMiddleware(MiddlewareMixin):
    """
    Django middleware for automatic GenOps AI attribution context management.

    This middleware integrates with Django's authentication system,
    session management, and user models to provide comprehensive
    attribution context for all AI operations.

    Add to MIDDLEWARE in settings.py:
        MIDDLEWARE = [
            # ... other middleware
            'path.to.GenOpsDjangoMiddleware',
        ]
    """

    def __init__(self, get_response: Callable = None):
        super().__init__(get_response)
        self.get_response = get_response

        # Configuration from Django settings
        self.config = {
            "customer_header": getattr(
                settings, "GENOPS_CUSTOMER_HEADER", "HTTP_X_CUSTOMER_ID"
            ),
            "tenant_header": getattr(
                settings, "GENOPS_TENANT_HEADER", "HTTP_X_TENANT_ID"
            ),
            "trace_header": getattr(settings, "GENOPS_TRACE_HEADER", "HTTP_X_TRACE_ID"),
            "enable_session_tracking": getattr(
                settings, "GENOPS_ENABLE_SESSION_TRACKING", True
            ),
            "enable_performance_tracking": getattr(
                settings, "GENOPS_ENABLE_PERFORMANCE_TRACKING", True
            ),
            "debug": getattr(settings, "DEBUG", False),
            "user_customer_field": getattr(
                settings, "GENOPS_USER_CUSTOMER_FIELD", "customer_id"
            ),
            "user_tier_field": getattr(settings, "GENOPS_USER_TIER_FIELD", "tier"),
        }

        # Set up global defaults from Django settings
        defaults = getattr(settings, "GENOPS_DEFAULTS", {})
        defaults.setdefault("framework", "django")
        defaults.setdefault(
            "environment", getattr(settings, "ENVIRONMENT", "development")
        )
        defaults.setdefault("service", getattr(settings, "SERVICE_NAME", "django-app"))

        genops.set_default_attributes(**defaults)

    def __call__(self, request: HttpRequest) -> HttpResponse:
        """Process request with attribution context."""

        # Set up attribution context
        self.process_request(request)

        try:
            # Process the request
            response = self.get_response(request)

            # Add response metrics
            self.process_response(request, response)

            return response

        except Exception as e:
            # Add error context
            self.process_exception(request, e)
            raise

        finally:
            # Always clean up context
            genops.clear_context()

    def process_request(self, request: HttpRequest) -> None:
        """Set up attribution context at the beginning of request processing."""

        start_time = time.time()
        request._genops_start_time = start_time

        # Generate or extract request ID
        request_id = self._extract_request_id(request)

        # Extract attribution information
        user_info = self._extract_user_info(request)
        customer_info = self._extract_customer_info(request)
        session_info = self._extract_session_info(request)

        # Build attribution context
        context_attrs = {
            "request_id": request_id,
            "method": request.method,
            "path": request.path_info,
            "view_name": self._get_view_name(request),
            "user_agent": request.META.get("HTTP_USER_AGENT"),
            "client_ip": self._get_client_ip(request),
            "start_time": start_time,
        }

        # Add user information
        context_attrs.update(user_info)

        # Add customer information
        context_attrs.update(customer_info)

        # Add session information
        if self.config["enable_session_tracking"]:
            context_attrs.update(session_info)

        # Set the context
        genops.set_context(**context_attrs)

        if self.config["debug"]:
            print(f"GenOps Django context set: {context_attrs}")

    def process_response(
        self, request: HttpRequest, response: HttpResponse
    ) -> HttpResponse:
        """Add performance metrics to context."""

        if self.config["enable_performance_tracking"] and hasattr(
            request, "_genops_start_time"
        ):
            duration = time.time() - request._genops_start_time

            genops.set_context(
                request_duration_ms=round(duration * 1000, 2),
                response_status=response.status_code,
                response_size=len(response.content)
                if hasattr(response, "content")
                else None,
            )

        return response

    def process_exception(self, request: HttpRequest, exception: Exception) -> None:
        """Add exception information to context."""

        genops.set_context(
            error_type=type(exception).__name__,
            error_message=str(exception),
            error_occurred=True,
        )

    def _extract_request_id(self, request: HttpRequest) -> str:
        """Extract or generate request ID."""
        return (
            request.META.get(self.config["trace_header"])
            or request.META.get("HTTP_X_REQUEST_ID")
            or str(uuid.uuid4())
        )

    def _extract_user_info(self, request: HttpRequest) -> dict[str, Any]:
        """Extract user attribution information."""
        user_info = {}

        # Check if user is authenticated
        if hasattr(request, "user") and request.user.is_authenticated:
            user = request.user
            user_info["user_id"] = str(user.pk)

            # Add basic user information
            if hasattr(user, "email") and user.email:
                user_info["user_email"] = user.email

            if hasattr(user, "username") and user.username:
                user_info["username"] = user.username

            # Add custom user fields
            customer_field = self.config["user_customer_field"]
            if hasattr(user, customer_field):
                customer_value = getattr(user, customer_field)
                if customer_value:
                    user_info["user_customer_id"] = str(customer_value)

            tier_field = self.config["user_tier_field"]
            if hasattr(user, tier_field):
                tier_value = getattr(user, tier_field)
                if tier_value:
                    user_info["user_tier"] = str(tier_value)

            # Add staff/superuser status
            if user.is_staff:
                user_info["user_role"] = "staff"
            elif user.is_superuser:
                user_info["user_role"] = "superuser"
            else:
                user_info["user_role"] = "user"

            # Check for DRF token authentication
            if HAS_DRF:
                user_info.update(self._extract_drf_info(request))

        return user_info

    def _extract_customer_info(self, request: HttpRequest) -> dict[str, Any]:
        """Extract customer/tenant information."""
        customer_info = {}

        # Check headers first
        customer_id = request.META.get(self.config["customer_header"])
        if customer_id:
            customer_info["customer_id"] = customer_id

        tenant_id = request.META.get(self.config["tenant_header"])
        if tenant_id:
            customer_info["tenant_id"] = tenant_id

        # Try to get customer from user if not in headers
        if (
            "customer_id" not in customer_info
            and hasattr(request, "user")
            and request.user.is_authenticated
        ):
            customer_field = self.config["user_customer_field"]
            if hasattr(request.user, customer_field):
                customer_value = getattr(request.user, customer_field)
                if customer_value:
                    customer_info["customer_id"] = str(customer_value)

        return customer_info

    def _extract_session_info(self, request: HttpRequest) -> dict[str, Any]:
        """Extract session attribution information."""
        session_info = {}

        if hasattr(request, "session"):
            session_info["session_key"] = request.session.session_key

            # Add custom session data
            if "customer_id" in request.session:
                session_info["session_customer_id"] = request.session["customer_id"]

            if "tenant_id" in request.session:
                session_info["session_tenant_id"] = request.session["tenant_id"]

        return session_info

    def _extract_drf_info(self, request: HttpRequest) -> dict[str, Any]:
        """Extract Django REST Framework specific information."""
        drf_info = {}

        # Check if this is a DRF request
        if isinstance(request, DRFRequest):
            # Try to get token information
            auth_header = request.META.get("HTTP_AUTHORIZATION", "")
            if auth_header.startswith("Token "):
                token_key = auth_header.split(" ")[1]
                try:
                    Token.objects.select_related("user").get(key=token_key)
                    drf_info["auth_method"] = "token"
                    drf_info["token_key"] = (
                        token_key[:8] + "..."
                    )  # Partial for security
                except ObjectDoesNotExist:
                    pass

        return drf_info

    def _get_view_name(self, request: HttpRequest) -> Optional[str]:
        """Get the view name for the current request."""
        try:
            if hasattr(request, "resolver_match") and request.resolver_match:
                return request.resolver_match.view_name
        except Exception:
            pass

        return None

    def _get_client_ip(self, request: HttpRequest) -> Optional[str]:
        """Extract client IP address from request."""

        # Check for forwarded headers first
        forwarded_for = request.META.get("HTTP_X_FORWARDED_FOR")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()

        real_ip = request.META.get("HTTP_X_REAL_IP")
        if real_ip:
            return real_ip

        # Fallback to REMOTE_ADDR
        return request.META.get("REMOTE_ADDR")


# Django management command for GenOps setup
class Command:
    """
    Django management command to set up GenOps attribution.

    Save as: management/commands/setup_genops.py
    Run with: python manage.py setup_genops
    """

    help = "Set up GenOps AI attribution for this Django project"

    def add_arguments(self, parser):
        parser.add_argument(
            "--team", type=str, help="Default team name for attribution"
        )
        parser.add_argument(
            "--project", type=str, help="Default project name for attribution"
        )
        parser.add_argument(
            "--environment",
            type=str,
            default="development",
            help="Environment name (default: development)",
        )

    def handle(self, *args, **options):
        """Handle the management command."""

        # Set up defaults from command arguments
        defaults = {"framework": "django", "environment": options["environment"]}

        if options["team"]:
            defaults["team"] = options["team"]

        if options["project"]:
            defaults["project"] = options["project"]

        genops.set_default_attributes(**defaults)

        self.stdout.write(
            self.style.SUCCESS(
                f"GenOps attribution configured with defaults: {defaults}"
            )
        )


# Example Django views using GenOps attribution
import json  # noqa: E402

from django.http import JsonResponse  # noqa: E402
from django.views.decorators.csrf import csrf_exempt  # noqa: E402
from django.views.decorators.http import require_http_methods  # noqa: E402


def attribution_view(request):
    """View showing current attribution context."""

    return JsonResponse(
        {
            "message": "Django + GenOps AI Attribution",
            "defaults": genops.get_default_attributes(),
            "context": genops.get_context(),
            "effective": genops.get_effective_attributes(),
        }
    )


def ai_operation_view(request):
    """View performing an AI operation with attribution."""

    # Add operation-specific context
    operation_context = {
        "operation_name": "django_ai_operation",
        "operation_type": "ai.inference",
        "feature": request.GET.get("feature", "general"),
    }

    # Get effective attributes including operation context
    effective_attrs = genops.get_effective_attributes(**operation_context)

    # Simulate AI processing
    result = {
        "message": "AI operation completed",
        "attribution": effective_attrs,
        "processing_time": "120ms",
        "model": "django-example-model",
    }

    return JsonResponse(result)


@csrf_exempt
@require_http_methods(["POST"])
def set_session_attribution(request):
    """Set attribution information in user session."""

    try:
        data = json.loads(request.body)

        # Set session data
        if "customer_id" in data:
            request.session["customer_id"] = data["customer_id"]

        if "tenant_id" in data:
            request.session["tenant_id"] = data["tenant_id"]

        return JsonResponse(
            {
                "message": "Session attribution updated",
                "session_data": {
                    "customer_id": request.session.get("customer_id"),
                    "tenant_id": request.session.get("tenant_id"),
                    "session_key": request.session.session_key,
                },
            }
        )

    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)


# Example URL configuration
"""
# urls.py

from django.urls import path
from . import views

urlpatterns = [
    path('attribution/', views.attribution_view, name='attribution'),
    path('ai-operation/', views.ai_operation_view, name='ai_operation'),
    path('set-session/', views.set_session_attribution, name='set_session'),
]
"""

# Example settings.py configuration
"""
# settings.py

# GenOps AI Configuration
GENOPS_DEFAULTS = {
    'team': 'backend-engineering',
    'project': 'django-ai-app',
    'service': 'django-example'
}

GENOPS_CUSTOMER_HEADER = 'HTTP_X_CUSTOMER_ID'
GENOPS_TENANT_HEADER = 'HTTP_X_TENANT_ID'
GENOPS_ENABLE_SESSION_TRACKING = True
GENOPS_ENABLE_PERFORMANCE_TRACKING = True
GENOPS_USER_CUSTOMER_FIELD = 'customer_id'  # Field on User model
GENOPS_USER_TIER_FIELD = 'tier'             # Field on User model

# Add middleware
MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',

    # Add GenOps middleware
    'path.to.GenOpsDjangoMiddleware',
]
"""

if __name__ == "__main__":
    print("🎸 Django + GenOps AI Attribution Middleware")
    print("=" * 50)
    print()
    print("This middleware provides automatic attribution context for Django apps.")
    print()
    print("Setup Instructions:")
    print("1. Add GenOpsDjangoMiddleware to MIDDLEWARE in settings.py")
    print("2. Configure GENOPS_DEFAULTS in settings.py")
    print("3. Optionally add custom user model fields for customer/tier")
    print("4. Use genops.get_effective_attributes() in views for AI operations")
    print()
    print("Example requests:")
    print("  GET /attribution/                 - Show attribution context")
    print("  GET /ai-operation/?feature=chat   - AI operation with attribution")
    print("  POST /set-session/                - Set session attribution")
    print()
    print("Example headers:")
    print("  X-Customer-ID: enterprise-123")
    print("  X-Tenant-ID: tenant-456")
    print("  X-Trace-ID: trace-789")
