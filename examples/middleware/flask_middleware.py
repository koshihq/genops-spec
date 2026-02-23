#!/usr/bin/env python3
"""
🌟 Flask Middleware for GenOps AI Attribution

Complete working Flask middleware that automatically sets up
attribution context for all AI operations in your Flask application.

Features:
✅ Automatic user/customer/request attribution
✅ Error handling and fallback behavior
✅ Request tracing and session tracking
✅ Custom header support for multi-tenant apps
✅ Integration with Flask-Login and Flask-JWT-Extended
✅ Performance monitoring and debugging
"""

import os
import time
import uuid
from functools import wraps
from typing import Any, Optional

from flask import Flask, current_app, g, jsonify, request, session

import genops

# Optional integrations
try:
    from flask_login import current_user

    HAS_FLASK_LOGIN = True
except ImportError:
    HAS_FLASK_LOGIN = False
    current_user = None

try:
    from flask_jwt_extended import get_jwt, get_jwt_identity

    HAS_JWT_EXTENDED = True
except ImportError:
    HAS_JWT_EXTENDED = False


class GenOpsFlaskMiddleware:
    """
    Flask middleware for automatic GenOps AI attribution context management.

    This middleware automatically sets up attribution context at the beginning
    of each request and cleans it up at the end, ensuring all AI operations
    within the request are properly attributed.
    """

    def __init__(self, app: Optional[Flask] = None, **config):
        self.app = app
        self.config = {
            "customer_header": "X-Customer-ID",
            "user_header": "X-User-ID",
            "tenant_header": "X-Tenant-ID",
            "trace_header": "X-Trace-ID",
            "environment": "production",
            "enable_session_tracking": True,
            "enable_performance_tracking": True,
            "fallback_customer_id": "unknown",
            "debug": False,
            **config,
        }

        if app:
            self.init_app(app)

    def init_app(self, app: Flask):
        """Initialize the middleware with a Flask app."""
        self.app = app

        # Set up global defaults for the application
        app_defaults = {
            "service": app.name,
            "environment": self.config["environment"],
            "framework": "flask",
        }

        # Add any app-specific defaults from config
        if hasattr(app, "config") and "GENOPS_DEFAULTS" in app.config:
            app_defaults.update(app.config["GENOPS_DEFAULTS"])

        genops.set_default_attributes(**app_defaults)

        # Register middleware hooks
        app.before_request(self._before_request)
        app.after_request(self._after_request)
        app.teardown_appcontext(self._teardown_appcontext)

    def _before_request(self):
        """Set up attribution context at the start of each request."""
        start_time = time.time()

        # Generate or extract request ID
        request_id = (
            request.headers.get(self.config["trace_header"])
            or request.headers.get("X-Request-ID")
            or str(uuid.uuid4())
        )

        # Extract user information
        user_id = self._extract_user_id()
        user_info = self._extract_user_info()

        # Extract customer/tenant information
        customer_id = self._extract_customer_id()
        tenant_id = request.headers.get(self.config["tenant_header"])

        # Build attribution context
        context_attrs = {
            "request_id": request_id,
            "endpoint": request.endpoint,
            "method": request.method,
            "path": request.path,
            "user_agent": request.user_agent.string if request.user_agent else None,
            "remote_addr": request.remote_addr,
        }

        # Add user information
        if user_id:
            context_attrs["user_id"] = user_id
        if user_info:
            context_attrs.update(user_info)

        # Add customer/tenant information
        if customer_id:
            context_attrs["customer_id"] = customer_id
        if tenant_id:
            context_attrs["tenant_id"] = tenant_id

        # Add session information if enabled
        if self.config["enable_session_tracking"] and session:
            if "session_id" in session:
                context_attrs["session_id"] = session["session_id"]
            else:
                session_id = str(uuid.uuid4())
                session["session_id"] = session_id
                context_attrs["session_id"] = session_id

        # Store request start time for performance tracking
        if self.config["enable_performance_tracking"]:
            g.genops_start_time = start_time

        # Set the context
        genops.set_context(**context_attrs)

        # Debug logging
        if self.config["debug"]:
            current_app.logger.debug(
                f"GenOps context set for {request_id}: {context_attrs}"
            )

    def _after_request(self, response):
        """Handle response and record performance metrics."""
        if self.config["enable_performance_tracking"] and hasattr(
            g, "genops_start_time"
        ):
            request_duration = time.time() - g.genops_start_time

            # Add performance context
            genops.set_context(
                request_duration_ms=round(request_duration * 1000, 2),
                response_status=response.status_code,
                response_size=response.content_length,
            )

        return response

    def _teardown_appcontext(self, error=None):
        """Clean up attribution context at the end of request."""
        if error:
            # Add error information to context before clearing
            genops.set_context(
                error_type=type(error).__name__, error_message=str(error)
            )

        # Clear the context
        genops.clear_context()

        if self.config["debug"]:
            current_app.logger.debug("GenOps context cleared")

    def _extract_user_id(self) -> Optional[str]:
        """Extract user ID from various sources."""
        # Try explicit header first
        user_id = request.headers.get(self.config["user_header"])
        if user_id:
            return user_id

        # Try Flask-Login
        if HAS_FLASK_LOGIN and current_user and hasattr(current_user, "id"):
            return str(current_user.id)

        # Try JWT
        if HAS_JWT_EXTENDED:
            try:
                jwt_user = get_jwt_identity()
                if jwt_user:
                    return str(jwt_user)
            except Exception:
                pass

        # Try session
        if session and "user_id" in session:
            return str(session["user_id"])

        return None

    def _extract_user_info(self) -> dict[str, Any]:
        """Extract additional user information."""
        user_info = {}

        # From Flask-Login
        if (
            HAS_FLASK_LOGIN
            and current_user
            and hasattr(current_user, "is_authenticated")
        ):
            if current_user.is_authenticated:
                if hasattr(current_user, "email"):
                    user_info["user_email"] = current_user.email
                if hasattr(current_user, "role"):
                    user_info["user_role"] = current_user.role
                if hasattr(current_user, "tier"):
                    user_info["user_tier"] = current_user.tier

        # From JWT claims
        if HAS_JWT_EXTENDED:
            try:
                jwt_claims = get_jwt()
                if jwt_claims:
                    if "role" in jwt_claims:
                        user_info["user_role"] = jwt_claims["role"]
                    if "tier" in jwt_claims:
                        user_info["user_tier"] = jwt_claims["tier"]
                    if "customer_id" in jwt_claims:
                        user_info["jwt_customer_id"] = jwt_claims["customer_id"]
            except Exception:
                pass

        return user_info

    def _extract_customer_id(self) -> Optional[str]:
        """Extract customer ID from various sources."""
        # Try explicit header first
        customer_id = request.headers.get(self.config["customer_header"])
        if customer_id:
            return customer_id

        # Try JWT claims
        if HAS_JWT_EXTENDED:
            try:
                jwt_claims = get_jwt()
                if jwt_claims and "customer_id" in jwt_claims:
                    return str(jwt_claims["customer_id"])
            except Exception:
                pass

        # Try user object
        if HAS_FLASK_LOGIN and current_user and hasattr(current_user, "customer_id"):
            return str(current_user.customer_id)

        # Try session
        if session and "customer_id" in session:
            return str(session["customer_id"])

        # Fallback
        return (
            self.config["fallback_customer_id"]
            if self.config["fallback_customer_id"] != "unknown"
            else None
        )


def require_attribution(**required_attrs):
    """
    Decorator to ensure specific attribution attributes are set.

    Usage:
        @require_attribution(customer_id=True, user_id=True)
        def protected_endpoint():
            # This endpoint requires customer_id and user_id
            pass
    """

    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            context = genops.get_context()

            missing_attrs = []
            for attr, required in required_attrs.items():
                if required and attr not in context:
                    missing_attrs.append(attr)

            if missing_attrs:
                return jsonify(
                    {
                        "error": "Missing required attribution",
                        "missing_attributes": missing_attrs,
                    }
                ), 400

            return f(*args, **kwargs)

        return decorated_function

    return decorator


def with_ai_operation(operation_name: str, **operation_attrs):
    """
    Decorator to add operation-specific attribution to AI operations.

    Usage:
        @with_ai_operation('document_processing', feature='pdf_analysis')
        def process_document():
            # AI operations in this function get operation_name and feature attributes
            pass
    """

    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            # Add operation-specific context
            operation_context = {"operation_name": operation_name, **operation_attrs}

            # Get current context and merge
            current_context = genops.get_context()
            merged_context = {**current_context, **operation_context}

            # Temporarily set merged context
            genops.set_context(**merged_context)

            try:
                return f(*args, **kwargs)
            finally:
                # Restore original context
                genops.set_context(**current_context)

        return decorated_function

    return decorator


# Example Flask application with GenOps middleware
def create_example_app():
    """Create an example Flask app with GenOps middleware."""

    app = Flask(__name__)
    app.secret_key = "demo-secret-key"

    # Configure GenOps defaults
    app.config["GENOPS_DEFAULTS"] = {
        "team": "backend-engineering",
        "project": "ai-api",
        "service": "flask-example",
    }

    # Initialize GenOps middleware
    GenOpsFlaskMiddleware(
        app, environment="development", debug=True, enable_performance_tracking=True
    )

    @app.route("/")
    def index():
        """Basic endpoint showing attribution context."""
        context = genops.get_context()
        return jsonify(
            {"message": "Flask + GenOps AI Attribution", "attribution_context": context}
        )

    @app.route("/protected")
    @require_attribution(customer_id=True)
    def protected():
        """Protected endpoint requiring customer attribution."""
        return jsonify({"message": "Protected endpoint accessed"})

    @app.route("/ai-operation")
    @with_ai_operation("customer_support", feature="chat_response")
    def ai_operation():
        """Endpoint with AI operation attribution."""
        # Simulate AI operation with attribution
        effective_attrs = genops.get_effective_attributes()

        return jsonify(
            {
                "message": "AI operation completed",
                "effective_attribution": effective_attrs,
            }
        )

    @app.route("/login", methods=["POST"])
    def login():
        """Example login endpoint that sets session attribution."""
        data = request.get_json() or {}
        user_id = data.get("user_id", "demo_user")
        customer_id = data.get("customer_id", "demo_customer")

        # Set session information
        session["user_id"] = user_id
        session["customer_id"] = customer_id

        return jsonify(
            {"message": "Logged in", "user_id": user_id, "customer_id": customer_id}
        )

    @app.route("/context")
    def show_context():
        """Show current attribution context."""
        return jsonify(
            {
                "defaults": genops.get_default_attributes(),
                "context": genops.get_context(),
                "effective": genops.get_effective_attributes(),
            }
        )

    return app


if __name__ == "__main__":
    # Create and run the example app
    app = create_example_app()

    print("🌟 Flask + GenOps AI Attribution Example")
    print("=" * 50)
    print("Endpoints:")
    print("  GET  /                 - Basic attribution demo")
    print("  GET  /protected        - Requires customer_id header")
    print("  GET  /ai-operation     - AI operation with attribution")
    print("  POST /login            - Set session attribution")
    print("  GET  /context          - Show current context")
    print()
    print("Try these requests:")
    print("  curl http://localhost:5000/")
    print("  curl -H 'X-Customer-ID: enterprise-123' http://localhost:5000/protected")
    print("  curl -H 'X-User-ID: user_456' http://localhost:5000/ai-operation")
    print()

    # Security: Control debug mode via environment variable
    # Never use debug=True in production - allows arbitrary code execution
    debug_mode = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    app.run(debug=debug_mode, port=5000)
