#!/usr/bin/env python3
"""
Production Flask application with GenOps OpenRouter integration.

This example demonstrates a production-ready web service that uses GenOps
for comprehensive AI governance across OpenRouter's 400+ models.
"""

import os
import logging
from flask import Flask, request, jsonify
from dotenv import load_dotenv
import structlog

# Load environment variables
load_dotenv()

# Initialize structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Initialize GenOps with production configuration
import genops
genops.init(
    service_name=os.getenv("OTEL_SERVICE_NAME", "openrouter-production-service"),
    service_version=os.getenv("OTEL_SERVICE_VERSION", "1.0.0"),
    environment=os.getenv("ENVIRONMENT", "production"),
    exporter_type="otlp",
    otlp_endpoint=os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"),
    otlp_headers=dict(pair.split("=") for pair in (os.getenv("OTEL_EXPORTER_OTLP_HEADERS", "").split(",") if os.getenv("OTEL_EXPORTER_OTLP_HEADERS") else [])),
    default_team=os.getenv("DEFAULT_TEAM", "platform"),
    default_project=os.getenv("DEFAULT_PROJECT", "openrouter-service")
)

from genops.providers.openrouter import instrument_openrouter

# Initialize Flask app
app = Flask(__name__)

# Global OpenRouter client
openrouter_client = None

def get_openrouter_client():
    """Get or create OpenRouter client with proper error handling."""
    global openrouter_client
    
    if openrouter_client is None:
        try:
            openrouter_client = instrument_openrouter(
                openrouter_api_key=os.getenv("OPENROUTER_API_KEY"),
                timeout=float(os.getenv("OPENROUTER_TIMEOUT", "30.0")),
                max_retries=int(os.getenv("OPENROUTER_MAX_RETRIES", "3"))
            )
            logger.info("OpenRouter client initialized successfully")
        except Exception as e:
            logger.error("Failed to initialize OpenRouter client", error=str(e))
            raise
    
    return openrouter_client


@app.route("/health")
def health_check():
    """Health check endpoint for container orchestration."""
    try:
        from genops.providers.openrouter import validate_setup
        
        result = validate_setup()
        
        return jsonify({
            "status": "healthy" if result.is_valid else "degraded",
            "service": os.getenv("OTEL_SERVICE_NAME", "openrouter-service"),
            "version": os.getenv("OTEL_SERVICE_VERSION", "1.0.0"),
            "environment": os.getenv("ENVIRONMENT", "production"),
            "validation": {
                "is_valid": result.is_valid,
                "error_count": result.summary.get("error_count", 0),
                "warning_count": result.summary.get("warning_count", 0)
            }
        }), 200 if result.is_valid else 503
    
    except Exception as e:
        logger.error("Health check failed", error=str(e))
        return jsonify({
            "status": "unhealthy",
            "error": str(e)
        }), 500


@app.route("/ready")
def readiness_check():
    """Readiness check for Kubernetes deployments."""
    try:
        client = get_openrouter_client()
        
        return jsonify({
            "status": "ready",
            "openrouter_client": "initialized",
            "timestamp": structlog.processors.TimeStamper(fmt="iso").__call__(None, None, None)["timestamp"]
        }), 200
    
    except Exception as e:
        logger.error("Readiness check failed", error=str(e))
        return jsonify({
            "status": "not_ready",
            "error": str(e)
        }), 503


@app.route("/chat/completions", methods=["POST"])
def chat_completions():
    """
    Production chat completions endpoint with comprehensive governance.
    
    Request body should include:
    - model: OpenRouter model name
    - messages: Array of messages
    - governance attributes (optional): team, project, customer_id, etc.
    """
    try:
        client = get_openrouter_client()
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "Missing request body"}), 400
        
        if "model" not in data:
            return jsonify({"error": "Missing 'model' parameter"}), 400
            
        if "messages" not in data:
            return jsonify({"error": "Missing 'messages' parameter"}), 400
        
        # Extract governance attributes from request
        governance_attrs = {}
        governance_keys = [
            "team", "project", "customer_id", "customer", "environment",
            "cost_center", "feature", "user_id", "experiment_id", "region",
            "model_version", "priority", "compliance_level"
        ]
        
        for key in governance_keys:
            if key in data:
                governance_attrs[key] = data[key]
        
        # Add request-level context
        governance_attrs.update({
            "request_id": request.headers.get("X-Request-ID", "unknown"),
            "user_agent": request.headers.get("User-Agent", "unknown"),
            "endpoint": "/chat/completions"
        })
        
        # Prepare OpenAI-compatible parameters
        openai_params = {
            "model": data["model"],
            "messages": data["messages"]
        }
        
        # Add optional OpenAI parameters
        openai_optional_params = [
            "temperature", "max_tokens", "top_p", "frequency_penalty",
            "presence_penalty", "stream", "stop", "n", "logit_bias",
            "user", "response_format", "seed", "tools", "tool_choice"
        ]
        
        for param in openai_optional_params:
            if param in data:
                openai_params[param] = data[param]
        
        # Add OpenRouter-specific parameters
        openrouter_params = ["provider", "route", "fallbacks"]
        for param in openrouter_params:
            if param in data:
                openai_params[param] = data[param]
        
        logger.info(
            "Processing chat completion request",
            model=data["model"],
            team=governance_attrs.get("team"),
            customer_id=governance_attrs.get("customer_id")
        )
        
        # Make the instrumented request
        response = client.chat_completions_create(
            **openai_params,
            **governance_attrs
        )
        
        logger.info(
            "Chat completion successful",
            model=data["model"],
            tokens_used=getattr(response.usage, "total_tokens", 0) if hasattr(response, "usage") else 0
        )
        
        # Return the response in OpenAI-compatible format
        return response.model_dump() if hasattr(response, 'model_dump') else response.dict()
    
    except Exception as e:
        logger.error(
            "Chat completion failed",
            error=str(e),
            model=data.get("model") if 'data' in locals() else "unknown"
        )
        return jsonify({
            "error": {
                "message": str(e),
                "type": "internal_error",
                "code": "openrouter_error"
            }
        }), 500


@app.route("/models")
def list_models():
    """List available models endpoint."""
    try:
        # This would typically fetch from OpenRouter API
        # For now, return a sample of supported models
        models = [
            {"id": "openai/gpt-4o", "provider": "openai", "pricing_tier": "premium"},
            {"id": "anthropic/claude-3-5-sonnet", "provider": "anthropic", "pricing_tier": "premium"},
            {"id": "google/gemini-1.5-pro", "provider": "google", "pricing_tier": "premium"},
            {"id": "meta-llama/llama-3.1-8b-instruct", "provider": "meta", "pricing_tier": "balanced"},
            {"id": "meta-llama/llama-3.2-3b-instruct", "provider": "meta", "pricing_tier": "economy"},
            {"id": "anthropic/claude-3-haiku", "provider": "anthropic", "pricing_tier": "balanced"},
            {"id": "openai/gpt-3.5-turbo", "provider": "openai", "pricing_tier": "balanced"}
        ]
        
        return jsonify({
            "object": "list",
            "data": models
        })
    
    except Exception as e:
        logger.error("Failed to list models", error=str(e))
        return jsonify({"error": str(e)}), 500


@app.route("/cost/estimate", methods=["POST"])
def estimate_cost():
    """Estimate cost for a potential request."""
    try:
        from genops.providers.openrouter_pricing import calculate_openrouter_cost
        
        data = request.get_json()
        if not data or "model" not in data:
            return jsonify({"error": "Missing model parameter"}), 400
        
        model = data["model"]
        input_tokens = data.get("input_tokens", 100)
        output_tokens = data.get("output_tokens", 50)
        
        cost = calculate_openrouter_cost(
            model,
            input_tokens=input_tokens,
            output_tokens=output_tokens
        )
        
        return jsonify({
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "estimated_cost": cost,
            "currency": "USD"
        })
    
    except Exception as e:
        logger.error("Cost estimation failed", error=str(e))
        return jsonify({"error": str(e)}), 500


@app.errorhandler(Exception)
def handle_error(error):
    """Global error handler."""
    logger.error("Unhandled exception", error=str(error), exc_info=True)
    return jsonify({
        "error": {
            "message": "An internal error occurred",
            "type": "internal_error"
        }
    }), 500


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    debug = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    
    logger.info(
        "Starting OpenRouter service",
        port=port,
        debug=debug,
        environment=os.getenv("ENVIRONMENT", "production")
    )
    
    app.run(host="0.0.0.0", port=port, debug=debug)