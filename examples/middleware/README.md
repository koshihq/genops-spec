# 🔧 Web Framework Middleware for GenOps AI Attribution

This directory contains production-ready middleware implementations for popular Python web frameworks. These middleware components automatically set up attribution context for all AI operations in your web applications.

## 📁 Available Middleware

### 🌟 [Flask Middleware](flask_middleware.py)
Complete Flask middleware with session management, JWT integration, and performance tracking.

**Features:**
- ✅ Automatic request/response attribution context
- ✅ Flask-Login and Flask-JWT-Extended integration
- ✅ Session-based attribution tracking
- ✅ Custom decorators for operation-specific attribution
- ✅ Performance monitoring and debugging support

### 🚀 [FastAPI Middleware](fastapi_middleware.py) 
Async-compatible FastAPI middleware with dependency injection and JWT authentication.

**Features:**
- ✅ Full async/await support with proper context management
- ✅ JWT token parsing and claims extraction
- ✅ Dependency injection for attribution context
- ✅ OpenAPI documentation integration
- ✅ Request tracing and performance metrics

### 🎸 [Django Middleware](django_middleware.py)
Django middleware integrated with Django's authentication system and user models.

**Features:**
- ✅ Django User model integration
- ✅ Django REST Framework token authentication
- ✅ Session-based attribution management
- ✅ Custom user model field support
- ✅ Management command for setup

## 🚀 Quick Start

### Flask Setup

```python
from flask import Flask
from examples.middleware.flask_middleware import GenOpsFlaskMiddleware

app = Flask(__name__)

# Configure defaults
app.config['GENOPS_DEFAULTS'] = {
    'team': 'backend-engineering',
    'project': 'ai-api',
    'environment': 'production'
}

# Initialize middleware
GenOpsFlaskMiddleware(app, debug=True)

@app.route('/ai-operation')
def ai_operation():
    # All AI operations automatically get attribution
    attrs = genops.get_effective_attributes()
    return {'attribution': attrs}
```

### FastAPI Setup

```python
from fastapi import FastAPI
from examples.middleware.fastapi_middleware import GenOpsFastAPIMiddleware

app = FastAPI(title="AI Service")

# Initialize middleware
GenOpsFastAPIMiddleware(
    app,
    team="backend-engineering",
    project="ai-api",
    environment="production"
)

@app.post("/ai-operation")
async def ai_operation(
    effective_attrs: dict = Depends(get_effective_attributes)
):
    # Attribution automatically available via dependency injection
    return {"attribution": effective_attrs}
```

### Django Setup

```python
# settings.py
MIDDLEWARE = [
    # ... other middleware
    'examples.middleware.django_middleware.GenOpsDjangoMiddleware',
]

GENOPS_DEFAULTS = {
    'team': 'backend-engineering',
    'project': 'ai-api',
    'environment': 'production'
}

# views.py  
from django.http import JsonResponse
import genops

def ai_operation(request):
    attrs = genops.get_effective_attributes()
    return JsonResponse({'attribution': attrs})
```

## 🏷️ Attribution Headers

All middleware implementations support these standard headers for multi-tenant attribution:

| Header | Purpose | Example |
|--------|---------|---------|
| `X-Customer-ID` | Customer identification | `enterprise-123` |
| `X-User-ID` | User identification | `user_456` |
| `X-Tenant-ID` | Tenant/organization ID | `tenant_789` |
| `X-Trace-ID` | Request tracing | `trace_abc123` |

## 🎯 Attribution Context

The middleware automatically captures and sets attribution context including:

### Request Information
- `request_id` - Unique request identifier
- `method` - HTTP method (GET, POST, etc.)
- `path` - Request path
- `endpoint` - Framework-specific endpoint name
- `user_agent` - Client user agent
- `client_ip` - Client IP address

### User Information  
- `user_id` - Authenticated user ID
- `user_email` - User email address
- `user_role` - User role/permissions
- `user_tier` - User subscription tier

### Customer Information
- `customer_id` - Customer/organization ID
- `tenant_id` - Multi-tenant organization ID
- `customer_tier` - Customer subscription level

### Performance Metrics
- `request_duration_ms` - Request processing time
- `response_status` - HTTP response status code
- `response_size` - Response content size

### Session Information (where applicable)
- `session_id` - User session identifier
- `session_key` - Framework session key

## 🔐 Authentication Integration

### JWT Tokens
All middleware can extract attribution from JWT token claims:

```json
{
  "sub": "user_123",
  "role": "admin", 
  "customer_id": "enterprise-456",
  "tier": "premium",
  "exp": 1234567890
}
```

### Session-Based Authentication
Framework-specific user objects and sessions are automatically integrated:

- **Flask**: Flask-Login `current_user`
- **FastAPI**: JWT dependency injection
- **Django**: `request.user` and `request.session`

## 🛡️ Security Best Practices

### Header Validation
```python
# Only accept customer IDs from trusted sources
@require_attribution(customer_id=True)
def protected_endpoint():
    # Guaranteed to have customer_id in context
    pass
```

### Token Security
```python
# JWT tokens are validated before extraction
jwt_bearer = JWTBearer(jwt_secret=os.environ["JWT_SECRET"])

@app.get("/protected")
async def protected(token: dict = Depends(jwt_bearer)):
    # Token is validated and claims extracted
    pass
```

### PII Protection
```python
# Middleware automatically excludes sensitive data
# Only IDs and roles are captured, not PII like emails or names
```

## 📊 Performance Impact

The middleware is designed for minimal performance overhead:

- **Flask**: ~0.1ms per request
- **FastAPI**: ~0.05ms per request (async optimized)
- **Django**: ~0.2ms per request

Performance tracking can be disabled in production if needed:
```python
middleware = GenOpsMiddleware(enable_performance_tracking=False)
```

## 🔧 Configuration Options

All middleware support these configuration options:

```python
{
    'customer_header': 'x-customer-id',           # Customer ID header
    'user_header': 'x-user-id',                  # User ID header
    'tenant_header': 'x-tenant-id',              # Tenant ID header
    'trace_header': 'x-trace-id',                # Trace ID header
    'environment': 'production',                 # Environment name
    'enable_session_tracking': True,             # Track sessions
    'enable_performance_tracking': True,         # Performance metrics
    'debug': False,                              # Debug logging
    'fallback_customer_id': None                 # Default customer ID
}
```

## 🔄 Context Lifecycle

1. **Request Start**: Middleware extracts attribution from headers, JWT, session
2. **Context Set**: `genops.set_context()` called with attribution data
3. **Request Processing**: All AI operations inherit the context automatically
4. **Response**: Performance metrics added to context
5. **Request End**: `genops.clear_context()` called to clean up

## 🧪 Testing Your Integration

### Test Attribution Context
```bash
# Test basic attribution
curl -H "X-Customer-ID: test-123" -H "X-User-ID: user-456" http://localhost:8000/attribution

# Test protected endpoints  
curl -H "X-Customer-ID: enterprise-123" http://localhost:8000/protected

# Test JWT authentication
curl -H "Authorization: Bearer eyJ..." http://localhost:8000/protected-jwt
```

### Verify Context Inheritance
```python
def test_ai_operation():
    # Context should be automatically available
    attrs = genops.get_effective_attributes()
    assert 'customer_id' in attrs
    assert 'user_id' in attrs
    assert 'request_id' in attrs
```

## 📈 Monitoring & Observability

All attribution data automatically flows to your observability platform via OpenTelemetry:

```sql
-- Query attribution in your observability platform
SELECT customer_id, COUNT(*) as requests, AVG(cost) as avg_cost
FROM ai_operations 
WHERE genops.team = 'backend-engineering'
GROUP BY customer_id
```

## 🆘 Troubleshooting

### Context Not Available
- Ensure middleware is properly installed and configured
- Check that `genops.clear_context()` isn't called prematurely
- Verify header names match your configuration

### Performance Issues
- Disable performance tracking in high-traffic environments
- Use async middleware (FastAPI) for better concurrency
- Consider caching user/customer lookups

### Authentication Integration
- Verify JWT secrets match between auth and middleware
- Check user model field names in Django configuration
- Ensure Flask-Login is properly initialized

## 🔗 Next Steps

1. **Install Middleware**: Choose your framework and integrate the middleware
2. **Configure Headers**: Set up client applications to send attribution headers
3. **Test Integration**: Verify attribution context is properly set
4. **Monitor Results**: Check your observability platform for attribution data
5. **Customize Rules**: Add validation rules for your specific requirements

For more examples and advanced configurations, see the individual middleware files and the main [attribution guide](../attribution_guide.py).