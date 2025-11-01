# OpenAI Integration Guide

## Overview

The GenOps OpenAI adapter provides comprehensive governance telemetry for OpenAI applications, including:

- **Chat completion tracking** with detailed cost and performance metrics
- **Multi-model cost optimization** with intelligent model selection
- **Token usage analytics** for cost forecasting and optimization
- **Error tracking and success rate monitoring** for reliability insights
- **Policy enforcement** with governance attribute propagation

## Quick Start

### Installation

```bash
pip install genops-ai[openai]
```

### Basic Setup

The simplest way to add GenOps tracking to your OpenAI application:

```python
from genops.providers.openai import instrument_openai

# Initialize GenOps OpenAI adapter
client = instrument_openai(api_key="your_openai_key")

# Your existing OpenAI code works unchanged
response = client.chat_completions_create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "What is artificial intelligence?"}],
    team="ai-research",
    project="knowledge-base",
    customer_id="customer_123"
)
```

### Auto-Instrumentation (Recommended)

For zero-code setup, enable auto-instrumentation:

```python
from genops import auto_instrument

# Automatically instrument all supported providers
auto_instrument()

# Your OpenAI code automatically gets governance telemetry
from openai import OpenAI
client = OpenAI()
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Your query here"}]
)  # Automatically tracked!
```

## Core Features

### 1. Chat Completion Tracking

Track OpenAI chat completions with detailed telemetry:

```python
from genops.providers.openai import instrument_openai

client = instrument_openai()

# Track completion with governance attributes
response = client.chat_completions_create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Explain quantum computing"}
    ],
    
    # Governance attributes for cost attribution
    team="education-team",
    project="ai-tutoring", 
    environment="production",
    customer_id="edu_customer_456",
    
    # OpenAI parameters
    temperature=0.7,
    max_tokens=500
)
```

**Telemetry Captured:**
- Request/response timing and latency
- Token usage (input, output, total) by model
- Exact cost calculation using current OpenAI pricing
- Success/error rates and error categorization
- Governance attribute propagation

### 2. Legacy Completion Support

Support for OpenAI legacy completion endpoints:

```python
# Legacy completions also supported
response = client.completions_create(
    model="gpt-3.5-turbo-instruct",
    prompt="Write a haiku about programming",
    max_tokens=100,
    
    # Same governance attributes
    team="content-team",
    project="creative-writing"
)
```

### 3. Cost Optimization and Model Selection

Intelligent model selection based on use case complexity:

```python
def smart_completion(prompt: str, complexity: str = "simple"):
    """Choose optimal model based on complexity for cost efficiency."""
    
    model_configs = {
        "simple": {
            "model": "gpt-3.5-turbo",
            "max_tokens": 150,
            "temperature": 0.3,
            "cost_per_1k_input": 0.0015,
            "cost_per_1k_output": 0.002
        },
        "balanced": {
            "model": "gpt-4o-mini", 
            "max_tokens": 300,
            "temperature": 0.5,
            "cost_per_1k_input": 0.00015,
            "cost_per_1k_output": 0.0006
        },
        "complex": {
            "model": "gpt-4",
            "max_tokens": 800,
            "temperature": 0.7,
            "cost_per_1k_input": 0.03,
            "cost_per_1k_output": 0.06
        },
        "advanced": {
            "model": "gpt-4-turbo",
            "max_tokens": 1000,
            "temperature": 0.7,
            "cost_per_1k_input": 0.01,
            "cost_per_1k_output": 0.03
        }
    }
    
    config = model_configs.get(complexity, model_configs["simple"])
    
    response = client.chat_completions_create(
        model=config["model"],
        messages=[{"role": "user", "content": prompt}],
        max_tokens=config["max_tokens"],
        temperature=config["temperature"],
        
        # Cost attribution
        team="optimization-team",
        project="smart-routing",
        complexity_level=complexity,
        estimated_cost_per_1k=config["cost_per_1k_input"]
    )
    
    return response.choices[0].message.content
```

### 4. Batch Processing with Cost Tracking

Handle batch operations with comprehensive cost tracking:

```python
from genops import track

def process_customer_queries(queries: list, customer_id: str):
    """Process multiple queries with detailed cost attribution."""
    
    total_cost = 0
    results = []
    
    with track("batch_processing", 
               customer_id=customer_id, 
               team="customer-support") as span:
        
        for i, query in enumerate(queries):
            response = client.chat_completions_create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Provide helpful customer support"},
                    {"role": "user", "content": query}
                ],
                
                # Individual query attribution
                team="customer-support",
                customer_id=customer_id,
                query_index=i,
                batch_id=f"batch_{customer_id}"
            )
            
            results.append(response.choices[0].message.content)
        
        # Track batch-level metrics
        span.set_attribute("queries_processed", len(queries))
        span.set_attribute("batch_size", len(queries))
        
    return results
```

### 5. Function Calling and Tool Usage

Track OpenAI function calling with detailed metrics:

```python
def weather_assistant(location: str):
    """Assistant with function calling capabilities."""
    
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"}
                    }
                }
            }
        }
    ]
    
    response = client.chat_completions_create(
        model="gpt-4-turbo",
        messages=[
            {"role": "user", "content": f"What's the weather like in {location}?"}
        ],
        tools=tools,
        tool_choice="auto",
        
        # Function calling attribution
        team="assistant-team",
        project="weather-bot",
        feature="function_calling",
        tools_available=len(tools)
    )
    
    # Handle function calls
    if response.choices[0].message.tool_calls:
        for tool_call in response.choices[0].message.tool_calls:
            if tool_call.function.name == "get_weather":
                # Your weather API call here
                weather_data = {"temperature": "72°F", "condition": "sunny"}
                return f"Weather in {location}: {weather_data}"
    
    return response.choices[0].message.content
```

## Integration Patterns

### Pattern 1: Decorator-Based Instrumentation

```python
from genops.decorators import track_openai

@track_openai(
    team="content-generation",
    project="blog-automation"
)
def generate_blog_post(topic: str, style: str = "informative") -> str:
    response = client.chat_completions_create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": f"Write a {style} blog post"},
            {"role": "user", "content": f"Topic: {topic}"}
        ]
    )
    return response.choices[0].message.content

# Automatic telemetry on every call
post = generate_blog_post("AI in Healthcare")
```

### Pattern 2: Context Manager Pattern

```python
from genops import track

def multi_step_analysis(document: str, customer_id: str):
    with track(f"document_analysis_{customer_id}", 
               customer_id=customer_id, team="analysis-team") as span:
        
        # Step 1: Summarization
        summary = client.chat_completions_create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": f"Summarize: {document}"}]
        )
        
        # Step 2: Key points extraction  
        key_points = client.chat_completions_create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": f"Extract key points: {document}"}]
        )
        
        # Step 3: Sentiment analysis
        sentiment = client.chat_completions_create(
            model="gpt-3.5-turbo", 
            messages=[{"role": "user", "content": f"Analyze sentiment: {document}"}]
        )
        
        span.set_attribute("analysis_steps", 3)
        return {
            "summary": summary.choices[0].message.content,
            "key_points": key_points.choices[0].message.content,
            "sentiment": sentiment.choices[0].message.content
        }
```

### Pattern 3: Policy Enforcement

```python
from genops.core.policy import enforce_policy

@enforce_policy("content_moderation")
def process_user_content(content: str, user_id: str):
    return client.chat_completions_create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "Moderate this content for safety"},
            {"role": "user", "content": content}
        ],
        user_id=user_id,
        team="content-safety"
    )
```

## Configuration

### Environment Variables

```bash
# OpenAI configuration
export OPENAI_API_KEY="your_openai_key"
export OPENAI_ORG_ID="your_org_id"  # Optional
export OPENAI_PROJECT_ID="your_project_id"  # Optional

# OpenTelemetry configuration
export OTEL_SERVICE_NAME="my-openai-app"
export OTEL_EXPORTER_OTLP_ENDPOINT="http://localhost:4317"

# GenOps OpenAI configuration
export GENOPS_OPENAI_AUTO_INSTRUMENT=true
export GENOPS_OPENAI_COST_TRACKING=true
export GENOPS_OPENAI_MAX_RETRIES=3
```

### Programmatic Configuration

```python
from genops.providers.openai import configure_openai_adapter

configure_openai_adapter({
    "auto_instrument": True,
    "cost_tracking": {
        "enabled": True,
        "include_embeddings": True,
        "track_streaming": True
    },
    "telemetry": {
        "service_name": "my-openai-service",
        "attributes": {
            "deployment.environment": "production",
            "service.version": "1.0.0"
        }
    },
    "rate_limiting": {
        "requests_per_minute": 60,
        "tokens_per_minute": 90000
    }
})
```

## Advanced Features

### Streaming Responses

```python
def streaming_completion(prompt: str):
    """Handle streaming responses with telemetry."""
    
    stream = client.chat_completions_create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        stream=True,
        
        # Governance attributes
        team="streaming-team",
        project="real-time-chat",
        streaming=True
    )
    
    full_response = ""
    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            content = chunk.choices[0].delta.content
            full_response += content
            print(content, end="")
    
    return full_response
```

### Embeddings Support

```python
def semantic_search(query: str, documents: list):
    """Create embeddings with cost tracking."""
    
    # Get query embedding
    query_embedding = client.embeddings.create(
        model="text-embedding-3-small",
        input=query,
        
        # Governance attributes
        team="search-team",
        project="semantic-search",
        operation_type="query_embedding"
    )
    
    # Get document embeddings
    doc_embeddings = client.embeddings.create(
        model="text-embedding-3-small",
        input=documents[:100],  # Batch limit
        
        team="search-team", 
        project="semantic-search",
        operation_type="document_embedding",
        document_count=len(documents)
    )
    
    # Your similarity calculation here
    return {"query_embedding": query_embedding, "doc_embeddings": doc_embeddings}
```

### Image Analysis (Vision)

```python
def analyze_image(image_url: str, question: str):
    """Analyze images with GPT-4 Vision."""
    
    response = client.chat_completions_create(
        model="gpt-4-vision-preview",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {"type": "image_url", "image_url": {"url": image_url}}
                ]
            }
        ],
        max_tokens=300,
        
        # Vision-specific attributes
        team="vision-team",
        project="image-analysis",
        has_image=True,
        image_source=image_url
    )
    
    return response.choices[0].message.content
```

## Troubleshooting

### Common Issues

#### Issue: "OpenAI API key not found"
```python
# Solution: Verify API key setup
import os
print("API key set:", bool(os.getenv("OPENAI_API_KEY")))

# Or set programmatically
from genops.providers.openai import instrument_openai
client = instrument_openai(api_key="your_key_here")
```

#### Issue: Cost tracking not working
```python
# Check if cost calculation is enabled
from genops.providers.openai import validate_setup, print_validation_result

result = validate_setup()
print_validation_result(result)

# Enable debug logging
import logging
logging.getLogger("genops.providers.openai").setLevel(logging.DEBUG)
```

#### Issue: Telemetry not appearing in observability platform
```python
# Verify OpenTelemetry configuration
from opentelemetry import trace

tracer = trace.get_tracer(__name__)
with tracer.start_as_current_span("test-span") as span:
    span.set_attribute("test", "value")
    print("OpenTelemetry is working")

# Check OTLP exporter configuration
import os
print("OTLP Endpoint:", os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"))
```

### Debug Mode

Enable comprehensive debug logging:

```python
import logging

# Enable GenOps debug logging
logging.getLogger("genops").setLevel(logging.DEBUG)

# Enable OpenAI adapter debug logging
logging.getLogger("genops.providers.openai").setLevel(logging.DEBUG)

# Enable OpenTelemetry debug logging
logging.getLogger("opentelemetry").setLevel(logging.DEBUG)
```

### Validation Utilities

Verify your setup is working correctly:

```python
from genops.providers.openai import validate_setup, print_validation_result

# Run comprehensive setup validation
validation_result = validate_setup()
print_validation_result(validation_result)

if validation_result.is_valid:
    print("✅ GenOps OpenAI setup is valid!")
else:
    print("❌ Setup issues found:")
    for issue in validation_result.issues:
        if issue.level == "error":
            print(f"  - ERROR: {issue.message}")
            if issue.fix_suggestion:
                print(f"    Fix: {issue.fix_suggestion}")
```

## Performance Considerations

### Best Practices

1. **Use appropriate models** for task complexity to optimize costs
2. **Enable batch processing** for multiple requests to reduce API overhead
3. **Configure reasonable timeouts** to handle network issues gracefully
4. **Implement retry logic** with exponential backoff for rate limits

### Performance Tuning

```python
from genops.providers.openai import configure_performance

configure_performance({
    "connection_pool_size": 10,
    "request_timeout": 30,
    "max_retries": 3,
    "retry_delay": 1.0,
    "batch_size": 20,
    "async_export": True
})
```

## Cost Management

### Model Cost Comparison

| Model | Input (per 1K tokens) | Output (per 1K tokens) | Best For |
|-------|----------------------|------------------------|----------|
| gpt-4o-mini | $0.00015 | $0.0006 | Simple tasks, high volume |
| gpt-3.5-turbo | $0.0015 | $0.002 | General purpose, balanced |
| gpt-4o | $0.005 | $0.015 | Complex reasoning |
| gpt-4-turbo | $0.01 | $0.03 | Advanced capabilities |
| gpt-4 | $0.03 | $0.06 | Highest quality |

### Cost Optimization Strategies

```python
def cost_optimized_completion(prompt: str, max_cost: float = 0.10):
    """Choose model based on cost constraints."""
    
    estimated_tokens = len(prompt.split()) * 1.3
    
    models = [
        ("gpt-4o-mini", 0.00015, 0.0006),
        ("gpt-3.5-turbo", 0.0015, 0.002),
        ("gpt-4o", 0.005, 0.015),
        ("gpt-4-turbo", 0.01, 0.03)
    ]
    
    for model, input_cost, output_cost in models:
        estimated_cost = (estimated_tokens * input_cost) + (200 * output_cost)  # Assume 200 output tokens
        
        if estimated_cost <= max_cost:
            response = client.chat_completions_create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                
                # Cost tracking
                team="cost-optimization",
                estimated_cost=estimated_cost,
                max_budget=max_cost
            )
            return response.choices[0].message.content
    
    raise ValueError(f"No model available within budget of ${max_cost}")
```

## Next Steps

- Explore the [complete examples](../examples/openai/) for advanced patterns
- Check out [governance scenarios](../examples/governance_scenarios/) for policy enforcement
- Review [observability integration](../observability/) for dashboard setup
- See [API reference](../api/openai.md) for detailed method documentation

## Support

- **Issues:** [GitHub Issues](https://github.com/genops-ai/genops-ai/issues)
- **Discussions:** [GitHub Discussions](https://github.com/genops-ai/genops-ai/discussions)  
- **Documentation:** [Full Documentation](https://docs.genops.ai)
- **OpenAI Docs:** [OpenAI API Documentation](https://platform.openai.com/docs/)