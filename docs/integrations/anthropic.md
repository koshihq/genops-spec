# Anthropic Integration Guide

## Overview

The GenOps Anthropic adapter provides comprehensive governance telemetry for Claude applications, including:

- **Message completion tracking** with detailed cost and performance metrics
- **Multi-model cost optimization** across Claude 3 variants (Haiku, Sonnet, Opus)
- **Token usage analytics** for cost forecasting and optimization
- **Conversation tracking** for multi-turn dialog systems
- **Policy enforcement** with governance attribute propagation

## Quick Start

### Installation

```bash
pip install genops-ai[anthropic]
```

### Basic Setup

The simplest way to add GenOps tracking to your Anthropic application:

```python
from genops.providers.anthropic import instrument_anthropic

# Initialize GenOps Anthropic adapter
client = instrument_anthropic(api_key="your_anthropic_key")

# Your existing Anthropic code works unchanged
response = client.messages_create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=300,
    messages=[{"role": "user", "content": "Explain machine learning"}],
    team="ai-research",
    project="claude-assistant",
    customer_id="customer_123"
)
```

### Auto-Instrumentation (Recommended)

For zero-code setup, enable auto-instrumentation:

```python
from genops import auto_instrument

# Automatically instrument all supported providers
auto_instrument()

# Your Anthropic code automatically gets governance telemetry
from anthropic import Anthropic
client = Anthropic()
response = client.messages.create(
    model="claude-3-5-haiku-20241022",
    max_tokens=200,
    messages=[{"role": "user", "content": "Your query here"}]
)  # Automatically tracked!
```

## Core Features

### 1. Message Completion Tracking

Track Claude messages with detailed telemetry:

```python
from genops.providers.anthropic import instrument_anthropic

client = instrument_anthropic()

# Track message with governance attributes
response = client.messages_create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1000,
    messages=[
        {"role": "user", "content": "Analyze this business strategy document and provide insights"}
    ],
    
    # Governance attributes for cost attribution
    team="strategy-team",
    project="business-analysis", 
    environment="production",
    customer_id="enterprise_customer_789",
    
    # Claude parameters
    temperature=0.7,
    top_p=0.9,
    top_k=40
)
```

**Telemetry Captured:**
- Request/response timing and latency
- Token usage (input, output) by Claude model
- Exact cost calculation using current Anthropic pricing
- Success/error rates and error categorization
- Governance attribute propagation

### 2. Multi-Model Intelligence and Cost Optimization

Intelligent model selection across Claude 3 variants:

```python
def smart_claude_completion(prompt: str, complexity: str = "balanced"):
    """Choose optimal Claude model based on task complexity."""
    
    model_configs = {
        "simple": {
            "model": "claude-3-haiku-20240307",
            "max_tokens": 200,
            "temperature": 0.3,
            "cost_per_1m_input": 0.25,
            "cost_per_1m_output": 1.25,
            "use_case": "Simple Q&A, basic text processing"
        },
        "balanced": {
            "model": "claude-3-5-haiku-20241022", 
            "max_tokens": 500,
            "temperature": 0.5,
            "cost_per_1m_input": 1.00,
            "cost_per_1m_output": 5.00,
            "use_case": "General tasks, moderate complexity"
        },
        "advanced": {
            "model": "claude-3-5-sonnet-20241022",
            "max_tokens": 1000,
            "temperature": 0.7,
            "cost_per_1m_input": 3.00,
            "cost_per_1m_output": 15.00,
            "use_case": "Complex reasoning, analysis, coding"
        },
        "expert": {
            "model": "claude-3-opus-20240229",
            "max_tokens": 1500,
            "temperature": 0.8,
            "cost_per_1m_input": 15.00,
            "cost_per_1m_output": 75.00,
            "use_case": "Highest quality, creative tasks"
        }
    }
    
    config = model_configs.get(complexity, model_configs["balanced"])
    
    response = client.messages_create(
        model=config["model"],
        max_tokens=config["max_tokens"],
        temperature=config["temperature"],
        messages=[{"role": "user", "content": prompt}],
        
        # Cost attribution and optimization tracking
        team="optimization-team",
        project="smart-routing",
        complexity_level=complexity,
        estimated_cost_per_1m=config["cost_per_1m_input"],
        use_case=config["use_case"]
    )
    
    return response.content[0].text
```

### 3. Multi-Turn Conversations

Handle conversational flows with comprehensive tracking:

```python
from genops import track

def conversational_agent(conversation_history: list, customer_id: str):
    """Handle multi-turn conversations with detailed cost tracking."""
    
    with track("conversation_session", 
               customer_id=customer_id, 
               team="customer-support") as span:
        
        response = client.messages_create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=600,
            messages=conversation_history,
            
            # Conversation-specific attributes
            team="customer-support",
            customer_id=customer_id,
            conversation_turn=len(conversation_history),
            conversation_type="support_chat"
        )
        
        # Track conversation metrics
        total_chars = sum(len(msg.get("content", "")) for msg in conversation_history)
        span.set_attribute("conversation_turns", len(conversation_history))
        span.set_attribute("total_conversation_chars", total_chars)
        span.set_attribute("customer_tier", "enterprise")  # Dynamic customer data
        
        return response.content[0].text
```

### 4. Document Analysis and Processing

Specialized patterns for document analysis:

```python
def analyze_legal_document(document_text: str, analysis_type: str):
    """Analyze legal documents with specialized prompts."""
    
    analysis_prompts = {
        "contract_review": "Review this contract for key terms, obligations, and potential risks:",
        "compliance_check": "Check this document for regulatory compliance issues:",
        "summary": "Provide a concise executive summary of this legal document:",
        "risk_assessment": "Identify and assess legal risks in this document:"
    }
    
    system_prompt = analysis_prompts.get(analysis_type, analysis_prompts["summary"])
    
    response = client.messages_create(
        model="claude-3-5-sonnet-20241022",  # Best for complex analysis
        max_tokens=2000,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": document_text}
        ],
        
        # Legal analysis specific attributes
        team="legal-team",
        project="document-analysis",
        analysis_type=analysis_type,
        document_length=len(document_text),
        requires_expertise="legal"
    )
    
    return response.content[0].text
```

### 5. Code Generation and Review

Track coding assistance with detailed metrics:

```python
def code_assistant(code_request: str, language: str = "python"):
    """Generate or review code with Claude."""
    
    system_prompts = {
        "python": "You are an expert Python developer. Write clean, efficient, well-documented code.",
        "javascript": "You are an expert JavaScript developer. Follow modern ES6+ standards.",
        "sql": "You are a database expert. Write efficient, secure SQL queries.",
        "review": "You are a senior code reviewer. Provide constructive feedback on code quality."
    }
    
    response = client.messages_create(
        model="claude-3-5-sonnet-20241022",  # Best for coding
        max_tokens=1500,
        messages=[
            {"role": "system", "content": system_prompts.get(language, system_prompts["python"])},
            {"role": "user", "content": code_request}
        ],
        
        # Code-specific attributes
        team="engineering-team",
        project="ai-coding-assistant",
        programming_language=language,
        task_type="code_generation",
        complexity="intermediate"
    )
    
    return response.content[0].text
```

## Integration Patterns

### Pattern 1: Decorator-Based Instrumentation

```python
from genops.decorators import track_anthropic

@track_anthropic(
    team="research-team",
    project="academic-writing"
)
def generate_research_summary(papers: list, topic: str) -> str:
    combined_content = "\n\n".join(papers)
    
    response = client.messages_create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1200,
        messages=[
            {"role": "system", "content": "Synthesize research papers into a comprehensive summary"},
            {"role": "user", "content": f"Topic: {topic}\n\nPapers:\n{combined_content}"}
        ]
    )
    return response.content[0].text

# Automatic telemetry on every call
summary = generate_research_summary(paper_list, "AI Ethics")
```

### Pattern 2: Context Manager Pattern

```python
from genops import track

def multi_step_content_creation(brief: str, customer_id: str):
    """Create content through multiple Claude interactions."""
    
    with track(f"content_creation_{customer_id}", 
               customer_id=customer_id, 
               team="content-marketing") as span:
        
        # Step 1: Outline creation
        outline = client.messages_create(
            model="claude-3-5-haiku-20241022",  # Fast for outlining
            max_tokens=300,
            messages=[{"role": "user", "content": f"Create an outline for: {brief}"}]
        )
        
        # Step 2: Content expansion
        content = client.messages_create(
            model="claude-3-5-sonnet-20241022",  # Better for detailed content
            max_tokens=1500,
            messages=[
                {"role": "user", "content": f"Write detailed content based on: {outline.content[0].text}"}
            ]
        )
        
        # Step 3: SEO optimization
        seo_content = client.messages_create(
            model="claude-3-5-haiku-20241022",  # Cost-effective for optimization
            max_tokens=800,
            messages=[
                {"role": "user", "content": f"Optimize for SEO: {content.content[0].text}"}
            ]
        )
        
        span.set_attribute("content_creation_steps", 3)
        span.set_attribute("total_tokens_estimated", 2600)
        
        return seo_content.content[0].text
```

### Pattern 3: Policy Enforcement

```python
from genops.core.policy import enforce_policy

@enforce_policy("content_safety")
def process_user_content(user_input: str, user_id: str):
    """Process user content with safety checks."""
    
    return client.messages_create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=500,
        messages=[
            {"role": "system", "content": "Review and moderate user content for safety"},
            {"role": "user", "content": user_input}
        ],
        user_id=user_id,
        team="content-moderation",
        safety_check=True
    )
```

## Configuration

### Environment Variables

```bash
# Anthropic configuration
export ANTHROPIC_API_KEY="your_anthropic_key"

# OpenTelemetry configuration
export OTEL_SERVICE_NAME="my-claude-app"
export OTEL_EXPORTER_OTLP_ENDPOINT="http://localhost:4317"

# GenOps Anthropic configuration
export GENOPS_ANTHROPIC_AUTO_INSTRUMENT=true
export GENOPS_ANTHROPIC_COST_TRACKING=true
export GENOPS_ANTHROPIC_MAX_RETRIES=3
```

### Programmatic Configuration

```python
from genops.providers.anthropic import configure_anthropic_adapter

configure_anthropic_adapter({
    "auto_instrument": True,
    "cost_tracking": {
        "enabled": True,
        "include_system_messages": True,
        "track_conversation_context": True
    },
    "telemetry": {
        "service_name": "my-claude-service",
        "attributes": {
            "deployment.environment": "production",
            "service.version": "1.0.0"
        }
    },
    "model_defaults": {
        "temperature": 0.7,
        "max_tokens": 1000,
        "top_p": 0.9
    }
})
```

## Advanced Features

### Streaming Responses

```python
def streaming_claude_response(prompt: str):
    """Handle streaming responses from Claude."""
    
    stream = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
        
        # Governance attributes
        team="streaming-team",
        project="real-time-chat",
        streaming=True
    )
    
    full_response = ""
    for event in stream:
        if event.type == "content_block_delta":
            content = event.delta.text
            full_response += content
            print(content, end="", flush=True)
    
    return full_response
```

### System Message Optimization

```python
def optimized_system_prompts(task_type: str, user_query: str):
    """Use optimized system prompts for different tasks."""
    
    system_prompts = {
        "analysis": """You are an expert analyst. Provide thorough, structured analysis with:
        1. Executive summary
        2. Key findings
        3. Detailed analysis
        4. Recommendations
        Be concise but comprehensive.""",
        
        "creative": """You are a creative writing expert. Focus on:
        - Engaging storytelling
        - Vivid imagery  
        - Compelling characters
        - Original ideas
        Let creativity flow while maintaining quality.""",
        
        "technical": """You are a technical expert. Provide:
        - Accurate technical information
        - Step-by-step explanations
        - Best practices
        - Practical examples
        Be precise and actionable."""
    }
    
    system_prompt = system_prompts.get(task_type, "You are a helpful assistant.")
    
    response = client.messages_create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1200,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query}
        ],
        
        # System prompt optimization tracking
        team="prompt-optimization",
        task_type=task_type,
        system_prompt_version="v2.1",
        optimization_strategy="task_specific"
    )
    
    return response.content[0].text
```

### Batch Processing Optimization

```python
def batch_process_documents(documents: list, operation: str, customer_id: str):
    """Process multiple documents efficiently with cost optimization."""
    
    # Choose model based on operation complexity
    model_map = {
        "summarize": "claude-3-5-haiku-20241022",      # Fast and cost-effective
        "analyze": "claude-3-5-sonnet-20241022",       # Balanced capability/cost
        "detailed_review": "claude-3-opus-20240229"    # Highest quality
    }
    
    model = model_map.get(operation, "claude-3-5-haiku-20241022")
    
    results = []
    
    with track(f"batch_{operation}_{customer_id}", 
               customer_id=customer_id,
               team="document-processing") as span:
        
        for i, document in enumerate(documents):
            response = client.messages_create(
                model=model,
                max_tokens=500 if operation == "summarize" else 1000,
                messages=[
                    {"role": "system", "content": f"Please {operation} this document"},
                    {"role": "user", "content": document}
                ],
                
                # Individual document tracking
                team="document-processing",
                customer_id=customer_id,
                document_index=i,
                batch_operation=operation,
                batch_size=len(documents)
            )
            
            results.append(response.content[0].text)
        
        # Batch-level metrics
        span.set_attribute("documents_processed", len(documents))
        span.set_attribute("operation_type", operation)
        span.set_attribute("model_used", model)
        
    return results
```

## Troubleshooting

### Common Issues

#### Issue: "Anthropic API key not found"
```python
# Solution: Verify API key setup
import os
print("API key set:", bool(os.getenv("ANTHROPIC_API_KEY")))

# Check key format
key = os.getenv("ANTHROPIC_API_KEY")
if key:
    print("Correct format:", key.startswith("sk-ant-"))

# Or set programmatically
from genops.providers.anthropic import instrument_anthropic
client = instrument_anthropic(api_key="your_key_here")
```

#### Issue: Cost tracking not working
```python
# Check if cost calculation is enabled
from genops.providers.anthropic import validate_setup, print_validation_result

result = validate_setup()
print_validation_result(result)

# Enable debug logging
import logging
logging.getLogger("genops.providers.anthropic").setLevel(logging.DEBUG)
```

#### Issue: Model not available errors
```python
# Use current Claude model names
models = {
    "fastest": "claude-3-haiku-20240307",
    "balanced": "claude-3-5-haiku-20241022", 
    "advanced": "claude-3-5-sonnet-20241022",
    "expert": "claude-3-opus-20240229"
}

# Always check Anthropic docs for latest model names
response = client.messages_create(
    model=models["balanced"],  # Use mapped model names
    max_tokens=500,
    messages=[{"role": "user", "content": "Hello Claude"}]
)
```

### Debug Mode

Enable comprehensive debug logging:

```python
import logging

# Enable GenOps debug logging
logging.getLogger("genops").setLevel(logging.DEBUG)

# Enable Anthropic adapter debug logging
logging.getLogger("genops.providers.anthropic").setLevel(logging.DEBUG)

# Enable OpenTelemetry debug logging
logging.getLogger("opentelemetry").setLevel(logging.DEBUG)
```

### Validation Utilities

Verify your setup is working correctly:

```python
from genops.providers.anthropic import validate_setup, print_validation_result

# Run comprehensive setup validation
validation_result = validate_setup()
print_validation_result(validation_result)

if validation_result.is_valid:
    print("✅ GenOps Anthropic setup is valid!")
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

1. **Choose appropriate Claude models** based on task complexity and cost sensitivity
2. **Use system messages effectively** to provide context and reduce prompt repetition
3. **Implement streaming** for long responses to improve user experience
4. **Batch similar operations** to reduce API overhead

### Performance Tuning

```python
from genops.providers.anthropic import configure_performance

configure_performance({
    "connection_pool_size": 8,
    "request_timeout": 60,  # Claude can take longer than OpenAI
    "max_retries": 3,
    "retry_delay": 1.0,
    "stream_timeout": 120,
    "async_export": True
})
```

## Cost Management

### Claude Model Cost Comparison

| Model | Input (per 1M tokens) | Output (per 1M tokens) | Best For |
|-------|----------------------|------------------------|----------|
| Claude 3 Haiku | $0.25 | $1.25 | Simple tasks, high volume |
| Claude 3.5 Haiku | $1.00 | $5.00 | General purpose, speed |
| Claude 3.5 Sonnet | $3.00 | $15.00 | Complex reasoning, analysis |
| Claude 3 Opus | $15.00 | $75.00 | Highest quality, creative tasks |

### Cost Optimization Strategies

```python
def cost_aware_completion(prompt: str, max_cost: float = 0.50):
    """Choose Claude model based on cost constraints."""
    
    estimated_tokens = len(prompt.split()) * 1.3
    output_tokens = 500  # Estimated
    
    models = [
        ("claude-3-haiku-20240307", 0.25/1000000, 1.25/1000000),
        ("claude-3-5-haiku-20241022", 1.00/1000000, 5.00/1000000),
        ("claude-3-5-sonnet-20241022", 3.00/1000000, 15.00/1000000),
        ("claude-3-opus-20240229", 15.00/1000000, 75.00/1000000)
    ]
    
    for model, input_cost, output_cost in models:
        estimated_cost = (estimated_tokens * input_cost) + (output_tokens * output_cost)
        
        if estimated_cost <= max_cost:
            response = client.messages_create(
                model=model,
                max_tokens=output_tokens,
                messages=[{"role": "user", "content": prompt}],
                
                # Cost tracking
                team="cost-optimization",
                estimated_cost=estimated_cost,
                max_budget=max_cost,
                model_selection="cost_optimized"
            )
            return response.content[0].text
    
    raise ValueError(f"No Claude model available within budget of ${max_cost}")
```

## Next Steps

- Explore the [complete examples](../examples/anthropic/) for advanced patterns
- Check out [governance scenarios](../examples/governance_scenarios/) for policy enforcement
- Review [observability integration](../observability/) for dashboard setup
- See [API reference](../api/anthropic.md) for detailed method documentation

## Support

- **Issues:** [GitHub Issues](https://github.com/genops-ai/genops-ai/issues)
- **Discussions:** [GitHub Discussions](https://github.com/genops-ai/genops-ai/discussions)  
- **Documentation:** [Full Documentation](https://docs.genops.ai)
- **Anthropic Docs:** [Claude API Documentation](https://docs.anthropic.com/claude/reference/)