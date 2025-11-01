#!/usr/bin/env python3
"""
OpenAI Advanced Features Example

This example demonstrates advanced OpenAI features with GenOps telemetry including
streaming responses, function calling, embeddings, and vision capabilities.

What you'll learn:
- Streaming responses with real-time cost tracking
- Function calling and tool usage monitoring
- Embeddings generation with cost analysis
- Vision API (GPT-4 Vision) cost tracking
- Batch operations optimization

Usage:
    python advanced_features.py
    
Prerequisites:
    pip install genops-ai[openai]
    export OPENAI_API_KEY="your_api_key_here"
"""

import os
import sys
import time
import json
from typing import Dict, List, Any, Generator

def streaming_responses_example():
    """Demonstrate streaming responses with GenOps cost tracking."""
    print("🌊 Streaming Responses with Cost Tracking")
    print("-" * 50)
    
    try:
        from genops.providers.openai import instrument_openai
        
        client = instrument_openai()
        
        print("🚀 Starting streaming completion...")
        print("📝 Response (streaming): ", end="", flush=True)
        
        # Create streaming completion
        stream = client.chat_completions_create(
            model="gpt-4",
            messages=[
                {"role": "user", "content": "Write a short story about a robot learning to paint. Make it creative and engaging."}
            ],
            max_tokens=400,
            temperature=0.8,
            stream=True,  # Enable streaming
            
            # Governance attributes for streaming operations
            team="streaming-team",
            project="real-time-content",
            customer_id="streaming-demo",
            feature="creative-writing",
            streaming_enabled=True
        )
        
        # Process streaming response
        full_response = ""
        chunk_count = 0
        start_time = time.time()
        
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                content = chunk.choices[0].delta.content
                full_response += content
                print(content, end="", flush=True)
                chunk_count += 1
                
                # Brief pause for demonstration
                time.sleep(0.02)
        
        end_time = time.time()
        
        print(f"\n\n✅ Streaming completed!")
        print(f"📊 Streaming Stats:")
        print(f"   • Total chunks: {chunk_count}")
        print(f"   • Total time: {end_time - start_time:.2f} seconds")
        print(f"   • Response length: {len(full_response)} characters")
        print(f"   • Average chunk size: {len(full_response) / chunk_count if chunk_count > 0 else 0:.1f} chars")
        
        print(f"\n💰 Cost tracking: Automatically calculated for streaming operations")
        print(f"🏷️  Governance: Attributed to 'streaming-team' for real-time applications")
        
        return True
        
    except Exception as e:
        print(f"❌ Streaming example error: {e}")
        return False

def function_calling_example():
    """Demonstrate function calling with detailed cost and usage tracking."""
    print("\n\n🔧 Function Calling with Usage Monitoring")
    print("-" * 50)
    
    try:
        from genops.providers.openai import instrument_openai
        
        client = instrument_openai()
        
        # Define available functions/tools
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get current weather for a specific location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "City and country, e.g. 'San Francisco, CA'"
                            },
                            "unit": {
                                "type": "string", 
                                "enum": ["celsius", "fahrenheit"],
                                "description": "Temperature unit"
                            }
                        },
                        "required": ["location"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "calculate_tip",
                    "description": "Calculate tip amount for a restaurant bill",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "bill_amount": {
                                "type": "number",
                                "description": "The total bill amount"
                            },
                            "tip_percentage": {
                                "type": "number",
                                "description": "Tip percentage (default 18%)",
                                "default": 18
                            }
                        },
                        "required": ["bill_amount"]
                    }
                }
            }
        ]
        
        # Test queries that should trigger function calls
        test_queries = [
            "What's the weather like in New York?",
            "Calculate a 20% tip on a $125 restaurant bill",
            "I need weather for London, UK in celsius"
        ]
        
        print(f"🎯 Available functions: {len(tools)}")
        for tool in tools:
            print(f"   • {tool['function']['name']}: {tool['function']['description']}")
        
        total_function_calls = 0
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n🔄 Query {i}: {query}")
            
            response = client.chat_completions_create(
                model="gpt-4-turbo",
                messages=[{"role": "user", "content": query}],
                tools=tools,
                tool_choice="auto",
                
                # Function calling governance tracking
                team="function-calling-team",
                project="tool-usage-analysis",
                customer_id=f"function-demo-{i}",
                query_index=i,
                available_functions=len(tools),
                feature="function_calling"
            )
            
            message = response.choices[0].message
            
            # Handle function calls
            if message.tool_calls:
                print(f"🔧 Function calls detected: {len(message.tool_calls)}")
                total_function_calls += len(message.tool_calls)
                
                for tool_call in message.tool_calls:
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)
                    
                    print(f"   📞 Calling: {function_name}")
                    print(f"   📋 Arguments: {function_args}")
                    
                    # Simulate function execution
                    if function_name == "get_weather":
                        result = f"Weather in {function_args.get('location', 'Unknown')}: 72°F, Sunny"
                    elif function_name == "calculate_tip":
                        bill = function_args.get("bill_amount", 0)
                        tip_pct = function_args.get("tip_percentage", 18)
                        tip_amount = bill * (tip_pct / 100)
                        result = f"Tip: ${tip_amount:.2f} ({tip_pct}% of ${bill})"
                    else:
                        result = f"Function {function_name} executed successfully"
                    
                    print(f"   ✅ Result: {result}")
                    
                    # In a real application, you would send the function result back
                    # to the model for a follow-up response
            else:
                print(f"   💬 Direct response: {message.content[:100]}...")
        
        print(f"\n📊 Function Calling Summary:")
        print(f"   • Total queries: {len(test_queries)}")
        print(f"   • Total function calls: {total_function_calls}")
        print(f"   • Available functions: {len(tools)}")
        print(f"   • Function call rate: {total_function_calls / len(test_queries):.1f} calls/query")
        
        return True
        
    except Exception as e:
        print(f"❌ Function calling example error: {e}")
        return False

def embeddings_example():
    """Demonstrate embeddings generation with cost analysis."""
    print("\n\n🔢 Embeddings Generation with Cost Analysis")
    print("-" * 50)
    
    try:
        from genops.providers.openai import instrument_openai
        
        client = instrument_openai()
        
        # Sample texts for embedding
        sample_texts = [
            "Machine learning is a subset of artificial intelligence.",
            "Python is a popular programming language for data science.",
            "OpenAI develops large language models like GPT-4.",
            "Vector databases store and search high-dimensional data.",
            "Natural language processing enables computers to understand text."
        ]
        
        print(f"📝 Generating embeddings for {len(sample_texts)} texts...")
        
        # Generate embeddings with cost tracking
        embeddings_response = client.embeddings_create(
            model="text-embedding-3-small",  # Cost-effective embedding model
            input=sample_texts,
            
            # Embeddings governance tracking
            team="embeddings-team",
            project="vector-analysis",
            customer_id="embeddings-demo",
            operation_type="batch_embedding",
            text_count=len(sample_texts),
            embedding_model="text-embedding-3-small"
        )
        
        embeddings_data = embeddings_response.data
        
        print(f"✅ Embeddings generated successfully!")
        print(f"📊 Embedding Stats:")
        print(f"   • Number of embeddings: {len(embeddings_data)}")
        print(f"   • Embedding dimensions: {len(embeddings_data[0].embedding)}")
        print(f"   • Total tokens: {embeddings_response.usage.total_tokens}")
        
        # Calculate embedding costs (text-embedding-3-small pricing)
        embedding_cost = (embeddings_response.usage.total_tokens / 1000) * 0.00002  # $0.00002 per 1K tokens
        print(f"   • Estimated cost: ${embedding_cost:.6f}")
        
        # Demonstrate simple similarity calculation
        print(f"\n🔍 Sample similarity analysis:")
        
        # Simple cosine similarity between first two embeddings
        emb1 = embeddings_data[0].embedding
        emb2 = embeddings_data[1].embedding
        
        dot_product = sum(a * b for a, b in zip(emb1, emb2))
        magnitude1 = sum(a * a for a in emb1) ** 0.5
        magnitude2 = sum(b * b for b in emb2) ** 0.5
        similarity = dot_product / (magnitude1 * magnitude2)
        
        print(f"   • Text 1: '{sample_texts[0][:50]}...'")
        print(f"   • Text 2: '{sample_texts[1][:50]}...'")
        print(f"   • Cosine similarity: {similarity:.4f}")
        
        # Cost analysis for embedding operations
        print(f"\n💰 Embedding Cost Analysis:")
        print(f"   • Cost per text: ${embedding_cost / len(sample_texts):.6f}")
        print(f"   • Cost per 1K tokens: ${0.00002:.6f}")
        print(f"   • Projected cost for 10K texts: ${(embedding_cost / len(sample_texts)) * 10000:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Embeddings example error: {e}")
        return False

def vision_api_example():
    """Demonstrate GPT-4 Vision with image analysis cost tracking."""
    print("\n\n👁️  Vision API with Image Analysis Tracking")
    print("-" * 50)
    
    try:
        from genops.providers.openai import instrument_openai
        
        client = instrument_openai()
        
        # Note: For this demo, we'll use a placeholder image URL
        # In practice, you would use actual image URLs or base64 encoded images
        sample_image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
        
        print(f"🖼️  Analyzing image with GPT-4 Vision...")
        print(f"📷 Image URL: {sample_image_url[:60]}...")
        
        # Vision API call with cost tracking
        response = client.chat_completions_create(
            model="gpt-4-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "What do you see in this image? Describe the scene, colors, and any notable features."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": sample_image_url,
                                "detail": "auto"  # Can be "low", "high", or "auto"
                            }
                        }
                    ]
                }
            ],
            max_tokens=300,
            
            # Vision API governance tracking
            team="vision-team",
            project="image-analysis",
            customer_id="vision-demo",
            has_image=True,
            image_detail_level="auto",
            feature="image_description"
        )
        
        print(f"✅ Vision analysis completed!")
        print(f"📝 Analysis result:")
        print(f"   {response.choices[0].message.content}")
        
        print(f"\n📊 Vision API Stats:")
        print(f"   • Input tokens: {response.usage.prompt_tokens}")
        print(f"   • Output tokens: {response.usage.completion_tokens}")
        print(f"   • Total tokens: {response.usage.total_tokens}")
        
        # Vision API cost calculation (simplified)
        # GPT-4 Vision has different pricing for image processing
        vision_cost = (response.usage.prompt_tokens / 1000) * 0.01 + (response.usage.completion_tokens / 1000) * 0.03
        print(f"   • Estimated cost: ${vision_cost:.4f}")
        
        print(f"\n💡 Vision API Cost Factors:")
        print(f"   • Image detail level affects token count and cost")
        print(f"   • Higher detail = more tokens = higher cost")
        print(f"   • Image dimensions and complexity impact processing")
        
        return True
        
    except Exception as e:
        print(f"❌ Vision API example error: {e}")
        print("💡 Vision API requires specific model access and may have usage restrictions")
        return False

def batch_operations_optimization():
    """Demonstrate optimized batch operations with cost efficiency."""
    print("\n\n📦 Batch Operations Optimization")
    print("-" * 50)
    
    try:
        from genops.providers.openai import instrument_openai
        from genops import track
        
        client = instrument_openai()
        
        # Sample batch tasks
        batch_tasks = [
            "Summarize: AI is transforming healthcare through diagnostic tools and personalized medicine.",
            "Translate to Spanish: The weather is beautiful today and perfect for a walk in the park.",
            "Generate keywords for: E-commerce website selling sustainable fashion and eco-friendly clothing.",
            "Classify sentiment: I absolutely love this new product, it exceeded all my expectations!",
            "Extract entities: Apple Inc. announced new iPhone models at their event in Cupertino, California."
        ]
        
        print(f"🔄 Processing {len(batch_tasks)} tasks in optimized batch...")
        
        with track("batch_optimization", 
                   team="batch-team",
                   project="operation-efficiency",
                   customer_id="batch-demo") as span:
            
            batch_results = []
            total_tokens = 0
            total_cost = 0
            start_time = time.time()
            
            # Process batch with optimizations
            for i, task in enumerate(batch_tasks):
                # Use cost-effective model for batch operations
                response = client.chat_completions_create(
                    model="gpt-3.5-turbo",  # Cost-effective for batch
                    messages=[{"role": "user", "content": task}],
                    max_tokens=100,  # Shorter responses for efficiency
                    temperature=0.3,  # Lower temperature for consistency
                    
                    # Batch operation tracking
                    team="batch-team",
                    project="operation-efficiency", 
                    customer_id="batch-demo",
                    batch_id="optimization-demo-001",
                    task_index=i,
                    batch_size=len(batch_tasks),
                    optimization_strategy="cost_effective"
                )
                
                result = response.choices[0].message.content
                tokens = response.usage.total_tokens
                cost = (response.usage.prompt_tokens / 1000) * 0.0015 + (response.usage.completion_tokens / 1000) * 0.002
                
                batch_results.append({
                    "task": task[:50] + "..." if len(task) > 50 else task,
                    "result": result[:80] + "..." if len(result) > 80 else result,
                    "tokens": tokens,
                    "cost": cost
                })
                
                total_tokens += tokens
                total_cost += cost
                
                print(f"   ✅ Task {i+1}: {tokens} tokens, ${cost:.4f}")
                
                # Brief pause to avoid rate limits
                time.sleep(0.1)
            
            end_time = time.time()
            
            # Set batch-level metrics
            span.set_attribute("tasks_completed", len(batch_tasks))
            span.set_attribute("total_tokens", total_tokens)
            span.set_attribute("total_cost", total_cost)
            span.set_attribute("batch_duration", end_time - start_time)
            
            print(f"\n📊 Batch Optimization Results:")
            print(f"   • Tasks completed: {len(batch_tasks)}")
            print(f"   • Total processing time: {end_time - start_time:.2f} seconds")
            print(f"   • Total tokens: {total_tokens}")
            print(f"   • Total cost: ${total_cost:.4f}")
            print(f"   • Average cost per task: ${total_cost / len(batch_tasks):.4f}")
            print(f"   • Average tokens per task: {total_tokens / len(batch_tasks):.0f}")
            
            # Efficiency analysis
            print(f"\n💡 Optimization Benefits:")
            print(f"   • Used cost-effective GPT-3.5-Turbo for batch processing")
            print(f"   • Limited max_tokens to control costs")
            print(f"   • Consistent temperature for predictable results") 
            print(f"   • Batch attribution for unified cost tracking")
            
        return True
        
    except Exception as e:
        print(f"❌ Batch optimization error: {e}")
        return False

def main():
    """Run advanced OpenAI features demonstrations."""
    print("🚀 OpenAI Advanced Features with GenOps Telemetry")
    print("=" * 70)
    
    # Check prerequisites
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY environment variable not set")
        print("💡 Fix: export OPENAI_API_KEY='your_api_key_here'")
        return False
    
    success = True
    
    # Run advanced feature examples
    success &= streaming_responses_example()
    success &= function_calling_example()
    success &= embeddings_example()
    
    # Vision API is optional (may require special access)
    try:
        success &= vision_api_example()
    except Exception as e:
        print(f"ℹ️  Vision API skipped: {e}")
    
    success &= batch_operations_optimization()
    
    # Summary
    print("\n" + "=" * 70)
    if success:
        print("🎉 Advanced features demonstration completed!")
        
        print("\n🔧 Advanced Features Covered:")
        print("   ✅ Streaming responses with real-time cost tracking")
        print("   ✅ Function calling and tool usage monitoring")
        print("   ✅ Embeddings generation with batch cost analysis")
        print("   ✅ Vision API integration (GPT-4 Vision)")
        print("   ✅ Optimized batch operations for cost efficiency")
        
        print("\n💰 Cost Optimization Insights:")
        print("   • Streaming enables real-time user experience with full cost tracking")
        print("   • Function calling costs include both model inference and tool usage")
        print("   • Embeddings offer cost-effective semantic analysis capabilities")
        print("   • Batch operations achieve significant per-task cost savings")
        print("   • Vision API requires careful cost management due to complexity")
        
        print("\n🚀 Next Steps:")
        print("   • Run 'python production_patterns.py' for enterprise deployment")
        print("   • Explore governance scenarios for policy enforcement")
        print("   • Set up observability dashboard to visualize these metrics")
        
        return True
    else:
        print("❌ Some advanced features encountered issues.")
        print("💡 Check API access and model availability for specialized features")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)