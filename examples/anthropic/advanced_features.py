#!/usr/bin/env python3
"""
Anthropic Advanced Features Example

This example demonstrates advanced Anthropic Claude features with GenOps telemetry including
streaming responses, multi-turn conversations, document analysis, and system prompt optimization.

What you'll learn:
- Streaming responses with real-time cost tracking
- Multi-turn conversation management and cost attribution
- Document analysis and processing workflows
- System prompt optimization and A/B testing
- Long-form content generation with Claude

Usage:
    python advanced_features.py

Prerequisites:
    pip install genops-ai[anthropic]
    export ANTHROPIC_API_KEY="your_anthropic_key_here"
"""

import os
import sys
import time


def streaming_responses_example():
    """Demonstrate streaming responses with GenOps cost tracking."""
    print("🌊 Streaming Claude Responses with Cost Tracking")
    print("-" * 55)

    try:
        from genops.providers.anthropic import instrument_anthropic

        client = instrument_anthropic()

        print("🚀 Starting streaming Claude completion...")
        print("📝 Response (streaming): ", end="", flush=True)

        # Create streaming completion
        stream = client.messages_create(
            model="claude-3-5-sonnet-20241022",
            messages=[
                {
                    "role": "user",
                    "content": "Write a thoughtful analysis of how artificial intelligence is reshaping the future of work, considering both opportunities and challenges. Make it comprehensive and well-structured.",
                }
            ],
            max_tokens=600,
            temperature=0.7,
            stream=True,  # Enable streaming
            # Governance attributes for streaming operations
            team="streaming-team",
            project="real-time-content",
            customer_id="streaming-demo",
            feature="analysis-writing",
            streaming_enabled=True,
        )

        # Process streaming response
        full_response = ""
        chunk_count = 0
        start_time = time.time()

        for event in stream:
            if event.type == "content_block_delta":
                content = event.delta.text
                full_response += content
                print(content, end="", flush=True)
                chunk_count += 1

                # Brief pause for demonstration
                time.sleep(0.01)

        end_time = time.time()

        print("\n\n✅ Streaming completed!")
        print("📊 Streaming Stats:")
        print(f"   • Total chunks: {chunk_count}")
        print(f"   • Total time: {end_time - start_time:.2f} seconds")
        print(f"   • Response length: {len(full_response)} characters")
        print(
            f"   • Average chunk size: {len(full_response) / chunk_count if chunk_count > 0 else 0:.1f} chars"
        )
        print(
            f"   • Streaming rate: {len(full_response) / (end_time - start_time):.0f} chars/second"
        )

        print(
            "\n💰 Cost tracking: Automatically calculated for streaming Claude operations"
        )
        print(
            "🏷️  Governance: Attributed to 'streaming-team' for real-time applications"
        )

        return True

    except Exception as e:
        print(f"❌ Streaming example error: {e}")
        return False


def multi_turn_conversation_example():
    """Demonstrate multi-turn conversation management with detailed cost tracking."""
    print("\n\n💬 Multi-Turn Conversation Management")
    print("-" * 50)

    try:
        from genops import track
        from genops.providers.anthropic import instrument_anthropic

        client = instrument_anthropic()

        # Start a conversation session
        conversation_history = []
        customer_id = "conversation-demo-user"

        with track(
            "multi_turn_conversation",
            team="conversation-team",
            project="dialogue-management",
            customer_id=customer_id,
        ) as span:
            conversation_turns = [
                {
                    "user": "I'm interested in starting a small business. What should I consider first?",
                    "context": "initial_inquiry",
                },
                {
                    "user": "That's helpful. I'm thinking about a sustainable fashion brand. What are the unique challenges?",
                    "context": "specific_domain",
                },
                {
                    "user": "How much capital would I typically need to start?",
                    "context": "financial_planning",
                },
                {
                    "user": "What about online vs physical retail?",
                    "context": "business_model",
                },
            ]

            total_conversation_cost = 0
            turn_costs = []

            print("🗣️  Multi-turn business consultation conversation:")

            for turn_num, turn in enumerate(conversation_turns, 1):
                print(f"\n--- Turn {turn_num} ---")
                print(f"👤 User: {turn['user']}")

                # Add user message to history
                conversation_history.append({"role": "user", "content": turn["user"]})

                # Claude response with conversation context
                response = client.messages_create(
                    model="claude-3-5-sonnet-20241022",
                    messages=conversation_history,
                    max_tokens=300,
                    temperature=0.7,
                    system="You are an experienced business consultant. Provide practical, actionable advice based on the conversation context.",
                    # Turn-specific governance tracking
                    team="conversation-team",
                    project="dialogue-management",
                    customer_id=customer_id,
                    conversation_turn=turn_num,
                    conversation_context=turn["context"],
                    total_turns_so_far=turn_num,
                    conversation_history_length=len(conversation_history),
                )

                assistant_response = response.content[0].text
                print(f"🤖 Claude: {assistant_response}")

                # Add Claude's response to history
                conversation_history.append(
                    {"role": "assistant", "content": assistant_response}
                )

                # Calculate turn cost
                turn_cost = (
                    response.usage.input_tokens / 1000000 * 3.00
                    + response.usage.output_tokens / 1000000 * 15.00
                )
                total_conversation_cost += turn_cost
                turn_costs.append(turn_cost)

                print(
                    f"💰 Turn cost: ${turn_cost:.6f} ({response.usage.input_tokens + response.usage.output_tokens} tokens)"
                )

                # Brief pause between turns
                time.sleep(0.5)

            # Set conversation-level metrics
            span.set_attribute("total_turns", len(conversation_turns))
            span.set_attribute("total_cost", total_conversation_cost)
            span.set_attribute(
                "average_cost_per_turn",
                total_conversation_cost / len(conversation_turns),
            )
            span.set_attribute("conversation_topic", "business_consultation")

            print("\n📊 Conversation Summary:")
            print(f"   • Total turns: {len(conversation_turns)}")
            print(f"   • Total conversation cost: ${total_conversation_cost:.6f}")
            print(
                f"   • Average cost per turn: ${total_conversation_cost / len(conversation_turns):.6f}"
            )
            print(f"   • Final context length: {len(conversation_history)} messages")

            print("\n💡 Multi-turn Benefits:")
            print("   • Context preservation across conversation")
            print("   • Per-turn cost attribution and tracking")
            print("   • Conversation flow optimization")
            print("   • Customer journey cost analysis")

        return True

    except Exception as e:
        print(f"❌ Multi-turn conversation error: {e}")
        return False


def document_analysis_workflow():
    """Demonstrate document analysis and processing with Claude."""
    print("\n\n📄 Document Analysis Workflow")
    print("-" * 40)

    try:
        from genops import track
        from genops.providers.anthropic import instrument_anthropic

        client = instrument_anthropic()

        # Sample documents for analysis
        sample_documents = [
            {
                "type": "contract",
                "content": "Software License Agreement: This agreement grants the licensee non-exclusive rights to use the software for internal business purposes only. The license term is 24 months with automatic renewal unless terminated by either party with 30 days notice. Licensee agrees to pay $10,000 annually and comply with all security requirements including data encryption and access controls.",
                "analysis_focus": "key terms and obligations",
            },
            {
                "type": "policy",
                "content": "Remote Work Policy: Employees may work remotely up to 3 days per week with supervisor approval. Remote workers must maintain regular business hours, participate in scheduled meetings, and ensure secure internet connection. Company equipment must be returned within 5 business days of employment termination.",
                "analysis_focus": "compliance requirements",
            },
            {
                "type": "financial_report",
                "content": "Q3 Financial Summary: Revenue increased 18% to $2.4M compared to Q2. Operating expenses rose 12% primarily due to new hires and marketing campaigns. Net profit margin improved to 15.2%. Customer acquisition cost decreased by 8% while customer lifetime value increased by 22%.",
                "analysis_focus": "performance trends and insights",
            },
        ]

        with track(
            "document_analysis_workflow",
            team="document-processing-team",
            project="ai-document-analyzer",
            customer_id="doc-analysis-demo",
        ) as span:
            analysis_results = []
            total_analysis_cost = 0

            print("📋 Processing documents with Claude analysis:")

            for i, doc in enumerate(sample_documents, 1):
                print(f"\n🔍 Document {i}: {doc['type'].title()}")
                print(f"   Content: {doc['content'][:80]}...")
                print(f"   Focus: {doc['analysis_focus']}")

                # Claude document analysis
                response = client.messages_create(
                    model="claude-3-5-sonnet-20241022",  # Best for analysis
                    messages=[
                        {
                            "role": "user",
                            "content": f"Analyze this {doc['type']} document focusing on {doc['analysis_focus']}:\n\n{doc['content']}",
                        }
                    ],
                    max_tokens=400,
                    temperature=0.3,  # Lower temperature for analytical accuracy
                    system="You are an expert document analyst. Provide structured, accurate analysis with specific details and actionable insights.",
                    # Document analysis governance
                    team="document-processing-team",
                    project="ai-document-analyzer",
                    customer_id="doc-analysis-demo",
                    document_type=doc["type"],
                    document_index=i,
                    analysis_focus=doc["analysis_focus"],
                    requires_accuracy="high",
                )

                analysis = response.content[0].text
                analysis_cost = (
                    response.usage.input_tokens / 1000000 * 3.00
                    + response.usage.output_tokens / 1000000 * 15.00
                )

                analysis_results.append(
                    {
                        "document_type": doc["type"],
                        "analysis": analysis,
                        "cost": analysis_cost,
                        "tokens": response.usage.input_tokens
                        + response.usage.output_tokens,
                    }
                )

                total_analysis_cost += analysis_cost

                print(f"   📊 Analysis: {analysis[:100]}...")
                print(f"   💰 Cost: ${analysis_cost:.6f}")

            # Set workflow-level metrics
            span.set_attribute("documents_analyzed", len(sample_documents))
            span.set_attribute("total_analysis_cost", total_analysis_cost)
            span.set_attribute(
                "average_cost_per_document", total_analysis_cost / len(sample_documents)
            )
            span.set_attribute(
                "document_types", [doc["type"] for doc in sample_documents]
            )

            print("\n📊 Document Analysis Summary:")
            print(f"   • Documents processed: {len(sample_documents)}")
            print(f"   • Total analysis cost: ${total_analysis_cost:.6f}")
            print(
                f"   • Average cost per document: ${total_analysis_cost / len(sample_documents):.6f}"
            )
            print(
                f"   • Document types: {', '.join({doc['type'] for doc in sample_documents})}"
            )

            print("\n💡 Document Analysis Benefits:")
            print("   • Structured analysis with consistent format")
            print("   • Cost tracking per document type")
            print("   • Scalable processing for large document sets")
            print("   • Specialized analysis focus per document")

        return True

    except Exception as e:
        print(f"❌ Document analysis workflow error: {e}")
        return False


def system_prompt_optimization():
    """Demonstrate system prompt optimization and A/B testing."""
    print("\n\n🎯 System Prompt Optimization and A/B Testing")
    print("-" * 55)

    try:
        from genops.providers.anthropic import instrument_anthropic

        client = instrument_anthropic()

        # Test different system prompts for the same task
        test_prompts = [
            {
                "name": "Basic Prompt",
                "system": "You are a helpful assistant.",
                "expected_style": "general_helpful",
            },
            {
                "name": "Expert Persona",
                "system": "You are a senior software engineer with 15 years of experience in distributed systems and cloud architecture. Provide technical insights with practical examples.",
                "expected_style": "technical_expert",
            },
            {
                "name": "Structured Response",
                "system": "You are a technical consultant. Always structure your responses with: 1) Brief Summary, 2) Key Points, 3) Recommendations, 4) Next Steps. Be concise and actionable.",
                "expected_style": "structured_consultant",
            },
            {
                "name": "Educational Style",
                "system": "You are a patient teacher explaining complex topics. Use analogies, examples, and break down concepts step-by-step. Ensure clarity for someone learning the subject.",
                "expected_style": "educational_teacher",
            },
        ]

        test_query = "How should I design a microservices architecture for a high-traffic e-commerce platform?"

        print(f"📝 Test query: {test_query}")
        print("\n🧪 System Prompt A/B Testing Results:")
        print(
            f"{'Prompt Type':<20} {'Cost':<12} {'Tokens':<10} {'Response Quality':<15} {'Style Match'}"
        )
        print("-" * 85)

        prompt_results = []

        for prompt in test_prompts:
            try:
                response = client.messages_create(
                    model="claude-3-5-sonnet-20241022",
                    messages=[{"role": "user", "content": test_query}],
                    max_tokens=400,
                    temperature=0.7,
                    system=prompt["system"],
                    # System prompt optimization tracking
                    team="optimization-team",
                    project="system-prompt-testing",
                    customer_id="prompt-optimization-demo",
                    prompt_type=prompt["name"],
                    expected_style=prompt["expected_style"],
                    ab_test_variant=prompt["name"],
                )

                cost = (
                    response.usage.input_tokens / 1000000 * 3.00
                    + response.usage.output_tokens / 1000000 * 15.00
                )
                tokens = response.usage.input_tokens + response.usage.output_tokens
                response_text = response.content[0].text

                # Simple quality assessment
                quality_indicators = [
                    len(response_text.split()) > 100,  # Adequate length
                    "microservices" in response_text.lower(),  # Topic relevance
                    any(
                        word in response_text.lower()
                        for word in ["architecture", "design", "scalability"]
                    ),  # Key concepts
                    ":" in response_text
                    or "•" in response_text
                    or "\n" in response_text,  # Structure
                ]
                quality_score = sum(quality_indicators)
                quality_rating = "⭐" * quality_score

                # Style matching assessment
                style_matches = {
                    "general_helpful": "helpful" in response_text.lower()
                    or "here" in response_text.lower(),
                    "technical_expert": any(
                        word in response_text.lower()
                        for word in [
                            "distributed",
                            "cloud",
                            "scalability",
                            "performance",
                        ]
                    ),
                    "structured_consultant": any(
                        pattern in response_text
                        for pattern in ["1.", "2.", "Summary", "Key", "Recommendations"]
                    ),
                    "educational_teacher": any(
                        word in response_text.lower()
                        for word in ["example", "think", "consider", "like", "such as"]
                    ),
                }
                style_match = (
                    "✅" if style_matches.get(prompt["expected_style"], False) else "❌"
                )

                prompt_results.append(
                    {
                        "name": prompt["name"],
                        "cost": cost,
                        "tokens": tokens,
                        "quality": quality_score,
                        "style_match": style_match,
                        "response": response_text,
                    }
                )

                print(
                    f"{prompt['name']:<20} ${cost:<11.6f} {tokens:<10} {quality_rating:<15} {style_match}"
                )

            except Exception as e:
                print(f"{prompt['name']:<20} Error: {str(e)[:30]}...")

        # Analysis and recommendations
        if prompt_results:
            best_quality = max(prompt_results, key=lambda x: x["quality"])
            most_cost_effective = min(prompt_results, key=lambda x: x["cost"])

            print("\n🏆 Optimization Results:")
            print(
                f"   • Best quality: {best_quality['name']} ({best_quality['quality']} quality indicators)"
            )
            print(
                f"   • Most cost-effective: {most_cost_effective['name']} (${most_cost_effective['cost']:.6f})"
            )
            print(
                f"   • Style matching: {sum(1 for r in prompt_results if r['style_match'] == '✅')}/{len(prompt_results)} prompts matched expected style"
            )

            print("\n💡 System Prompt Optimization Insights:")
            print("   • Specific persona prompts improve response relevance")
            print("   • Structured prompts help with consistent formatting")
            print("   • Educational prompts increase explanation quality")
            print(
                f"   • Cost variation: {max(r['cost'] for r in prompt_results) / min(r['cost'] for r in prompt_results):.1f}x range"
            )

        return True

    except Exception as e:
        print(f"❌ System prompt optimization error: {e}")
        return False


def long_form_content_generation():
    """Demonstrate long-form content generation with cost tracking."""
    print("\n\n📝 Long-Form Content Generation")
    print("-" * 40)

    try:
        from genops import track
        from genops.providers.anthropic import instrument_anthropic

        client = instrument_anthropic()

        content_requests = [
            {
                "type": "technical_article",
                "topic": "The Evolution of Cloud Computing: From Mainframes to Serverless",
                "target_length": "comprehensive analysis",
                "audience": "technical professionals",
            },
            {
                "type": "business_report",
                "topic": "Market Analysis: Sustainable Technology Trends in 2024",
                "target_length": "executive summary with details",
                "audience": "business executives",
            },
        ]

        with track(
            "long_form_content_generation",
            team="content-team",
            project="ai-content-creation",
            customer_id="content-demo",
        ) as span:
            content_results = []
            total_content_cost = 0

            for i, request in enumerate(content_requests, 1):
                print(f"\n✍️  Content Request {i}: {request['type']}")
                print(f"   Topic: {request['topic']}")
                print(f"   Audience: {request['audience']}")

                # Generate long-form content with Claude
                response = client.messages_create(
                    model="claude-3-5-sonnet-20241022",  # Best for long-form content
                    messages=[
                        {
                            "role": "user",
                            "content": f"Write a {request['target_length']} {request['type']} about: {request['topic']}. Target audience: {request['audience']}. Make it engaging, informative, and well-structured with clear sections.",
                        }
                    ],
                    max_tokens=2000,  # Longer content
                    temperature=0.7,
                    system="You are an expert writer who creates engaging, well-researched content. Structure your writing with clear headings, compelling introductions, and actionable insights.",
                    # Long-form content governance
                    team="content-team",
                    project="ai-content-creation",
                    customer_id="content-demo",
                    content_type=request["type"],
                    content_topic=request["topic"],
                    target_audience=request["audience"],
                    content_length="long_form",
                )

                content = response.content[0].text
                content_cost = (
                    response.usage.input_tokens / 1000000 * 3.00
                    + response.usage.output_tokens / 1000000 * 15.00
                )

                content_results.append(
                    {
                        "type": request["type"],
                        "content": content,
                        "cost": content_cost,
                        "tokens": response.usage.input_tokens
                        + response.usage.output_tokens,
                        "word_count": len(content.split()),
                    }
                )

                total_content_cost += content_cost

                print(
                    f"   📊 Generated: {len(content)} characters, {len(content.split())} words"
                )
                print(f"   💰 Cost: ${content_cost:.6f}")
                print(f"   📄 Preview: {content[:150]}...")

            # Set content generation metrics
            span.set_attribute("content_pieces_generated", len(content_requests))
            span.set_attribute("total_content_cost", total_content_cost)
            span.set_attribute(
                "average_cost_per_piece", total_content_cost / len(content_requests)
            )
            span.set_attribute(
                "total_word_count", sum(r["word_count"] for r in content_results)
            )

            print("\n📊 Content Generation Summary:")
            print(f"   • Content pieces generated: {len(content_requests)}")
            print(f"   • Total generation cost: ${total_content_cost:.6f}")
            print(
                f"   • Total word count: {sum(r['word_count'] for r in content_results):,}"
            )
            print(
                f"   • Average cost per piece: ${total_content_cost / len(content_requests):.6f}"
            )
            print(
                f"   • Cost per 1000 words: ${total_content_cost / (sum(r['word_count'] for r in content_results) / 1000):.6f}"
            )

            print("\n💡 Long-Form Content Benefits:")
            print("   • High-quality, structured content generation")
            print("   • Cost tracking per content type and audience")
            print("   • Scalable content production pipeline")
            print("   • Audience-specific optimization")

        return True

    except Exception as e:
        print(f"❌ Long-form content generation error: {e}")
        return False


def main():
    """Run advanced Anthropic features demonstrations."""
    print("🚀 Anthropic Advanced Features with GenOps Telemetry")
    print("=" * 70)

    # Check prerequisites
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("❌ ANTHROPIC_API_KEY environment variable not set")
        print("💡 Fix: export ANTHROPIC_API_KEY='your_api_key_here'")
        return False

    success = True

    # Run advanced feature examples
    success &= streaming_responses_example()
    success &= multi_turn_conversation_example()
    success &= document_analysis_workflow()
    success &= system_prompt_optimization()
    success &= long_form_content_generation()

    # Summary
    print("\n" + "=" * 70)
    if success:
        print("🎉 Advanced Claude features demonstration completed!")

        print("\n🔧 Advanced Features Covered:")
        print("   ✅ Streaming responses with real-time cost tracking")
        print("   ✅ Multi-turn conversation management and context preservation")
        print("   ✅ Document analysis workflows for various document types")
        print("   ✅ System prompt optimization and A/B testing")
        print("   ✅ Long-form content generation with detailed cost analysis")

        print("\n💰 Cost Optimization Insights:")
        print(
            "   • Streaming enables real-time user experience with full cost tracking"
        )
        print("   • Multi-turn conversations require careful context cost management")
        print("   • Document analysis benefits from Claude's superior reasoning")
        print("   • System prompt optimization can improve cost-effectiveness")
        print("   • Long-form content generation scales efficiently with Claude")

        print("\n🚀 Next Steps:")
        print("   • Run 'python production_patterns.py' for enterprise deployment")
        print("   • Explore governance scenarios for Claude policy enforcement")
        print("   • Set up observability dashboard to visualize these metrics")

        return True
    else:
        print("❌ Some advanced features encountered issues.")
        print("💡 Check API access and network connectivity")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
