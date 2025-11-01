#!/usr/bin/env python3
"""
Anthropic Production Patterns Example

This example demonstrates enterprise-ready patterns for deploying Anthropic Claude applications
with GenOps governance telemetry in production environments.

What you'll learn:
- Context manager patterns for complex Claude workflows
- Policy enforcement and governance automation
- Error handling and resilience patterns for Claude
- Performance optimization and scaling
- Enterprise monitoring and alerting for Claude operations

Usage:
    python production_patterns.py
    
Prerequisites:
    pip install genops-ai[anthropic]
    export ANTHROPIC_API_KEY="your_anthropic_key_here"
"""

import os
import sys
import time
import asyncio
from typing import Dict, List, Optional, Any
from contextlib import contextmanager
from dataclasses import dataclass

@dataclass
class ClaudeWorkflowResult:
    """Result from a production Claude workflow with full telemetry."""
    workflow_id: str
    success: bool
    total_cost: float
    operations_count: int
    duration: float
    error: Optional[str] = None
    metadata: Dict[str, Any] = None

@contextmanager
def claude_production_workflow_context(workflow_name: str, customer_id: str, **kwargs):
    """Production-ready context manager for complex Claude AI workflows."""
    from genops import track
    
    workflow_id = f"{workflow_name}_claude_{customer_id}_{int(time.time())}"
    start_time = time.time()
    
    print(f"🚀 Starting Claude workflow: {workflow_name}")
    print(f"   Workflow ID: {workflow_id}")
    print(f"   Customer: {customer_id}")
    
    with track(workflow_name, 
               workflow_id=workflow_id,
               customer_id=customer_id,
               ai_provider="anthropic",
               **kwargs) as span:
        try:
            yield span, workflow_id
            
            duration = time.time() - start_time
            span.set_attribute("workflow_success", True)
            span.set_attribute("workflow_duration", duration)
            span.set_attribute("claude_workflow", True)
            
            print(f"✅ Claude workflow completed: {workflow_name}")
            print(f"   Duration: {duration:.2f} seconds")
            
        except Exception as e:
            duration = time.time() - start_time
            span.set_attribute("workflow_success", False)
            span.set_attribute("workflow_error", str(e))
            span.set_attribute("workflow_duration", duration)
            
            print(f"❌ Claude workflow failed: {workflow_name}")
            print(f"   Error: {e}")
            print(f"   Duration: {duration:.2f} seconds")
            raise

def legal_document_review_workflow():
    """Enterprise legal document review workflow with Claude."""
    print("⚖️  Enterprise Legal Document Review Workflow")
    print("-" * 55)
    
    try:
        from genops.providers.anthropic import instrument_anthropic
        
        client = instrument_anthropic()
        
        # Simulate legal document for review
        legal_document = {
            "document_id": "CONTRACT-2024-001",
            "customer_id": "enterprise-legal-001", 
            "document_type": "software_license",
            "priority": "high",
            "content": """SOFTWARE LICENSE AGREEMENT
            
This Software License Agreement (Agreement) is entered into between TechCorp Inc. (Licensor) and Client Company (Licensee). 

GRANT OF LICENSE: Licensor grants Licensee a non-exclusive, non-transferable license to use the Software solely for Licensee's internal business operations.

TERM: This license is effective for 36 months from the Effective Date and will automatically renew for successive 12-month periods unless terminated by either party with 60 days written notice.

FEES: Licensee shall pay annual license fees of $50,000, due within 30 days of each anniversary date. Late payments incur 1.5% monthly interest charges.

RESTRICTIONS: Licensee may not modify, reverse engineer, sublicense, or distribute the Software. Maximum of 100 concurrent users allowed.

TERMINATION: Either party may terminate for material breach with 30 days cure period. Upon termination, Licensee must destroy all copies and return confidential information.

LIABILITY: Licensor's total liability shall not exceed the annual license fee. No liability for consequential or indirect damages.

GOVERNING LAW: This Agreement shall be governed by Delaware state law.""",
            "review_requirements": ["key_terms", "obligations", "risks", "compliance"]
        }
        
        with claude_production_workflow_context(
            "legal_document_review",
            legal_document["customer_id"],
            team="legal-team",
            project="contract-analysis",
            environment="production",
            document_id=legal_document["document_id"],
            document_type=legal_document["document_type"],
            priority=legal_document["priority"]
        ) as (span, workflow_id):
            
            total_cost = 0
            review_operations = []
            
            # Step 1: Initial document classification and risk assessment
            print("🔍 Step 1: Document Classification and Risk Assessment")
            classification_response = client.messages_create(
                model="claude-3-5-sonnet-20241022",  # Best for legal analysis
                messages=[
                    {"role": "user", "content": f"Classify this legal document and provide an initial risk assessment:\n\n{legal_document['content']}"}
                ],
                max_tokens=300,
                temperature=0.3,  # Lower temperature for accuracy
                system="You are an expert legal analyst. Provide structured analysis focusing on document classification, key risk factors, and initial assessment.",
                
                # Step-specific governance
                team="legal-team",
                project="contract-analysis",
                customer_id=legal_document["customer_id"],
                workflow_id=workflow_id,
                step="classification_risk_assessment",
                document_id=legal_document["document_id"],
                requires_accuracy="critical"
            )
            
            classification = classification_response.content[0].text
            classification_cost = (classification_response.usage.input_tokens / 1000000 * 3.00 + 
                                  classification_response.usage.output_tokens / 1000000 * 15.00)
            total_cost += classification_cost
            review_operations.append(("Document Classification", classification_cost))
            
            print(f"   Result: {classification[:120]}...")
            print(f"   Cost: ${classification_cost:.6f}")
            
            # Step 2: Detailed terms and obligations analysis
            print("\n📋 Step 2: Terms and Obligations Analysis")
            terms_response = client.messages_create(
                model="claude-3-5-sonnet-20241022",
                messages=[
                    {"role": "user", "content": f"Extract and analyze all key terms, obligations, and conditions from this contract:\n\n{legal_document['content']}"}
                ],
                max_tokens=500,
                temperature=0.2,  # Very low for precise extraction
                system="You are a contract attorney specializing in software licensing. Extract specific terms, obligations, dates, amounts, and conditions with precise details.",
                
                # Enhanced governance for critical analysis
                team="legal-team",
                project="contract-analysis", 
                customer_id=legal_document["customer_id"],
                workflow_id=workflow_id,
                step="terms_obligations_analysis",
                document_id=legal_document["document_id"],
                analysis_type="detailed_extraction",
                legal_specialization="software_licensing"
            )
            
            terms_analysis = terms_response.content[0].text
            terms_cost = (terms_response.usage.input_tokens / 1000000 * 3.00 + 
                         terms_response.usage.output_tokens / 1000000 * 15.00)
            total_cost += terms_cost
            review_operations.append(("Terms Analysis", terms_cost))
            
            print(f"   Analysis: {terms_analysis[:150]}...")
            print(f"   Cost: ${terms_cost:.6f}")
            
            # Step 3: Risk and compliance assessment
            print("\n⚠️  Step 3: Risk and Compliance Assessment")
            risk_response = client.messages_create(
                model="claude-3-5-sonnet-20241022",
                messages=[
                    {"role": "user", "content": f"Identify potential legal risks, compliance issues, and areas of concern in this contract:\n\n{legal_document['content']}\n\nPrevious analysis:\n{terms_analysis}"}
                ],
                max_tokens=400,
                temperature=0.3,
                system="You are a senior legal counsel specializing in risk assessment. Identify potential legal exposures, compliance risks, unfavorable terms, and recommend protective measures.",
                
                # Risk assessment governance
                team="legal-team",
                project="contract-analysis",
                customer_id=legal_document["customer_id"], 
                workflow_id=workflow_id,
                step="risk_compliance_assessment",
                document_id=legal_document["document_id"],
                risk_analysis=True,
                compliance_check=True
            )
            
            risk_assessment = risk_response.content[0].text
            risk_cost = (risk_response.usage.input_tokens / 1000000 * 3.00 + 
                        risk_response.usage.output_tokens / 1000000 * 15.00)
            total_cost += risk_cost
            review_operations.append(("Risk Assessment", risk_cost))
            
            print(f"   Assessment: {risk_assessment[:150]}...")
            print(f"   Cost: ${risk_cost:.6f}")
            
            # Step 4: Final recommendations and action items
            print("\n💼 Step 4: Recommendations and Action Items")
            recommendations_response = client.messages_create(
                model="claude-3-5-sonnet-20241022",
                messages=[
                    {"role": "user", "content": f"Based on this contract analysis, provide specific recommendations and action items:\n\nContract: {legal_document['content'][:500]}...\n\nRisk Assessment: {risk_assessment}"}
                ],
                max_tokens=350,
                temperature=0.4,
                system="You are a legal advisor providing actionable recommendations. Focus on specific steps, negotiations points, protective measures, and decision guidance for the client.",
                
                # Final recommendations governance
                team="legal-team",
                project="contract-analysis",
                customer_id=legal_document["customer_id"],
                workflow_id=workflow_id,
                step="final_recommendations",
                document_id=legal_document["document_id"],
                deliverable_type="actionable_recommendations"
            )
            
            recommendations = recommendations_response.content[0].text
            recommendations_cost = (recommendations_response.usage.input_tokens / 1000000 * 3.00 + 
                                   recommendations_response.usage.output_tokens / 1000000 * 15.00)
            total_cost += recommendations_cost
            review_operations.append(("Recommendations", recommendations_cost))
            
            print(f"   Recommendations: {recommendations[:150]}...")
            print(f"   Cost: ${recommendations_cost:.6f}")
            
            # Set workflow-level metrics
            span.set_attribute("total_review_operations", len(review_operations))
            span.set_attribute("total_workflow_cost", total_cost)
            span.set_attribute("document_type", legal_document["document_type"])
            span.set_attribute("review_priority", legal_document["priority"])
            span.set_attribute("claude_model_used", "claude-3-5-sonnet-20241022")
            
            print(f"\n📊 Legal Review Workflow Summary:")
            print(f"   • Total review operations: {len(review_operations)}")
            print(f"   • Total workflow cost: ${total_cost:.6f}")
            print(f"   • Average cost per operation: ${total_cost / len(review_operations):.6f}")
            
            for operation, cost in review_operations:
                print(f"   • {operation}: ${cost:.6f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Legal document review workflow error: {e}")
        return False

def intelligent_content_pipeline():
    """Content generation pipeline with Claude-specific optimizations."""
    print("\n\n📝 Intelligent Claude Content Pipeline")
    print("-" * 50)
    
    try:
        from genops.providers.anthropic import instrument_anthropic
        
        client = instrument_anthropic()
        
        # Content generation requests optimized for Claude
        content_requests = [
            {
                "type": "executive_summary",
                "topic": "AI transformation strategy for traditional manufacturing",
                "target_audience": "C-suite executives",
                "complexity": "high",
                "claude_strength": "strategic_analysis"
            },
            {
                "type": "technical_whitepaper",
                "topic": "Implementation guide for sustainable supply chain management",
                "target_audience": "operations_managers",
                "complexity": "very_high", 
                "claude_strength": "detailed_reasoning"
            },
            {
                "type": "marketing_copy",
                "topic": "Product launch campaign for AI-powered analytics platform",
                "target_audience": "technology_buyers",
                "complexity": "medium",
                "claude_strength": "persuasive_writing"
            }
        ]
        
        with claude_production_workflow_context(
            "intelligent_content_pipeline",
            "content-enterprise-001",
            team="content-operations",
            project="ai-content-automation",
            environment="production"
        ) as (span, workflow_id):
            
            total_pipeline_cost = 0
            generated_content = []
            
            for i, request in enumerate(content_requests, 1):
                print(f"\n🎯 Content Request {i}: {request['type']} - {request['topic'][:50]}...")
                
                # Claude model selection based on complexity
                model_selection = {
                    "medium": "claude-3-5-haiku-20241022",     # Cost-effective
                    "high": "claude-3-5-sonnet-20241022",     # Balanced
                    "very_high": "claude-3-opus-20240229"     # Premium quality
                }
                
                selected_model = model_selection.get(request["complexity"], "claude-3-5-sonnet-20241022")
                
                # Policy enforcement check
                policy_check = enforce_claude_content_policy(request)
                if not policy_check["approved"]:
                    print(f"   ❌ Policy violation: {policy_check['reason']}")
                    continue
                
                # Content generation with Claude optimization
                content_result = generate_content_with_claude(
                    client, request, selected_model, workflow_id, i
                )
                
                if content_result:
                    generated_content.append(content_result)
                    total_pipeline_cost += content_result["cost"]
                    
                    print(f"   ✅ Generated: {content_result['word_count']} words")
                    print(f"   🤖 Model: {selected_model}")
                    print(f"   💰 Cost: ${content_result['cost']:.6f}")
                    print(f"   🎯 Claude strength: {request['claude_strength']}")
            
            # Pipeline summary
            span.set_attribute("content_requests_processed", len(content_requests))
            span.set_attribute("content_pieces_generated", len(generated_content))
            span.set_attribute("pipeline_total_cost", total_pipeline_cost)
            span.set_attribute("claude_pipeline_optimization", True)
            
            print(f"\n📊 Claude Content Pipeline Results:")
            print(f"   • Requests processed: {len(content_requests)}")
            print(f"   • Content pieces generated: {len(generated_content)}")
            print(f"   • Total pipeline cost: ${total_pipeline_cost:.6f}")
            print(f"   • Average cost per piece: ${total_pipeline_cost / max(len(generated_content), 1):.6f}")
            print(f"   • Total word count: {sum(c.get('word_count', 0) for c in generated_content):,}")
        
        return True
        
    except Exception as e:
        print(f"❌ Claude content pipeline error: {e}")
        return False

def enforce_claude_content_policy(request: Dict) -> Dict:
    """Enforce content policy for Claude operations."""
    # Claude-specific policy enforcement
    restricted_topics = ["controversial political", "medical diagnosis", "financial advice"]
    sensitive_audiences = ["minors", "healthcare_patients"]
    
    topic_lower = request["topic"].lower()
    audience_lower = request["target_audience"].lower()
    
    for restricted in restricted_topics:
        if restricted in topic_lower:
            return {
                "approved": False,
                "reason": f"Topic contains restricted content: {restricted}"
            }
    
    for sensitive in sensitive_audiences:
        if sensitive in audience_lower:
            return {
                "approved": False,
                "reason": f"Sensitive audience requires special handling: {sensitive}"
            }
    
    return {"approved": True, "reason": "Content approved for Claude processing"}

def generate_content_with_claude(client, request: Dict, model: str, workflow_id: str, request_index: int) -> Optional[Dict]:
    """Generate content with Claude-specific optimizations."""
    try:
        # Claude-optimized system prompts
        claude_system_prompts = {
            "executive_summary": "You are a senior business consultant writing for C-suite executives. Create compelling, strategic content with clear business value propositions and actionable insights.",
            "technical_whitepaper": "You are a technical expert and thought leader. Write authoritative, detailed content with practical implementation guidance and real-world examples.",
            "marketing_copy": "You are a persuasive marketing copywriter. Create engaging, benefit-focused content that resonates with your target audience and drives action."
        }
        
        system_prompt = claude_system_prompts.get(request["type"], "You are a professional writer creating high-quality content.")
        
        response = client.messages_create(
            model=model,
            messages=[
                {"role": "user", "content": f"Create a comprehensive {request['type']} about: {request['topic']}. Target audience: {request['target_audience']}. Make it engaging, well-structured, and valuable."}
            ],
            max_tokens=1500 if request["complexity"] == "very_high" else 1000,
            temperature=0.7,
            system=system_prompt,
            
            # Detailed Claude content governance
            team="content-operations",
            project="ai-content-automation", 
            workflow_id=workflow_id,
            content_type=request["type"],
            target_audience=request["target_audience"],
            complexity_level=request["complexity"],
            claude_strength=request["claude_strength"],
            request_index=request_index,
            model_selection_reason="complexity_optimized"
        )
        
        content = response.content[0].text
        
        # Calculate cost based on actual Claude model used
        if model == "claude-3-opus-20240229":
            cost = (response.usage.input_tokens / 1000000 * 15.00 + 
                   response.usage.output_tokens / 1000000 * 75.00)
        elif model == "claude-3-5-sonnet-20241022":
            cost = (response.usage.input_tokens / 1000000 * 3.00 + 
                   response.usage.output_tokens / 1000000 * 15.00)
        else:  # Haiku
            cost = (response.usage.input_tokens / 1000000 * 1.00 + 
                   response.usage.output_tokens / 1000000 * 5.00)
        
        return {
            "content": content,
            "cost": cost,
            "tokens": response.usage.input_tokens + response.usage.output_tokens,
            "model": model,
            "type": request["type"],
            "word_count": len(content.split())
        }
        
    except Exception as e:
        print(f"   ❌ Content generation failed: {e}")
        return None

def claude_resilience_and_monitoring():
    """Demonstrate production-grade resilience and monitoring for Claude."""
    print("\n\n🛡️  Claude Resilience and Monitoring Patterns")
    print("-" * 55)
    
    try:
        from genops.providers.anthropic import instrument_anthropic
        
        client = instrument_anthropic()
        
        # Test scenarios including Claude-specific considerations
        test_scenarios = [
            {
                "name": "Normal Claude Operation",
                "model": "claude-3-5-haiku-20241022",
                "prompt": "Explain the benefits of renewable energy in business.",
                "expected_success": True
            },
            {
                "name": "Long Context Test",
                "model": "claude-3-5-sonnet-20241022",
                "prompt": "Analyze this extensive document: " + "Sample content. " * 100,
                "expected_success": True
            },
            {
                "name": "Model Availability Test",
                "model": "claude-3-5-sonnet-20241022",
                "prompt": "This tests Claude model availability and response.",
                "expected_success": True
            }
        ]
        
        with claude_production_workflow_context(
            "claude_resilience_testing",
            "resilience-demo",
            team="sre-team",
            project="claude-reliability"
        ) as (span, workflow_id):
            
            results = []
            
            for scenario in test_scenarios:
                print(f"\n🧪 Testing: {scenario['name']}")
                
                try:
                    # Claude-specific retry logic
                    max_retries = 3
                    retry_delay = 2  # Longer delay for Claude
                    
                    for attempt in range(max_retries):
                        try:
                            start_time = time.time()
                            
                            response = client.messages_create(
                                model=scenario["model"],
                                messages=[{"role": "user", "content": scenario["prompt"]}],
                                max_tokens=300,
                                
                                # Resilience testing governance
                                team="sre-team",
                                project="claude-reliability",
                                workflow_id=workflow_id,
                                test_scenario=scenario["name"],
                                attempt_number=attempt + 1,
                                max_retries=max_retries,
                                claude_resilience_test=True
                            )
                            
                            duration = time.time() - start_time
                            
                            results.append({
                                "scenario": scenario["name"],
                                "success": True,
                                "attempt": attempt + 1,
                                "duration": duration,
                                "tokens": response.usage.input_tokens + response.usage.output_tokens,
                                "claude_model": scenario["model"]
                            })
                            
                            print(f"   ✅ Success on attempt {attempt + 1}")
                            print(f"   📊 Duration: {duration:.2f}s, Tokens: {response.usage.input_tokens + response.usage.output_tokens}")
                            print(f"   🤖 Claude response: {response.content[0].text[:80]}...")
                            break
                            
                        except Exception as e:
                            if attempt < max_retries - 1:
                                print(f"   ⚠️  Attempt {attempt + 1} failed: {e}")
                                print(f"   🔄 Retrying Claude request in {retry_delay}s...")
                                time.sleep(retry_delay)
                                retry_delay *= 1.5  # Gentle exponential backoff
                            else:
                                # Final failure
                                results.append({
                                    "scenario": scenario["name"],
                                    "success": False,
                                    "error": str(e),
                                    "attempts": max_retries,
                                    "claude_model": scenario["model"]
                                })
                                print(f"   ❌ Failed after {max_retries} attempts: {e}")
                
                except Exception as e:
                    results.append({
                        "scenario": scenario["name"],
                        "success": False,
                        "error": str(e),
                        "attempts": 1,
                        "claude_model": scenario["model"]
                    })
                    print(f"   ❌ Immediate failure: {e}")
            
            # Analyze Claude-specific results
            successful_tests = sum(1 for r in results if r["success"])
            total_tests = len(results)
            
            span.set_attribute("total_claude_tests", total_tests)
            span.set_attribute("successful_claude_tests", successful_tests)
            span.set_attribute("claude_success_rate", successful_tests / total_tests if total_tests > 0 else 0)
            span.set_attribute("claude_resilience_patterns", True)
            
            print(f"\n📊 Claude Resilience Test Results:")
            print(f"   • Total tests: {total_tests}")
            print(f"   • Successful: {successful_tests}")
            print(f"   • Success rate: {successful_tests / total_tests * 100:.1f}%")
            
            print(f"\n💡 Claude Production Resilience Patterns:")
            print(f"   • Retry logic optimized for Claude response patterns")
            print(f"   • Model-specific error handling and fallbacks")
            print(f"   • Context length management for large documents")
            print(f"   • Claude-specific rate limiting and throttling")
        
        return True
        
    except Exception as e:
        print(f"❌ Claude resilience testing error: {e}")
        return False

def main():
    """Run production patterns demonstrations."""
    print("🏭 Anthropic Claude Production Patterns with GenOps")
    print("=" * 65)
    
    # Check prerequisites
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("❌ ANTHROPIC_API_KEY environment variable not set")
        print("💡 Fix: export ANTHROPIC_API_KEY='your_api_key_here'")
        return False
    
    success = True
    
    # Run Claude production pattern examples
    success &= legal_document_review_workflow()
    success &= intelligent_content_pipeline()
    success &= claude_resilience_and_monitoring()
    
    # Summary
    print("\n" + "=" * 65)
    if success:
        print("🎉 Claude production patterns demonstration completed!")
        
        print("\n🏭 Claude Production Patterns Covered:")
        print("   ✅ Legal document review workflows with detailed analysis")
        print("   ✅ Intelligent content pipeline with Claude model optimization")
        print("   ✅ Resilience patterns with Claude-specific error handling")
        print("   ✅ Enterprise governance and policy enforcement")
        
        print("\n💼 Claude Enterprise Benefits:")
        print("   • Superior reasoning for legal and analytical workflows")
        print("   • Natural language understanding for complex documents")
        print("   • Cost-effective model selection across Claude variants")
        print("   • Complete audit trail and governance compliance")
        print("   • Production-ready resilience and monitoring")
        
        print("\n🚀 Claude Deployment Recommendations:")
        print("   • Use Claude 3.5 Sonnet for complex reasoning and analysis")
        print("   • Implement Claude 3.5 Haiku for high-volume, cost-sensitive operations")
        print("   • Deploy Claude 3 Opus for highest quality creative and strategic work")
        print("   • Set up Claude-specific monitoring and alerting thresholds")
        print("   • Establish backup strategies and graceful degradation patterns")
        
        return True
    else:
        print("❌ Claude production patterns demonstration encountered issues.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)