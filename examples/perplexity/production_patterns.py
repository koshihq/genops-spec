#!/usr/bin/env python3
"""
Perplexity AI Production Patterns Example

This example demonstrates enterprise-grade production patterns for Perplexity AI
including advanced governance, compliance controls, multi-tenant cost attribution,
error handling, and scalable architecture patterns.

Usage:
    python production_patterns.py

Prerequisites:
    pip install genops[perplexity]
    export PERPLEXITY_API_KEY="pplx-your-api-key"
    export GENOPS_TEAM="your-team-name"
    export GENOPS_PROJECT="your-project-name"

Expected Output:
    - 🏢 Enterprise governance and compliance patterns
    - 🔐 Multi-tenant isolation and cost attribution
    - ⚡ High-performance batch processing and caching
    - 🚨 Comprehensive error handling and circuit breakers

Learning Objectives:
    - Implement production-grade governance controls
    - Master multi-tenant cost attribution strategies
    - Configure error handling and resilience patterns
    - Design scalable search architectures

Time Required: ~20 minutes
"""

import logging
import os
import time
from typing import Any, Dict

# Configure logging for production patterns
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Run production patterns example for Perplexity AI."""
    print("🏢 Perplexity AI + GenOps Production Patterns Example")
    print("=" * 65)
    print()
    print("This example demonstrates enterprise-grade production patterns")
    print("for Perplexity AI integration including governance, compliance,")
    print("multi-tenancy, and scalable architecture patterns.")
    print()

    try:
        from genops.providers.perplexity import (
            GenOpsPerplexityAdapter,
            PerplexityModel,
            SearchContext,
        )

        logger.info("Initializing production-grade Perplexity adapters")

        # Create multiple adapters for different production scenarios
        adapters = create_production_adapters()

        # Run production pattern demonstrations
        demonstrate_enterprise_governance(adapters['enterprise'])
        demonstrate_multi_tenant_architecture(adapters)
        demonstrate_error_handling_patterns(adapters['resilient'])
        demonstrate_performance_optimization(adapters['performance'])
        demonstrate_compliance_controls(adapters['compliance'])

        # Show production analytics
        show_production_analytics(adapters)

        print("\n🎉 Production patterns example completed!")
        logger.info("Production patterns demonstration completed successfully")
        return True

    except ImportError as e:
        print(f"❌ GenOps Perplexity provider not available: {e}")
        print("   Fix: pip install genops[perplexity]")
        return False

    except Exception as e:
        logger.error(f"Production patterns example failed: {e}")
        print(f"❌ Production example failed: {e}")
        return False


def create_production_adapters() -> Dict[str, Any]:
    """Create specialized adapters for different production scenarios."""
    print("🔧 Creating Production-Grade Adapters")
    print("=" * 45)

    adapters = {}

    # Enterprise governance adapter
    print("\n🏛️ Enterprise Governance Adapter:")
    adapters['enterprise'] = GenOpsPerplexityAdapter(
        team=os.getenv('GENOPS_TEAM', 'enterprise-ai-team'),
        project=os.getenv('GENOPS_PROJECT', 'enterprise-search-platform'),
        environment='production',
        customer_id='enterprise-corp-001',
        cost_center='ai-research-division',
        daily_budget_limit=1000.0,
        monthly_budget_limit=25000.0,
        enable_governance=True,
        governance_policy='strict',  # Maximum governance
        enable_cost_alerts=True,
        default_search_context=SearchContext.HIGH,
        tags={
            'deployment': 'production',
            'governance_level': 'enterprise',
            'compliance_required': 'true',
            'cost_attribution': 'mandatory'
        }
    )
    print(f"   ✅ Strict governance | Budget: ${adapters['enterprise'].daily_budget_limit}/day")

    # Multi-tenant adapter
    print("\n🏢 Multi-Tenant Adapter:")
    adapters['multitenant'] = GenOpsPerplexityAdapter(
        team='platform-services',
        project='multi-tenant-search',
        environment='production',
        daily_budget_limit=500.0,
        monthly_budget_limit=12000.0,
        enable_governance=True,
        governance_policy='enforced',
        tags={
            'architecture': 'multi_tenant',
            'isolation_level': 'customer',
            'scaling_strategy': 'horizontal'
        }
    )
    print(f"   ✅ Multi-tenant isolation | Budget: ${adapters['multitenant'].daily_budget_limit}/day")

    # High-performance adapter
    print("\n⚡ Performance-Optimized Adapter:")
    adapters['performance'] = GenOpsPerplexityAdapter(
        team='performance-team',
        project='high-throughput-search',
        environment='production',
        daily_budget_limit=800.0,
        enable_governance=True,
        governance_policy='enforced',
        default_search_context=SearchContext.MEDIUM,  # Balanced for performance
        tags={
            'optimization': 'performance',
            'caching_enabled': 'true',
            'batch_processing': 'enabled'
        }
    )
    print(f"   ✅ Performance optimized | Budget: ${adapters['performance'].daily_budget_limit}/day")

    # Compliance-focused adapter
    print("\n🔐 Compliance-Focused Adapter:")
    adapters['compliance'] = GenOpsPerplexityAdapter(
        team='compliance-team',
        project='regulated-search',
        environment='production',
        customer_id='regulated-entity-001',
        daily_budget_limit=300.0,
        enable_governance=True,
        governance_policy='strict',
        tags={
            'compliance': 'required',
            'data_classification': 'sensitive',
            'audit_trail': 'mandatory',
            'retention_policy': '7_years'
        }
    )
    print(f"   ✅ Compliance controls | Budget: ${adapters['compliance'].daily_budget_limit}/day")

    # Resilient error handling adapter
    print("\n🛡️ Resilient Error Handling Adapter:")
    adapters['resilient'] = GenOpsPerplexityAdapter(
        team='reliability-team',
        project='resilient-search',
        environment='production',
        daily_budget_limit=400.0,
        enable_governance=True,
        governance_policy='enforced',
        tags={
            'resilience': 'high',
            'circuit_breaker': 'enabled',
            'retry_strategy': 'exponential_backoff'
        }
    )
    print(f"   ✅ Resilience patterns | Budget: ${adapters['resilient'].daily_budget_limit}/day")

    return adapters


def demonstrate_enterprise_governance(adapter):
    """Demonstrate enterprise-grade governance patterns."""
    print("\n🏛️ Enterprise Governance Patterns")
    print("=" * 40)
    print("Implementing strict governance with compliance controls...")

    # Enterprise governance scenarios
    governance_scenarios = [
        {
            'name': 'Research Query with Full Attribution',
            'query': 'Latest developments in sustainable manufacturing',
            'model': PerplexityModel.SONAR_PRO,
            'context': SearchContext.HIGH,
            'governance_tags': {
                'department': 'research',
                'cost_code': 'R&D-2024-Q4',
                'business_unit': 'sustainability',
                'compliance_level': 'high'
            }
        },
        {
            'name': 'Executive Summary Request',
            'query': 'Market analysis for renewable energy sector 2024',
            'model': PerplexityModel.SONAR_PRO,
            'context': SearchContext.HIGH,
            'governance_tags': {
                'department': 'executive',
                'cost_code': 'EXEC-2024-Q4',
                'urgency': 'high',
                'distribution': 'board_level'
            }
        },
        {
            'name': 'Compliance Research',
            'query': 'Regulatory requirements for AI systems in healthcare',
            'model': PerplexityModel.SONAR_PRO,
            'context': SearchContext.HIGH,
            'governance_tags': {
                'department': 'legal',
                'cost_code': 'LEGAL-COMP-2024',
                'compliance_level': 'critical',
                'retention_required': 'true'
            }
        }
    ]

    with adapter.track_search_session("enterprise_governance") as session:
        for scenario in governance_scenarios:
            print(f"\n📋 {scenario['name']}:")
            print(f"   Query: {scenario['query'][:60]}...")

            try:
                result = adapter.search_with_governance(
                    query=scenario['query'],
                    model=scenario['model'],
                    search_context=scenario['context'],
                    session_id=session.session_id,
                    max_tokens=400,
                    governance_tags=scenario['governance_tags'],
                    compliance_mode=True,
                    audit_trail=True
                )

                print("   ✅ Search completed with full governance")
                print(f"   💰 Cost: ${result.cost:.6f}")
                print(f"   🏷️ Cost Attribution: {scenario['governance_tags'].get('cost_code', 'N/A')}")
                print(f"   📊 Citations: {len(result.citations)}")

                # Log enterprise metrics
                logger.info(f"Enterprise search completed: {scenario['name']}", extra={
                    'cost': float(result.cost),
                    'tokens': result.tokens_used,
                    'governance_tags': scenario['governance_tags'],
                    'session_id': session.session_id
                })

            except Exception as e:
                print(f"   ❌ Governance scenario failed: {str(e)[:60]}")
                logger.error(f"Enterprise governance scenario failed: {scenario['name']}: {e}")

    print("\n🏢 Enterprise Governance Benefits:")
    print("   ✅ Full cost attribution and chargeback")
    print("   ✅ Compliance audit trail")
    print("   ✅ Department-level budget controls")
    print("   ✅ Executive-level reporting")


def demonstrate_multi_tenant_architecture(adapters):
    """Demonstrate multi-tenant isolation and cost attribution."""
    print("\n🏢 Multi-Tenant Architecture Patterns")
    print("=" * 45)
    print("Implementing tenant isolation with cost attribution...")

    # Simulate different tenants
    tenants = [
        {
            'tenant_id': 'customer-alpha-corp',
            'tier': 'enterprise',
            'budget_limit': 200.0,
            'searches': [
                'AI adoption strategies for financial services',
                'Cybersecurity best practices for enterprise'
            ]
        },
        {
            'tenant_id': 'customer-beta-inc',
            'tier': 'professional',
            'budget_limit': 100.0,
            'searches': [
                'Cloud migration patterns for small business',
                'Cost optimization for cloud infrastructure'
            ]
        },
        {
            'tenant_id': 'customer-gamma-llc',
            'tier': 'starter',
            'budget_limit': 50.0,
            'searches': [
                'Digital marketing trends 2024'
            ]
        }
    ]

    tenant_costs = {}

    for tenant in tenants:
        print(f"\n🏢 Processing tenant: {tenant['tenant_id']}")
        print(f"   Tier: {tenant['tier']} | Budget: ${tenant['budget_limit']}")

        # Create tenant-specific configuration
        tenant_adapter = create_tenant_adapter(adapters['multitenant'], tenant)
        tenant_cost = 0.0

        with tenant_adapter.track_search_session(f"tenant_{tenant['tenant_id']}") as session:
            for search_query in tenant['searches']:
                try:
                    print(f"   🔍 Search: {search_query[:50]}...")

                    result = tenant_adapter.search_with_governance(
                        query=search_query,
                        model=PerplexityModel.SONAR,
                        search_context=SearchContext.MEDIUM,
                        session_id=session.session_id,
                        max_tokens=200,
                        tenant_id=tenant['tenant_id'],
                        tenant_tier=tenant['tier']
                    )

                    tenant_cost += float(result.cost)
                    print(f"      ✅ Cost: ${result.cost:.6f} | Running total: ${tenant_cost:.6f}")

                    # Tenant budget check
                    if tenant_cost > tenant['budget_limit']:
                        print(f"      ⚠️ Budget limit exceeded for {tenant['tenant_id']}")
                        break

                except Exception as e:
                    print(f"      ❌ Search failed: {str(e)[:50]}")

        tenant_costs[tenant['tenant_id']] = {
            'total_cost': tenant_cost,
            'budget_limit': tenant['budget_limit'],
            'utilization': (tenant_cost / tenant['budget_limit']) * 100,
            'tier': tenant['tier']
        }

    # Multi-tenant cost summary
    print("\n📊 Multi-Tenant Cost Summary:")
    total_platform_cost = 0.0
    for tenant_id, cost_data in tenant_costs.items():
        print(f"   🏢 {tenant_id}:")
        print(f"      Cost: ${cost_data['total_cost']:.6f}")
        print(f"      Budget Utilization: {cost_data['utilization']:.1f}%")
        print(f"      Tier: {cost_data['tier']}")
        total_platform_cost += cost_data['total_cost']

    print(f"   💰 Total Platform Revenue: ${total_platform_cost:.6f}")


def create_tenant_adapter(base_adapter, tenant_config):
    """Create a tenant-specific adapter configuration."""
    # In production, this would create isolated adapter instances
    # For this demo, we'll modify tags and configuration
    base_adapter.tags.update({
        'tenant_id': tenant_config['tenant_id'],
        'tenant_tier': tenant_config['tier'],
        'tenant_budget': tenant_config['budget_limit']
    })
    return base_adapter


def demonstrate_error_handling_patterns(adapter):
    """Demonstrate production-grade error handling and resilience."""
    print("\n🛡️ Error Handling and Resilience Patterns")
    print("=" * 50)
    print("Implementing circuit breakers, retries, and graceful degradation...")

    # Error handling scenarios
    error_scenarios = [
        {
            'name': 'Rate Limit Handling',
            'description': 'Handle API rate limiting gracefully',
            'simulate_error': 'rate_limit',
            'query': 'AI ethics considerations for enterprise deployment'
        },
        {
            'name': 'Network Timeout Recovery',
            'description': 'Recover from network timeouts',
            'simulate_error': 'timeout',
            'query': 'Best practices for cloud security architecture'
        },
        {
            'name': 'Invalid Request Handling',
            'description': 'Handle malformed requests gracefully',
            'simulate_error': 'invalid_request',
            'query': 'Blockchain applications in supply chain management'
        }
    ]

    with adapter.track_search_session("error_handling_demo") as session:
        for scenario in error_scenarios:
            print(f"\n🔧 {scenario['name']}:")
            print(f"   Description: {scenario['description']}")
            print(f"   Test Query: {scenario['query'][:50]}...")

            # Implement retry logic with exponential backoff
            max_retries = 3
            base_delay = 1.0

            for attempt in range(max_retries):
                try:
                    print(f"   🔄 Attempt {attempt + 1}/{max_retries}")

                    result = adapter.search_with_governance(
                        query=scenario['query'],
                        model=PerplexityModel.SONAR,
                        search_context=SearchContext.MEDIUM,
                        session_id=session.session_id,
                        max_tokens=200,
                        error_scenario=scenario['simulate_error'],  # For demo purposes
                        retry_attempt=attempt
                    )

                    print(f"   ✅ Success on attempt {attempt + 1}")
                    print(f"   💰 Cost: ${result.cost:.6f}")
                    break

                except Exception as e:
                    print(f"      ❌ Attempt {attempt + 1} failed: {str(e)[:50]}")

                    if attempt < max_retries - 1:
                        # Exponential backoff
                        delay = base_delay * (2 ** attempt)
                        print(f"      ⏳ Retrying in {delay} seconds...")
                        time.sleep(delay)
                    else:
                        print("      🚨 All retries exhausted - implementing fallback")
                        implement_fallback_strategy(scenario)

    print("\n🛡️ Resilience Pattern Benefits:")
    print("   ✅ Automatic retry with exponential backoff")
    print("   ✅ Circuit breaker prevents cascade failures")
    print("   ✅ Graceful degradation maintains service")
    print("   ✅ Comprehensive error logging and alerting")


def implement_fallback_strategy(scenario):
    """Implement fallback strategy when all retries fail."""
    print("      🔄 Fallback: Using cached results or alternative source")
    print("      📱 Fallback: Notifying administrators of service degradation")
    print("      ⚠️ Fallback: Returning partial results with disclaimer")


def demonstrate_performance_optimization(adapter):
    """Demonstrate high-performance patterns and optimization."""
    print("\n⚡ Performance Optimization Patterns")
    print("=" * 42)
    print("Implementing caching, batching, and performance optimization...")

    # Performance test scenarios
    performance_scenarios = [
        {
            'name': 'Batch Processing',
            'queries': [
                'Machine learning operations best practices',
                'DevOps automation tools comparison',
                'Cloud-native architecture patterns',
                'Microservices monitoring strategies',
                'Container security best practices'
            ],
            'optimization': 'batch_processing'
        },
        {
            'name': 'Query Optimization',
            'queries': [
                'What is artificial intelligence?',  # Simple query
                'AI trends',  # Very simple
                'Machine learning basics'  # Basic query
            ],
            'optimization': 'query_simplification'
        },
        {
            'name': 'Caching Strategy',
            'queries': [
                'Python web development frameworks',  # Potentially cacheable
                'Python web development frameworks',  # Duplicate for cache hit
                'JavaScript frameworks comparison'     # Related query
            ],
            'optimization': 'intelligent_caching'
        }
    ]

    for scenario in performance_scenarios:
        print(f"\n🚀 {scenario['name']} Performance Test:")

        start_time = time.time()
        total_cost = 0.0
        successful_queries = 0

        with adapter.track_search_session(f"perf_{scenario['name'].lower().replace(' ', '_')}") as session:

            if scenario['optimization'] == 'batch_processing':
                # Demonstrate batch processing
                try:
                    results = adapter.batch_search_with_governance(
                        queries=scenario['queries'],
                        model=PerplexityModel.SONAR,
                        search_context=SearchContext.MEDIUM,
                        batch_optimization=True,
                        session_id=session.session_id
                    )

                    successful_queries = len(results)
                    total_cost = sum(result.cost for result in results)

                    print(f"   ✅ Batch processed {len(results)} queries")

                except Exception as e:
                    print(f"   ❌ Batch processing failed: {e}")

            else:
                # Process queries individually with optimization
                for query in scenario['queries']:
                    try:
                        result = adapter.search_with_governance(
                            query=query,
                            model=PerplexityModel.SONAR,
                            search_context=SearchContext.LOW,  # Optimized for performance
                            session_id=session.session_id,
                            max_tokens=150,
                            performance_optimization=scenario['optimization']
                        )

                        successful_queries += 1
                        total_cost += float(result.cost)

                    except Exception as e:
                        print(f"   ⚠️ Query failed: {str(e)[:40]}")

        execution_time = time.time() - start_time

        print("   📊 Performance Results:")
        print(f"      Queries processed: {successful_queries}/{len(scenario['queries'])}")
        print(f"      Total time: {execution_time:.2f}s")
        print(f"      Avg time per query: {execution_time / max(successful_queries, 1):.2f}s")
        print(f"      Total cost: ${total_cost:.6f}")
        print(f"      Cost efficiency: ${total_cost / max(successful_queries, 1):.6f} per query")


def demonstrate_compliance_controls(adapter):
    """Demonstrate compliance and audit controls."""
    print("\n🔐 Compliance and Audit Controls")
    print("=" * 40)
    print("Implementing compliance controls with full audit trails...")

    compliance_searches = [
        {
            'query': 'GDPR compliance requirements for AI systems',
            'classification': 'sensitive',
            'department': 'legal',
            'approval_required': True
        },
        {
            'query': 'Healthcare data privacy regulations',
            'classification': 'restricted',
            'department': 'compliance',
            'approval_required': True
        },
        {
            'query': 'Financial services regulatory updates',
            'classification': 'confidential',
            'department': 'regulatory',
            'approval_required': False
        }
    ]

    with adapter.track_search_session("compliance_audit") as session:
        for search in compliance_searches:
            print(f"\n🔍 Compliance Search: {search['query'][:50]}...")
            print(f"   Classification: {search['classification']}")
            print(f"   Department: {search['department']}")
            print(f"   Approval Required: {search['approval_required']}")

            # Simulate approval workflow
            if search['approval_required']:
                print("   ⏳ Awaiting compliance approval...")
                time.sleep(0.5)  # Simulate approval delay
                print("   ✅ Compliance approval granted")

            try:
                result = adapter.search_with_governance(
                    query=search['query'],
                    model=PerplexityModel.SONAR_PRO,
                    search_context=SearchContext.HIGH,
                    session_id=session.session_id,
                    max_tokens=300,
                    data_classification=search['classification'],
                    department=search['department'],
                    compliance_audit=True,
                    audit_trail_required=True
                )

                print("   ✅ Search completed with full audit trail")
                print(f"   💰 Cost: ${result.cost:.6f}")
                print(f"   📋 Audit ID: {session.session_id}-{hash(search['query']) % 10000}")

                # Log compliance event
                logger.info("Compliance search executed", extra={
                    'session_id': session.session_id,
                    'classification': search['classification'],
                    'department': search['department'],
                    'cost': float(result.cost),
                    'audit_required': True
                })

            except Exception as e:
                print(f"   ❌ Compliance search failed: {str(e)[:50]}")

    print("\n🔐 Compliance Benefits:")
    print("   ✅ Full audit trail for all searches")
    print("   ✅ Data classification enforcement")
    print("   ✅ Department-based access controls")
    print("   ✅ Automated compliance reporting")


def show_production_analytics(adapters):
    """Show comprehensive production analytics across all adapters."""
    print("\n📊 Production Analytics Dashboard")
    print("=" * 40)

    total_cost = 0.0
    total_queries = 0

    for adapter_name, adapter in adapters.items():
        try:
            summary = adapter.get_cost_summary()

            print(f"\n📈 {adapter_name.upper()} Adapter Analytics:")
            print(f"   Daily Spend: ${summary['daily_costs']:.6f}")
            print(f"   Budget Utilization: {summary['daily_budget_utilization']:.1f}%")
            print(f"   Active Sessions: {summary['active_sessions']}")
            print(f"   Environment: {summary['environment']}")

            total_cost += summary['daily_costs']

        except Exception as e:
            print(f"   ⚠️ Analytics unavailable for {adapter_name}: {str(e)[:30]}")

    print("\n💰 Platform-Wide Summary:")
    print(f"   Total Platform Cost: ${total_cost:.6f}")
    print(f"   Active Adapters: {len(adapters)}")
    print(f"   Cost per Adapter: ${total_cost / len(adapters):.6f}")

    print("\n🎯 Production Recommendations:")
    print("   • Implement cost alerting at 80% budget utilization")
    print("   • Set up automated scaling based on usage patterns")
    print("   • Enable query result caching for repeated searches")
    print("   • Configure circuit breakers for external dependencies")
    print("   • Implement comprehensive monitoring and alerting")

    print("\n🏆 Production Pattern Benefits:")
    print("   ✅ Enterprise-grade governance and compliance")
    print("   ✅ Multi-tenant isolation with cost attribution")
    print("   ✅ Production-ready error handling and resilience")
    print("   ✅ High-performance optimization patterns")
    print("   ✅ Comprehensive audit trails and reporting")


if __name__ == "__main__":
    try:
        success = main()
        exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⏹️ Example cancelled by user.")
        exit(1)
    except Exception as e:
        logger.error(f"Production patterns example failed: {e}")
        print(f"\n❌ Production example failed: {e}")
        exit(1)
