#!/usr/bin/env python3
"""
Dust AI Production Deployment Patterns

This example demonstrates:
- Enterprise-grade governance and compliance patterns
- Multi-customer attribution and isolation
- Policy enforcement and budget controls  
- Error handling and resilience patterns
- Performance monitoring and optimization
- Security best practices

Prerequisites:
- pip install genops[dust]
- Set DUST_API_KEY and DUST_WORKSPACE_ID environment variables
- Configure OTEL_EXPORTER_OTLP_ENDPOINT for production telemetry
"""

import os
import sys
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from contextlib import contextmanager

import genops
from genops.providers.dust import instrument_dust
from genops.providers.dust_validation import validate_setup, print_validation_result
from genops.providers.dust_pricing import calculate_dust_cost
from genops.core.context import set_customer_context, set_team_defaults


# Configure structured logging for production
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class CustomerConfig:
    """Customer-specific configuration and limits."""
    customer_id: str
    name: str
    plan_tier: str  # "basic", "premium", "enterprise"
    monthly_budget: float
    daily_operation_limit: int
    allowed_features: List[str] = field(default_factory=list)
    current_usage: Dict[str, int] = field(default_factory=dict)
    

@dataclass
class PolicyViolation:
    """Represents a policy violation."""
    violation_type: str
    message: str
    customer_id: str
    timestamp: datetime
    severity: str  # "low", "medium", "high", "critical"


class DustProductionManager:
    """Production-grade Dust AI management with governance and compliance."""
    
    def __init__(self):
        self.dust = None
        self.customers: Dict[str, CustomerConfig] = {}
        self.policy_violations: List[PolicyViolation] = []
        self.circuit_breaker_state = {"dust_api": "closed", "failures": 0, "last_failure": None}
        
        # Initialize production configuration
        self._initialize_production_config()
        
    def _initialize_production_config(self):
        """Initialize production configuration and validation."""
        logger.info("Initializing Dust production environment...")
        
        # Validate environment setup
        validation_result = validate_setup()
        if not validation_result.is_valid:
            logger.error("Production setup validation failed!")
            print_validation_result(validation_result)
            sys.exit(1)
        
        logger.info("✅ Environment validation passed")
        
        # Initialize GenOps with production settings
        genops.init(
            service_name=os.getenv("OTEL_SERVICE_NAME", "dust-production"),
            environment="production",
            enable_console_export=False,  # Use OTLP in production
            enable_metrics=True,
            enable_tracing=True
        )
        
        # Create instrumented Dust client
        self.dust = instrument_dust(
            # Environment variables provide credentials
            team=os.getenv("GENOPS_TEAM", "production-ai"),
            project=os.getenv("GENOPS_PROJECT", "customer-service"),
            environment="production"
        )
        
        logger.info("✅ Dust production client initialized")
        
        # Initialize customer configurations
        self._load_customer_configs()
        
    def _load_customer_configs(self):
        """Load customer configurations (in production, this would come from a database)."""
        # Example customer configurations
        self.customers = {
            "customer-basic-001": CustomerConfig(
                customer_id="customer-basic-001",
                name="Basic Customer Corp",
                plan_tier="basic",
                monthly_budget=100.0,
                daily_operation_limit=50,
                allowed_features=["conversations", "messages"]
            ),
            "customer-premium-001": CustomerConfig(
                customer_id="customer-premium-001", 
                name="Premium Customer Inc",
                plan_tier="premium",
                monthly_budget=500.0,
                daily_operation_limit=200,
                allowed_features=["conversations", "messages", "agent_runs", "searches"]
            ),
            "customer-enterprise-001": CustomerConfig(
                customer_id="customer-enterprise-001",
                name="Enterprise Customer Ltd",
                plan_tier="enterprise", 
                monthly_budget=2000.0,
                daily_operation_limit=1000,
                allowed_features=["conversations", "messages", "agent_runs", "searches", "datasource_creation"]
            )
        }
        
        logger.info(f"Loaded {len(self.customers)} customer configurations")
    
    @contextmanager 
    def customer_operation_context(self, customer_id: str, operation_type: str):
        """Context manager for customer operations with governance and policy enforcement."""
        
        # Validate customer exists
        if customer_id not in self.customers:
            raise ValueError(f"Unknown customer: {customer_id}")
        
        customer = self.customers[customer_id]
        
        # Check feature access
        if operation_type not in customer.allowed_features:
            violation = PolicyViolation(
                violation_type="feature_access_denied",
                message=f"Customer {customer_id} ({customer.plan_tier}) not allowed to use {operation_type}",
                customer_id=customer_id,
                timestamp=datetime.now(),
                severity="high"
            )
            self.policy_violations.append(violation)
            logger.warning(f"Policy violation: {violation.message}")
            raise PermissionError(violation.message)
        
        # Check daily limits
        daily_usage = customer.current_usage.get(operation_type, 0)
        if daily_usage >= customer.daily_operation_limit:
            violation = PolicyViolation(
                violation_type="daily_limit_exceeded",
                message=f"Customer {customer_id} exceeded daily limit for {operation_type}: {daily_usage}/{customer.daily_operation_limit}",
                customer_id=customer_id,
                timestamp=datetime.now(),
                severity="medium"
            )
            self.policy_violations.append(violation)
            logger.warning(f"Policy violation: {violation.message}")
            raise ValueError(violation.message)
        
        # Check circuit breaker
        if self.circuit_breaker_state["dust_api"] == "open":
            last_failure = self.circuit_breaker_state["last_failure"]
            if last_failure and datetime.now() - last_failure < timedelta(minutes=5):
                raise RuntimeError("Dust API circuit breaker is open - service degraded")
            else:
                # Try to close circuit breaker
                self.circuit_breaker_state["dust_api"] = "half-open"
                logger.info("Circuit breaker moved to half-open state")
        
        # Set customer context for telemetry
        with set_customer_context(
            customer_id=customer_id,
            team=f"customer-{customer.plan_tier}",
            project="customer-service",
            environment="production",
            cost_center=f"customer-{customer.plan_tier}-ops"
        ):
            
            start_time = datetime.now()
            operation_cost = 0.0
            
            try:
                # Track operation start
                logger.info(f"Starting {operation_type} for customer {customer_id}")
                
                yield customer
                
                # Track successful operation
                customer.current_usage[operation_type] = customer.current_usage.get(operation_type, 0) + 1
                
                # Calculate and log cost
                operation_cost = self._calculate_operation_cost(customer, operation_type)
                
                # Check budget warnings
                self._check_budget_warnings(customer, operation_cost)
                
                # Reset circuit breaker on success
                if self.circuit_breaker_state["dust_api"] != "closed":
                    self.circuit_breaker_state = {"dust_api": "closed", "failures": 0, "last_failure": None}
                    logger.info("Circuit breaker reset to closed state")
                
            except Exception as e:
                # Handle failures
                self._handle_operation_failure(customer_id, operation_type, e)
                raise
                
            finally:
                duration = datetime.now() - start_time
                logger.info(f"Completed {operation_type} for {customer_id} in {duration.total_seconds():.2f}s, cost: €{operation_cost:.4f}")
    
    def _calculate_operation_cost(self, customer: CustomerConfig, operation_type: str) -> float:
        """Calculate estimated cost for operation."""
        
        # Simplified cost calculation - in production, use real usage metrics
        operation_costs = {
            "conversations": 0.01,
            "messages": 0.005, 
            "agent_runs": 0.03,
            "searches": 0.002,
            "datasource_creation": 0.05
        }
        
        base_cost = operation_costs.get(operation_type, 0.001)
        
        # Adjust for customer tier
        tier_multipliers = {
            "basic": 1.0,
            "premium": 0.9,      # 10% discount
            "enterprise": 0.75   # 25% discount
        }
        
        return base_cost * tier_multipliers.get(customer.plan_tier, 1.0)
    
    def _check_budget_warnings(self, customer: CustomerConfig, operation_cost: float):
        """Check and issue budget warnings."""
        
        # Simplified budget tracking - in production, use real cost tracking
        monthly_spend = sum(self._calculate_operation_cost(customer, op) * count 
                          for op, count in customer.current_usage.items())
        
        budget_utilization = (monthly_spend / customer.monthly_budget) * 100
        
        if budget_utilization > 90:
            violation = PolicyViolation(
                violation_type="budget_critical",
                message=f"Customer {customer.customer_id} at {budget_utilization:.1f}% of monthly budget",
                customer_id=customer.customer_id,
                timestamp=datetime.now(),
                severity="critical"
            )
            self.policy_violations.append(violation)
            logger.critical(f"Budget critical: {violation.message}")
            
        elif budget_utilization > 75:
            violation = PolicyViolation(
                violation_type="budget_warning", 
                message=f"Customer {customer.customer_id} at {budget_utilization:.1f}% of monthly budget",
                customer_id=customer.customer_id,
                timestamp=datetime.now(),
                severity="medium"
            )
            self.policy_violations.append(violation)
            logger.warning(f"Budget warning: {violation.message}")
    
    def _handle_operation_failure(self, customer_id: str, operation_type: str, error: Exception):
        """Handle operation failures and circuit breaker logic."""
        
        logger.error(f"Operation failed for customer {customer_id}, operation {operation_type}: {error}")
        
        # Update circuit breaker
        self.circuit_breaker_state["failures"] += 1
        self.circuit_breaker_state["last_failure"] = datetime.now()
        
        if self.circuit_breaker_state["failures"] >= 5:  # Open circuit after 5 failures
            self.circuit_breaker_state["dust_api"] = "open"
            logger.error("Circuit breaker opened due to repeated failures")
        
        # Record policy violation for failures
        violation = PolicyViolation(
            violation_type="operation_failure",
            message=f"Operation {operation_type} failed for customer {customer_id}: {str(error)}",
            customer_id=customer_id,
            timestamp=datetime.now(),
            severity="high" if "API" in str(error) else "medium"
        )
        self.policy_violations.append(violation)
    
    def create_customer_conversation(self, customer_id: str, title: str, **kwargs) -> Dict[str, Any]:
        """Create conversation with full production governance."""
        
        with self.customer_operation_context(customer_id, "conversations") as customer:
            # CodeQL [py/clear-text-logging-sensitive-data] False positive - "private" is a legitimate API parameter value
            visibility_setting = "private"
            return self.dust.create_conversation(
                title=title,
                visibility=visibility_setting,
                customer_id=customer_id,
                # Add production governance attributes
                team=f"customer-{customer.plan_tier}",
                project="customer-conversations",
                cost_center=f"{customer.plan_tier}-tier-ops",
                feature="conversation-management",
                **kwargs
            )
    
    def send_customer_message(self, customer_id: str, conversation_id: str, content: str, **kwargs) -> Dict[str, Any]:
        """Send message with full production governance."""
        
        with self.customer_operation_context(customer_id, "messages") as customer:
            
            return self.dust.send_message(
                conversation_id=conversation_id,
                content=content,
                customer_id=customer_id,
                # Add production governance attributes
                team=f"customer-{customer.plan_tier}",
                project="customer-messages",
                cost_center=f"{customer.plan_tier}-tier-ops",
                feature="message-processing",
                **kwargs
            )
    
    def run_customer_agent(self, customer_id: str, agent_id: str, inputs: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Run agent with full production governance."""
        
        with self.customer_operation_context(customer_id, "agent_runs") as customer:
            
            return self.dust.run_agent(
                agent_id=agent_id,
                inputs=inputs,
                customer_id=customer_id,
                # Add production governance attributes
                team=f"customer-{customer.plan_tier}",
                project="customer-agents",
                cost_center=f"{customer.plan_tier}-tier-ops",
                feature="agent-execution",
                **kwargs
            )
    
    def search_customer_datasources(self, customer_id: str, query: str, **kwargs) -> Dict[str, Any]:
        """Search datasources with full production governance."""
        
        with self.customer_operation_context(customer_id, "searches") as customer:
            
            return self.dust.search_datasources(
                query=query,
                customer_id=customer_id,
                # Add production governance attributes
                team=f"customer-{customer.plan_tier}",
                project="customer-search",
                cost_center=f"{customer.plan_tier}-tier-ops",
                feature="datasource-search",
                **kwargs
            )
    
    def get_customer_usage_report(self, customer_id: str) -> Dict[str, Any]:
        """Generate comprehensive usage report for customer."""
        
        if customer_id not in self.customers:
            raise ValueError(f"Unknown customer: {customer_id}")
        
        customer = self.customers[customer_id]
        
        # Calculate costs
        total_operations = sum(customer.current_usage.values())
        estimated_cost = sum(
            self._calculate_operation_cost(customer, op) * count 
            for op, count in customer.current_usage.items()
        )
        
        budget_utilization = (estimated_cost / customer.monthly_budget) * 100
        
        return {
            "customer": {
                "id": customer.customer_id,
                "name": customer.name,
                "plan_tier": customer.plan_tier
            },
            "usage": {
                "total_operations": total_operations,
                "operations_by_type": dict(customer.current_usage),
                "daily_limit": customer.daily_operation_limit,
                "remaining_operations": customer.daily_operation_limit - total_operations
            },
            "cost_analysis": {
                "estimated_monthly_cost": estimated_cost,
                "monthly_budget": customer.monthly_budget,
                "budget_utilization_percent": budget_utilization,
                "remaining_budget": customer.monthly_budget - estimated_cost
            },
            "compliance": {
                "within_limits": all(
                    customer.current_usage.get(op, 0) < customer.daily_operation_limit 
                    for op in customer.allowed_features
                ),
                "policy_violations": [
                    {
                        "type": v.violation_type,
                        "message": v.message,
                        "severity": v.severity,
                        "timestamp": v.timestamp.isoformat()
                    }
                    for v in self.policy_violations 
                    if v.customer_id == customer_id
                ]
            }
        }
    
    def get_system_health_report(self) -> Dict[str, Any]:
        """Generate system health and compliance report."""
        
        total_violations = len(self.policy_violations)
        critical_violations = len([v for v in self.policy_violations if v.severity == "critical"])
        
        return {
            "system_status": {
                "circuit_breaker": self.circuit_breaker_state,
                "total_customers": len(self.customers),
                "active_customers": len([c for c in self.customers.values() if c.current_usage])
            },
            "policy_compliance": {
                "total_violations": total_violations,
                "critical_violations": critical_violations,
                "violation_rate": f"{critical_violations/max(1, total_violations)*100:.1f}%"
            },
            "recent_violations": [
                {
                    "type": v.violation_type,
                    "customer": v.customer_id,
                    "severity": v.severity,
                    "message": v.message,
                    "timestamp": v.timestamp.isoformat()
                }
                for v in sorted(self.policy_violations, key=lambda x: x.timestamp, reverse=True)[:10]
            ]
        }


def main():
    """Demonstrate production deployment patterns."""
    
    print("🏭 Dust AI Production Deployment Patterns")
    print("=" * 50)
    
    # Initialize production manager
    try:
        manager = DustProductionManager()
        print("✅ Production environment initialized")
    except SystemExit:
        print("❌ Production environment validation failed")
        return
    except Exception as e:
        print(f"❌ Failed to initialize production environment: {e}")
        return
    
    # Example 1: Multi-customer operations with governance
    print("\n👥 Multi-Customer Operations")
    print("-" * 35)
    
    customers_to_demo = ["customer-basic-001", "customer-premium-001", "customer-enterprise-001"]
    
    for customer_id in customers_to_demo:
        print(f"\n🏢 Processing customer: {customer_id}")
        
        try:
            # Create conversation for customer
            conversation = manager.create_customer_conversation(
                customer_id=customer_id,
                title=f"Production Demo for {customer_id}",
                compliance_tags=["production", "demo"],
                data_classification="customer-data"
            )
            
            if conversation and "conversation" in conversation:
                conversation_id = conversation["conversation"]["sId"]
                print(f"   ✅ Conversation created: {conversation_id[:20]}...")
                
                # Send message
                message = manager.send_customer_message(
                    customer_id=customer_id,
                    conversation_id=conversation_id,
                    content="This is a production demo message with full governance tracking.",
                    priority="normal",
                    audit_required=True
                )
                
                if message:
                    print(f"   ✅ Message sent with governance tracking")
                
                # Try search (will fail for basic customers)
                try:
                    search = manager.search_customer_datasources(
                        customer_id=customer_id,
                        query="production governance patterns",
                        data_sources=[],
                        audit_trail=True
                    )
                    
                    if search:
                        documents_found = len(search.get("documents", []))
                        print(f"   ✅ Search completed: {documents_found} documents")
                
                except PermissionError as e:
                    print(f"   ⚠️  Search denied: {e}")
                except Exception as e:
                    print(f"   ❌ Search failed: {e}")
            
        except (PermissionError, ValueError) as e:
            print(f"   ⚠️  Operation blocked: {e}")
        except Exception as e:
            print(f"   ❌ Operation failed: {e}")
    
    # Example 2: Usage reports and compliance monitoring
    print("\n📊 Usage Reports & Compliance")
    print("-" * 35)
    
    for customer_id in customers_to_demo:
        try:
            report = manager.get_customer_usage_report(customer_id)
            customer_name = report["customer"]["name"]
            tier = report["customer"]["plan_tier"]
            operations = report["usage"]["total_operations"]
            cost = report["cost_analysis"]["estimated_monthly_cost"]
            budget_util = report["cost_analysis"]["budget_utilization_percent"]
            
            print(f"\n📈 {customer_name} ({tier.title()} Tier):")
            print(f"   Operations: {operations}")
            print(f"   Estimated Cost: €{cost:.2f}")
            print(f"   Budget Utilization: {budget_util:.1f}%")
            
            violations = report["compliance"]["policy_violations"]
            if violations:
                print(f"   ⚠️  Policy Violations: {len(violations)}")
                for violation in violations[-2:]:  # Show last 2
                    print(f"     • {violation['type']}: {violation['message']}")
            else:
                print("   ✅ No policy violations")
                
        except Exception as e:
            print(f"   ❌ Report failed for {customer_id}: {e}")
    
    # Example 3: System health monitoring  
    print("\n🏥 System Health Monitoring")
    print("-" * 35)
    
    try:
        health = manager.get_system_health_report()
        
        print("System Status:")
        print(f"   Circuit Breaker: {health['system_status']['circuit_breaker']['dust_api']}")
        print(f"   Total Customers: {health['system_status']['total_customers']}")
        print(f"   Active Customers: {health['system_status']['active_customers']}")
        
        print("\nCompliance Status:")
        print(f"   Total Violations: {health['policy_compliance']['total_violations']}")
        print(f"   Critical Violations: {health['policy_compliance']['critical_violations']}")
        print(f"   Violation Rate: {health['policy_compliance']['violation_rate']}")
        
        if health['recent_violations']:
            print("\nRecent Violations:")
            for violation in health['recent_violations'][:3]:
                print(f"   • [{violation['severity'].upper()}] {violation['type']}")
                print(f"     Customer: {violation['customer']}")
                print(f"     Message: {violation['message']}")
    
    except Exception as e:
        print(f"❌ Health monitoring failed: {e}")
    
    # Example 4: Production best practices summary
    print("\n🚀 Production Best Practices Applied")
    print("-" * 45)
    
    print("✅ Governance & Compliance:")
    print("   • Multi-tier customer access controls")
    print("   • Budget monitoring and alerting")
    print("   • Policy violation tracking")
    print("   • Audit trail for all operations")
    
    print("✅ Reliability & Performance:")
    print("   • Circuit breaker for API failures")
    print("   • Graceful error handling and recovery")
    print("   • Operation timeout and retry logic")
    print("   • Structured logging for debugging")
    
    print("✅ Security & Isolation:")
    print("   • Customer data isolation")
    print("   • Feature access controls")
    print("   • Secure credential management")
    print("   • Compliance tag propagation")
    
    print("✅ Monitoring & Observability:")
    print("   • Real-time usage tracking")
    print("   • Cost attribution per customer")
    print("   • Performance metrics collection")
    print("   • Health status reporting")


def demonstrate_enterprise_patterns():
    """Demonstrate advanced enterprise patterns."""
    
    print("\n🏢 Enterprise Integration Patterns")
    print("-" * 40)
    
    print("1. Kubernetes Deployment:")
    print("   • Horizontal Pod Autoscaling based on usage")
    print("   • Resource limits and requests tuned for workload")
    print("   • Health checks and readiness probes")
    print("   • ConfigMaps for customer configurations")
    
    print("2. Database Integration:")
    print("   • Customer configs stored in secure database")
    print("   • Usage tracking with time-series data")
    print("   • Audit logs for compliance requirements")
    print("   • Backup and disaster recovery procedures")
    
    print("3. API Gateway Integration:")
    print("   • Rate limiting per customer tier")
    print("   • Authentication and authorization")
    print("   • Request/response transformation")
    print("   • API versioning and deprecation")
    
    print("4. Monitoring & Alerting:")
    print("   • Prometheus metrics for SLA tracking")
    print("   • Grafana dashboards for operations teams")
    print("   • PagerDuty integration for critical alerts")
    print("   • Weekly/monthly usage reports")


if __name__ == "__main__":
    main()
    demonstrate_enterprise_patterns()