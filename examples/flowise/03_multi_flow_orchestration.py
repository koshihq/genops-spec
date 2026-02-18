#!/usr/bin/env python3
"""
Example: Multi-Flow Orchestration with Governance Context

Complexity: ⭐⭐ Intermediate

This example demonstrates orchestrating multiple Flowise flows in sequence
with shared governance context, session tracking, and cost aggregation.
Perfect for complex AI workflows that span multiple specialized flows.

Prerequisites:
- Flowise instance running
- Multiple chatflows created (or one flow for simulation)
- GenOps package installed

Usage:
    python 03_multi_flow_orchestration.py

Environment Variables:
    FLOWISE_BASE_URL: Flowise instance URL
    FLOWISE_API_KEY: API key (optional for local dev)
    GENOPS_TEAM: Team name for governance
"""

import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

from genops.providers.flowise import instrument_flowise
from genops.providers.flowise_validation import validate_flowise_setup

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class WorkflowStep:
    """Represents a single step in a multi-flow workflow."""

    name: str
    chatflow_id: str
    input_template: str
    depends_on: list[str] = field(default_factory=list)
    governance_overrides: dict[str, Any] = field(default_factory=dict)
    timeout_seconds: int = 30


@dataclass
class WorkflowResult:
    """Result of a workflow step execution."""

    step_name: str
    success: bool
    response: Optional[dict] = None
    error: Optional[str] = None
    execution_time_ms: int = 0
    estimated_cost: float = 0.0
    token_count: int = 0


@dataclass
class WorkflowSession:
    """Manages a complete workflow session with governance context."""

    session_id: str
    workflow_name: str
    customer_id: Optional[str] = None
    user_tier: str = "standard"
    steps: list[WorkflowStep] = field(default_factory=list)
    results: list[WorkflowResult] = field(default_factory=list)
    total_cost: float = 0.0
    total_tokens: int = 0
    start_time: float = field(default_factory=time.time)

    def add_step(self, step: WorkflowStep):
        """Add a step to the workflow."""
        self.steps.append(step)

    def get_step_result(self, step_name: str) -> Optional[WorkflowResult]:
        """Get result of a completed step."""
        for result in self.results:
            if result.step_name == step_name:
                return result
        return None

    def format_input(self, template: str, **kwargs) -> str:
        """Format input template with previous results and kwargs."""
        # Get results from previous steps
        step_results = {}
        for result in self.results:
            if result.success and result.response:
                response_text = (
                    result.response.get("text")
                    or result.response.get("answer")
                    or result.response.get("content")
                    or str(result.response)
                )
                step_results[result.step_name] = response_text

        # Combine with provided kwargs
        format_vars = {**step_results, **kwargs}

        try:
            return template.format(**format_vars)
        except KeyError as e:
            logger.warning(f"Missing template variable: {e}")
            return template


class MultiFlowOrchestrator:
    """Orchestrates multiple Flowise flows with governance tracking."""

    def __init__(self, flowise_adapter, default_governance: dict[str, Any]):
        self.flowise = flowise_adapter
        self.default_governance = default_governance

    def execute_workflow(self, session: WorkflowSession) -> bool:
        """Execute a complete workflow session."""

        logger.info(
            f"Starting workflow: {session.workflow_name} (Session: {session.session_id})"
        )

        # Track session-level governance context
        session_governance = {
            **self.default_governance,
            "session_id": session.session_id,
            "workflow_name": session.workflow_name,
            "customer_id": session.customer_id,
            "user_tier": session.user_tier,
        }

        for step in session.steps:
            # Check dependencies
            if not self._check_dependencies(step, session):
                logger.error(f"Dependencies not met for step: {step.name}")
                session.results.append(
                    WorkflowResult(
                        step_name=step.name, success=False, error="Dependencies not met"
                    )
                )
                continue

            # Execute step
            result = self._execute_step(step, session, session_governance)
            session.results.append(result)

            # Update session totals
            session.total_cost += result.estimated_cost
            session.total_tokens += result.token_count

            # Stop on failure if step is critical
            if not result.success:
                logger.error(f"Step failed: {step.name} - {result.error}")
                if step.name.endswith("_required"):
                    logger.error("Critical step failed, stopping workflow")
                    break

        # Calculate final metrics
        session.duration_seconds = time.time() - session.start_time
        successful_steps = sum(1 for r in session.results if r.success)

        logger.info(
            f"Workflow completed: {successful_steps}/{len(session.steps)} steps successful"
        )
        logger.info(
            f"Total cost: ${session.total_cost:.6f}, Total tokens: {session.total_tokens}"
        )

        return successful_steps == len(session.steps)

    def _check_dependencies(self, step: WorkflowStep, session: WorkflowSession) -> bool:
        """Check if step dependencies are satisfied."""
        for dep in step.depends_on:
            result = session.get_step_result(dep)
            if not result or not result.success:
                return False
        return True

    def _execute_step(
        self,
        step: WorkflowStep,
        session: WorkflowSession,
        session_governance: dict[str, Any],
    ) -> WorkflowResult:
        """Execute a single workflow step."""

        logger.info(f"Executing step: {step.name}")

        start_time = time.time()

        try:
            # Prepare governance attributes for this step
            step_governance = {
                **session_governance,
                **step.governance_overrides,
                "workflow_step": step.name,
                "step_index": len(session.results),
            }

            # Format input with previous results
            formatted_input = session.format_input(step.input_template)

            logger.debug(f"Step input: {formatted_input[:100]}...")

            # Execute the flow
            response = self.flowise.predict_flow(
                chatflow_id=step.chatflow_id,
                question=formatted_input,
                sessionId=session.session_id,  # Maintain session continuity
                **step_governance,
            )

            execution_time_ms = int((time.time() - start_time) * 1000)

            # Estimate tokens and cost (rough approximation)
            response_text = (
                response.get("text", "")
                if isinstance(response, dict)
                else str(response)
            )
            estimated_tokens = len(formatted_input.split()) + len(response_text.split())
            estimated_cost = estimated_tokens * 0.000002  # Rough estimate

            return WorkflowResult(
                step_name=step.name,
                success=True,
                response=response,
                execution_time_ms=execution_time_ms,
                estimated_cost=estimated_cost,
                token_count=estimated_tokens,
            )

        except Exception as e:
            execution_time_ms = int((time.time() - start_time) * 1000)

            return WorkflowResult(
                step_name=step.name,
                success=False,
                error=str(e),
                execution_time_ms=execution_time_ms,
            )


def create_document_analysis_workflow(chatflow_id: str) -> WorkflowSession:
    """Create a multi-step document analysis workflow."""

    session = WorkflowSession(
        session_id=f"doc-analysis-{uuid.uuid4().hex[:8]}",
        workflow_name="Document Analysis Pipeline",
        customer_id="enterprise-customer-456",
        user_tier="premium",
    )

    # Step 1: Initial document analysis
    session.add_step(
        WorkflowStep(
            name="document_intake",
            chatflow_id=chatflow_id,
            input_template=(
                "Please analyze the following document type and provide a structured summary: "
                "This is a business proposal document containing project requirements, "
                "budget information, timeline details, and technical specifications."
            ),
            governance_overrides={
                "feature": "document-intake",
                "document_type": "business-proposal",
            },
        )
    )

    # Step 2: Extract key information (depends on step 1)
    session.add_step(
        WorkflowStep(
            name="information_extraction",
            chatflow_id=chatflow_id,
            input_template=(
                "Based on this document analysis: {document_intake}\n\n"
                "Please extract and structure the following key information:\n"
                "1. Project timeline and milestones\n"
                "2. Budget breakdown and cost estimates\n"
                "3. Technical requirements and specifications\n"
                "4. Risk factors and dependencies\n"
                "Provide a JSON-like structured response."
            ),
            depends_on=["document_intake"],
            governance_overrides={
                "feature": "information-extraction",
                "extraction_type": "structured-data",
            },
        )
    )

    # Step 3: Generate executive summary (depends on step 2)
    session.add_step(
        WorkflowStep(
            name="executive_summary",
            chatflow_id=chatflow_id,
            input_template=(
                "Using this extracted information: {information_extraction}\n\n"
                "Create a concise executive summary suitable for C-level presentation. "
                "Focus on key business value, timeline, investment required, and strategic alignment. "
                "Keep it under 200 words and highlight critical decision points."
            ),
            depends_on=["information_extraction"],
            governance_overrides={
                "feature": "executive-summary",
                "output_type": "c-level-presentation",
            },
        )
    )

    # Step 4: Risk assessment (depends on step 2)
    session.add_step(
        WorkflowStep(
            name="risk_assessment",
            chatflow_id=chatflow_id,
            input_template=(
                "Based on this project information: {information_extraction}\n\n"
                "Conduct a comprehensive risk assessment covering:\n"
                "1. Technical risks and mitigation strategies\n"
                "2. Budget and timeline risks\n"
                "3. Resource availability risks\n"
                "4. Market and competitive risks\n"
                "Rate each risk level (Low/Medium/High) and provide actionable mitigation plans."
            ),
            depends_on=["information_extraction"],
            governance_overrides={
                "feature": "risk-assessment",
                "analysis_type": "comprehensive-risk",
            },
        )
    )

    return session


def create_customer_service_workflow(chatflow_id: str) -> WorkflowSession:
    """Create a multi-step customer service workflow."""

    session = WorkflowSession(
        session_id=f"customer-service-{uuid.uuid4().hex[:8]}",
        workflow_name="Customer Service Escalation",
        customer_id="standard-customer-789",
        user_tier="standard",
    )

    # Step 1: Initial inquiry analysis
    session.add_step(
        WorkflowStep(
            name="inquiry_analysis",
            chatflow_id=chatflow_id,
            input_template=(
                "Customer inquiry: 'I've been having issues with my account login for the past week. "
                "I've tried resetting my password multiple times but still can't access my account. "
                "I have an important presentation tomorrow and need access to my files urgently.'\n\n"
                "Analyze this inquiry and categorize it by: priority level, department, issue type, and sentiment."
            ),
            governance_overrides={
                "feature": "inquiry-analysis",
                "channel": "chat-support",
            },
        )
    )

    # Step 2: Solution recommendation
    session.add_step(
        WorkflowStep(
            name="solution_recommendation",
            chatflow_id=chatflow_id,
            input_template=(
                "Customer inquiry analysis: {inquiry_analysis}\n\n"
                "Based on this analysis, provide:\n"
                "1. Immediate troubleshooting steps the customer can try\n"
                "2. Escalation path if basic steps don't work\n"
                "3. Estimated resolution timeframe\n"
                "4. Proactive follow-up recommendations"
            ),
            depends_on=["inquiry_analysis"],
            governance_overrides={
                "feature": "solution-recommendation",
                "solution_type": "self-service-plus-escalation",
            },
        )
    )

    # Step 3: Follow-up communication
    session.add_step(
        WorkflowStep(
            name="followup_communication",
            chatflow_id=chatflow_id,
            input_template=(
                "Solution recommendation: {solution_recommendation}\n\n"
                "Draft a professional, empathetic customer communication that:\n"
                "1. Acknowledges the urgency and inconvenience\n"
                "2. Provides clear next steps\n"
                "3. Sets appropriate expectations\n"
                "4. Includes escalation contact information\n"
                "Keep it concise but thorough."
            ),
            depends_on=["solution_recommendation"],
            governance_overrides={
                "feature": "customer-communication",
                "communication_type": "urgent-issue-response",
            },
        )
    )

    return session


def create_content_generation_workflow(chatflow_id: str) -> WorkflowSession:
    """Create a multi-step content generation workflow."""

    session = WorkflowSession(
        session_id=f"content-gen-{uuid.uuid4().hex[:8]}",
        workflow_name="Marketing Content Pipeline",
        customer_id="marketing-team-internal",
        user_tier="internal",
    )

    # Step 1: Market research and analysis
    session.add_step(
        WorkflowStep(
            name="market_research",
            chatflow_id=chatflow_id,
            input_template=(
                "Conduct market research for a new AI-powered project management tool. "
                "Analyze current market trends, competitor landscape, target audience needs, "
                "and positioning opportunities in the project management software space."
            ),
            governance_overrides={
                "feature": "market-research",
                "campaign_type": "product-launch",
            },
        )
    )

    # Step 2: Content strategy development
    session.add_step(
        WorkflowStep(
            name="content_strategy",
            chatflow_id=chatflow_id,
            input_template=(
                "Market research insights: {market_research}\n\n"
                "Develop a comprehensive content strategy including:\n"
                "1. Key messaging pillars and value propositions\n"
                "2. Target audience personas and pain points\n"
                "3. Content themes and topic clusters\n"
                "4. Competitive differentiation angles\n"
                "5. Content distribution channel recommendations"
            ),
            depends_on=["market_research"],
            governance_overrides={
                "feature": "content-strategy",
                "strategy_type": "go-to-market",
            },
        )
    )

    # Step 3: Blog post creation
    session.add_step(
        WorkflowStep(
            name="blog_post_creation",
            chatflow_id=chatflow_id,
            input_template=(
                "Content strategy: {content_strategy}\n\n"
                "Write an engaging blog post (800-1000 words) titled: "
                "'5 Ways AI is Revolutionizing Project Management in 2024'\n"
                "Include practical examples, data points, and a compelling call-to-action. "
                "Optimize for SEO and readability."
            ),
            depends_on=["content_strategy"],
            governance_overrides={
                "feature": "content-creation",
                "content_type": "blog-post",
            },
        )
    )

    # Step 4: Social media adaptation
    session.add_step(
        WorkflowStep(
            name="social_media_adaptation",
            chatflow_id=chatflow_id,
            input_template=(
                "Blog post: {blog_post_creation}\n\n"
                "Adapt this blog post into social media content:\n"
                "1. LinkedIn post (professional tone, 150-200 words)\n"
                "2. Twitter thread (5-7 tweets with relevant hashtags)\n"
                "3. Facebook post (engaging, 100-150 words)\n"
                "Maintain key messaging while optimizing for each platform."
            ),
            depends_on=["blog_post_creation"],
            governance_overrides={
                "feature": "social-media-adaptation",
                "platforms": "linkedin-twitter-facebook",
            },
        )
    )

    return session


def demonstrate_multi_flow_orchestration():
    """Demonstrate multi-flow orchestration with governance context."""

    print("🔄 Multi-Flow Orchestration Example")
    print("=" * 50)

    # Configuration
    base_url = os.getenv("FLOWISE_BASE_URL", "http://localhost:3000")
    api_key = os.getenv("FLOWISE_API_KEY")
    team = os.getenv("GENOPS_TEAM", "orchestration-demo")
    project = "multi-flow-workflows"

    print(f"Flowise URL: {base_url}")
    print(f"Team: {team}")
    print(f"Project: {project}")

    # Step 1: Validate setup and get chatflow
    print("\n📋 Step 1: Validating setup and getting chatflows...")

    try:
        result = validate_flowise_setup(base_url, api_key)
        if not result.is_valid:
            print("❌ Setup validation failed.")
            return False

        flowise = instrument_flowise(
            base_url=base_url,
            api_key=api_key,
            team=team,
            project=project,
            environment="development",
        )

        chatflows = flowise.get_chatflows()
        if not chatflows:
            print("❌ No chatflows available.")
            return False

        # Use first available chatflow for all workflow steps
        chatflow_id = chatflows[0].get("id")
        chatflow_name = chatflows[0].get("name", "Unnamed")
        print(f"✅ Using chatflow: {chatflow_name} (ID: {chatflow_id})")

    except Exception as e:
        logger.error(f"Setup failed: {e}")
        return False

    # Step 2: Create orchestrator
    orchestrator = MultiFlowOrchestrator(
        flowise,
        default_governance={
            "team": team,
            "project": project,
            "environment": "development",
        },
    )

    # Step 3: Execute different workflow types
    workflows = [
        create_document_analysis_workflow(chatflow_id),
        create_customer_service_workflow(chatflow_id),
        create_content_generation_workflow(chatflow_id),
    ]

    successful_workflows = 0

    for i, workflow in enumerate(workflows, 1):
        print(f"\n🔄 Step {i + 2}: Executing '{workflow.workflow_name}'...")
        print(f"   Session ID: {workflow.session_id}")
        print(f"   Customer: {workflow.customer_id}")
        print(f"   Steps: {len(workflow.steps)}")

        try:
            success = orchestrator.execute_workflow(workflow)

            if success:
                print("   ✅ Workflow completed successfully!")
                successful_workflows += 1
            else:
                print("   ⚠️ Workflow completed with some failures")

            # Display results summary
            print("   📊 Results:")
            print(f"      Duration: {workflow.duration_seconds:.2f}s")
            print(f"      Total cost: ${workflow.total_cost:.6f}")
            print(f"      Total tokens: {workflow.total_tokens}")
            print(
                f"      Steps completed: {sum(1 for r in workflow.results if r.success)}/{len(workflow.results)}"
            )

            # Show step details
            for result in workflow.results:
                status = "✅" if result.success else "❌"
                print(
                    f"         {status} {result.step_name}: {result.execution_time_ms}ms"
                )
                if result.error:
                    print(f"            Error: {result.error}")

        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            continue

    # Step 4: Summary
    print("\n📈 Orchestration Summary")
    print("=" * 30)
    print(f"Total workflows: {len(workflows)}")
    print(f"Successful workflows: {successful_workflows}")
    print(f"Success rate: {successful_workflows / len(workflows) * 100:.1f}%")

    if successful_workflows > 0:
        print("\n✅ Multi-flow orchestration working!")
        print("📊 Benefits demonstrated:")
        print("   • Session-based governance context")
        print("   • Cross-flow data sharing and templating")
        print("   • Dependency management between steps")
        print("   • Aggregated cost and usage tracking")
        print("   • Error handling and partial success scenarios")

    return successful_workflows > 0


def demonstrate_advanced_patterns():
    """Show advanced orchestration patterns."""

    print("\n🔬 Advanced Orchestration Patterns")
    print("=" * 50)

    patterns = [
        {
            "name": "Parallel Execution with Synchronization",
            "description": "Execute multiple flows in parallel, then synchronize results",
            "use_case": "Multi-modal content analysis (text + image + audio)",
        },
        {
            "name": "Conditional Flow Routing",
            "description": "Route to different flows based on previous results",
            "use_case": "Customer service escalation based on sentiment analysis",
        },
        {
            "name": "Dynamic Flow Selection",
            "description": "Choose flows at runtime based on business rules",
            "use_case": "A/B testing different AI models or prompts",
        },
        {
            "name": "Rollback and Retry Logic",
            "description": "Automatic rollback and retry with backoff strategies",
            "use_case": "Fault-tolerant document processing pipelines",
        },
        {
            "name": "Budget-Constrained Execution",
            "description": "Stop or switch to cheaper flows when budget limits hit",
            "use_case": "Cost-optimized content generation workflows",
        },
    ]

    for pattern in patterns:
        print(f"\n📋 {pattern['name']}:")
        print(f"   Description: {pattern['description']}")
        print(f"   Use Case: {pattern['use_case']}")

    print("\n💡 Implementation Tips:")
    print("   • Use session IDs to maintain context across flows")
    print("   • Implement dependency checking for complex workflows")
    print("   • Track costs at both step and session levels")
    print("   • Use governance overrides for step-specific attribution")
    print("   • Include error handling and partial success scenarios")
    print("   • Consider timeout and resource limits for long workflows")


def main():
    """Main example function."""

    try:
        # Run main demonstration
        success = demonstrate_multi_flow_orchestration()

        if success:
            # Show advanced patterns
            demonstrate_advanced_patterns()

            print("\n🎉 Multi-Flow Orchestration Example Complete!")
            print("=" * 55)
            print("✅ You've learned how to:")
            print("   • Orchestrate multiple Flowise flows in sequence")
            print("   • Manage workflow sessions with governance context")
            print("   • Handle dependencies between workflow steps")
            print("   • Aggregate costs and usage across multiple flows")
            print("   • Track detailed execution metrics and timing")

            print("\n📚 Next Steps:")
            print("   • Explore cost optimization (04_cost_optimization.py)")
            print("   • Try multi-tenant patterns (05_multi_tenant_saas.py)")
            print("   • Set up production monitoring (07_production_monitoring.py)")

        return success

    except Exception as e:
        logger.error(f"Example failed: {e}")
        return False


if __name__ == "__main__":
    try:
        success = main()
        exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⏹️ Example interrupted by user")
        exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        exit(1)
