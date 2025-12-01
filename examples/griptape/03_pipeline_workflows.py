#!/usr/bin/env python3
"""
Example 03: Pipeline Workflows with GenOps Governance

Complexity: ⭐⭐ Intermediate

This example demonstrates how GenOps provides comprehensive governance for
Griptape Pipeline workflows, including task-level cost attribution,
multi-step governance tracking, and workflow performance monitoring.

Prerequisites:
- Griptape framework installed (pip install griptape)
- GenOps installed (pip install genops)
- OpenAI API key set in environment

Usage:
    python 03_pipeline_workflows.py

Environment Variables:
    OPENAI_API_KEY: Your OpenAI API key
    GENOPS_TEAM: Team identifier for governance
    GENOPS_PROJECT: Project identifier
"""

import os
import logging
from griptape.structures import Pipeline
from griptape.tasks import PromptTask, TextSummaryTask
from griptape.rules import Rule

# GenOps imports for pipeline tracking
from genops.providers.griptape import auto_instrument
from genops.providers.griptape.registration import is_instrumented

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_analysis_pipeline():
    """Create a multi-step analysis pipeline."""
    
    pipeline = Pipeline(
        tasks=[
            PromptTask(
                id="research",
                prompt="""Research the current state of AI governance in enterprise organizations.
                
                Focus on:
                1. Key challenges organizations face
                2. Current best practices being adopted
                3. Regulatory considerations
                
                Input data: {{ input }}""",
                rules=[
                    Rule("Provide structured, well-researched information"),
                    Rule("Use specific examples where possible"),
                    Rule("Keep response comprehensive but focused")
                ]
            ),
            PromptTask(
                id="analysis",
                prompt="""Analyze the research findings and identify key patterns:
                
                Research data: {{ research.output }}
                
                Provide:
                1. Top 3 governance challenges identified
                2. Most effective practices being adopted
                3. Gaps in current approaches""",
                rules=[
                    Rule("Focus on actionable insights"),
                    Rule("Prioritize findings by importance"),
                    Rule("Support conclusions with research data")
                ]
            ),
            PromptTask(
                id="recommendations",
                prompt="""Based on the analysis, create specific recommendations:
                
                Analysis: {{ analysis.output }}
                
                Provide:
                1. 3 concrete recommendations for improving AI governance
                2. Implementation timeline for each
                3. Expected ROI and risk mitigation benefits""",
                rules=[
                    Rule("Make recommendations specific and actionable"),
                    Rule("Include implementation considerations"),
                    Rule("Focus on practical business value")
                ]
            ),
            TextSummaryTask(
                id="executive_summary",
                prompt="""Create an executive summary of the complete analysis:
                
                Research: {{ research.output }}
                Analysis: {{ analysis.output }}
                Recommendations: {{ recommendations.output }}
                
                Summary should be suitable for C-level executives."""
            )
        ]
    )
    
    return pipeline

def create_content_pipeline():
    """Create a content generation pipeline."""
    
    pipeline = Pipeline(
        tasks=[
            PromptTask(
                id="outline",
                prompt="""Create a detailed outline for a blog post about:
                
                Topic: {{ input }}
                
                Include:
                1. Compelling headline
                2. 4-5 main sections with subpoints
                3. Key takeaways for readers""",
                rules=[
                    Rule("Make outline engaging and well-structured"),
                    Rule("Focus on reader value"),
                    Rule("Include actionable insights")
                ]
            ),
            PromptTask(
                id="introduction",
                prompt="""Write a compelling introduction based on this outline:
                
                Outline: {{ outline.output }}
                
                The introduction should:
                1. Hook the reader immediately
                2. Clearly state the value proposition
                3. Preview what they'll learn""",
                rules=[
                    Rule("Keep introduction concise but engaging"),
                    Rule("Use conversational tone"),
                    Rule("Create curiosity about the content")
                ]
            ),
            PromptTask(
                id="main_content",
                prompt="""Write the main content sections based on:
                
                Outline: {{ outline.output }}
                Introduction: {{ introduction.output }}
                
                Create comprehensive content for each main section.""",
                rules=[
                    Rule("Provide practical, actionable advice"),
                    Rule("Use examples and case studies"),
                    Rule("Maintain consistent voice throughout")
                ]
            )
        ]
    )
    
    return pipeline

def main():
    """Pipeline workflows with governance demonstration."""
    
    print("🤖 GenOps + Griptape - Pipeline Workflows Example")
    print("=" * 70)
    
    try:
        # Check environment
        openai_key = os.getenv('OPENAI_API_KEY')
        if not openai_key:
            print("❌ Error: OPENAI_API_KEY environment variable is required")
            return False
        
        team = os.getenv('GENOPS_TEAM', 'your-team')
        project = os.getenv('GENOPS_PROJECT', 'griptape-demo')
        
        # Enable GenOps governance
        print("📊 Enabling GenOps governance for pipeline workflows...")
        adapter = auto_instrument(
            team=team,
            project=project,
            environment="development",
            enable_cost_tracking=True,
            enable_performance_monitoring=True
        )
        
        print(f"✅ Governance enabled for team '{team}', project '{project}'")
        
        # === PIPELINE 1: Analysis Workflow ===
        print("\n📋 PIPELINE 1: Multi-Step Analysis Workflow")
        print("-" * 60)
        
        print("🚀 Creating analysis pipeline with 4 tasks...")
        analysis_pipeline = create_analysis_pipeline()
        
        print("📝 Pipeline structure:")
        for i, task in enumerate(analysis_pipeline.tasks, 1):
            print(f"  {i}. {task.id}: {task.__class__.__name__}")
        
        print("\n⚡ Executing analysis pipeline...")
        initial_spending = adapter.get_daily_spending()
        
        analysis_result = analysis_pipeline.run({
            "input": "Current state of AI governance in Fortune 500 companies, focusing on cost management, ethical AI practices, and regulatory compliance."
        })
        
        analysis_spending = adapter.get_daily_spending()
        analysis_cost = analysis_spending - initial_spending
        
        print(f"✅ Analysis pipeline completed!")
        print(f"💰 Pipeline cost: ${analysis_cost:.6f}")
        print(f"📊 Tasks executed: {len(analysis_pipeline.tasks)}")
        
        # Show final task output (executive summary)
        if hasattr(analysis_result, 'output') and analysis_result.output:
            summary_preview = str(analysis_result.output.value)[:200]
            print(f"📝 Executive Summary (preview): {summary_preview}...")
        
        # === PIPELINE 2: Content Generation Workflow ===
        print("\n📋 PIPELINE 2: Content Generation Workflow")
        print("-" * 60)
        
        print("🚀 Creating content generation pipeline...")
        content_pipeline = create_content_pipeline()
        
        print("📝 Pipeline structure:")
        for i, task in enumerate(content_pipeline.tasks, 1):
            print(f"  {i}. {task.id}: {task.__class__.__name__}")
        
        print("\n⚡ Executing content generation pipeline...")
        
        content_result = content_pipeline.run({
            "input": "The Future of AI Governance: Building Sustainable and Ethical AI Operations at Scale"
        })
        
        final_spending = adapter.get_daily_spending()
        content_cost = final_spending - analysis_spending
        total_cost = final_spending - initial_spending
        
        print(f"✅ Content pipeline completed!")
        print(f"💰 Pipeline cost: ${content_cost:.6f}")
        print(f"📊 Tasks executed: {len(content_pipeline.tasks)}")
        
        # === GOVERNANCE SUMMARY ===
        print("\n📊 Governance & Cost Analysis")
        print("-" * 60)
        
        print(f"💰 Cost Breakdown:")
        print(f"  Analysis Pipeline: ${analysis_cost:.6f}")
        print(f"  Content Pipeline:  ${content_cost:.6f}")
        print(f"  Total Session:     ${total_cost:.6f}")
        
        print(f"📈 Workflow Efficiency:")
        tasks_per_dollar_analysis = len(analysis_pipeline.tasks) / analysis_cost if analysis_cost > 0 else 0
        tasks_per_dollar_content = len(content_pipeline.tasks) / content_cost if content_cost > 0 else 0
        print(f"  Analysis Pipeline: {tasks_per_dollar_analysis:.0f} tasks per $0.001")
        print(f"  Content Pipeline:  {tasks_per_dollar_content:.0f} tasks per $0.001")
        
        # Budget compliance
        budget_status = adapter.check_budget_compliance()
        print(f"💳 Budget Status: {budget_status['status']}")
        
        # Governance attributes
        print(f"👥 Governance Attribution:")
        print(f"  Team: {adapter.governance_attrs.team}")
        print(f"  Project: {adapter.governance_attrs.project}")
        print(f"  Environment: {adapter.governance_attrs.environment}")
        
        print("\n🎉 Pipeline Workflows Example Complete!")
        print("\n✨ Key Takeaways:")
        print("  1. ✅ Multi-step pipelines automatically tracked")
        print("  2. ✅ Task-level cost attribution and monitoring")
        print("  3. ✅ Workflow performance metrics captured")
        print("  4. ✅ Complex reasoning chains fully governed")
        print("  5. ✅ Budget compliance monitoring across workflows")
        
        print("\n🚀 Next Steps:")
        print("  • Try parallel workflows with concurrent task execution")
        print("  • Explore memory-enhanced pipelines with conversation state")
        print("  • Set up production deployment with observability dashboards")
        print("  • Implement budget controls and cost optimization strategies")
        
        return True
        
    except ImportError as e:
        if "griptape" in str(e):
            print("❌ Error: Griptape not installed")
            print("   Install with: pip install griptape")
        elif "genops" in str(e):
            print("❌ Error: GenOps not installed")
            print("   Install with: pip install genops")
        else:
            print(f"❌ Import error: {e}")
        return False
        
    except Exception as e:
        logger.error(f"Pipeline workflows example failed: {e}")
        print(f"\n❌ Error occurred: {e}")
        print("\n🔧 Troubleshooting Tips:")
        print("  • Check your API keys are valid and have sufficient credits")
        print("  • Verify network connectivity for API calls")
        print("  • Ensure Griptape and GenOps are properly installed")
        print("  • Run setup validation script for detailed diagnostics")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)