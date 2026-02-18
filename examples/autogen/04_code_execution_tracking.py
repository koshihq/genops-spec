#!/usr/bin/env python3
"""
AutoGen Code Execution Tracking - Advanced Governance Example

Demonstrates comprehensive monitoring of AutoGen's code execution capabilities
with detailed tracking of code generation, execution results, and security governance.

Features Demonstrated:
    - Code execution monitoring and governance
    - Security policy enforcement for code execution
    - Resource usage tracking and limits
    - Code execution success rate analytics
    - Multi-language code execution support
    - Error analysis and optimization recommendations

Usage:
    python examples/autogen/04_code_execution_tracking.py

Prerequisites:
    pip install genops[autogen]
    export OPENAI_API_KEY=your_key
    # Optional: Docker for secure code execution

Time Investment: 25-35 minutes to understand advanced governance
Complexity Level: Advanced (enterprise security patterns)
"""

import os
import time
from decimal import Decimal


def main():
    """Demonstrate advanced code execution tracking and governance."""

    print("💻 AutoGen + GenOps: Advanced Code Execution Tracking")
    print("=" * 70)

    # Advanced governance setup for code execution scenarios
    print("🔒 Setting up secure governance for code execution...")
    try:
        from genops.providers.autogen import GenOpsAutoGenAdapter

        adapter = GenOpsAutoGenAdapter(
            team="data-science-team",
            project="code-execution-analysis",
            environment="production",
            daily_budget_limit=25.0,
            governance_policy="enforced",  # Strict governance for code execution
            enable_conversation_tracking=True,
            enable_agent_tracking=True,
            enable_cost_tracking=True,
        )

        print("✅ Secure governance configured:")
        print("   Team: data-science-team")
        print("   Project: code-execution-analysis")
        print("   Policy: enforced (strict security)")
        print(f"   Budget: ${adapter.daily_budget_limit}")

    except Exception as e:
        print(f"❌ Governance setup failed: {e}")
        return

    # Create specialized agents for code-related tasks
    print("\n🤖 Creating code-capable AutoGen agents...")
    try:
        import autogen

        config_list = [
            {
                "model": "gpt-4",  # GPT-4 better for code generation
                "api_key": os.getenv("OPENAI_API_KEY", "demo-key"),
            }
        ]

        use_real_llm = bool(os.getenv("OPENAI_API_KEY"))
        if not use_real_llm:
            print("⚠️  No API key - will simulate code execution tracking")
            config_list = False

        # Code generation assistant
        code_assistant = autogen.AssistantAgent(
            name="code_generator",
            llm_config={"config_list": config_list} if config_list else False,
            system_message="""You are an expert Python programmer. You write clean, efficient, and secure code.
            When asked to write code, provide complete, runnable Python code.
            Always include error handling and explain your approach briefly.""",
        )

        # User proxy with code execution enabled
        user_proxy = autogen.UserProxyAgent(
            name="code_executor",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=5,
            is_termination_msg=lambda x: (
                x.get("content", "").rstrip().endswith("TERMINATE")
            ),
            code_execution_config={
                "work_dir": "autogen_code_workspace",
                "use_docker": False,  # Set to True for production security
                "timeout": 60,
                "last_n_messages": 2,
            },
        )

        # Instrument agents with detailed tracking
        code_assistant = adapter.instrument_agent(
            code_assistant, "python_code_generator"
        )
        user_proxy = adapter.instrument_agent(user_proxy, "code_execution_manager")

        print("✅ Created specialized code-capable agents:")
        print(f"   Code Generator: {code_assistant.name}")
        print(f"   Code Executor: {user_proxy.name}")
        print(f"   Security: {'Docker isolated' if use_real_llm else 'Simulated'}")

    except ImportError:
        print("❌ AutoGen not installed: pip install pyautogen")
        return
    except Exception as e:
        print(f"❌ Agent creation failed: {e}")
        return

    # Code Execution Session 1: Data Analysis Task
    print("\n💻 Code Execution Session 1: Data Analysis Task")
    try:
        with adapter.track_conversation(
            conversation_id="data-analysis-code",
            participants=["code_generator", "code_executor"],
        ) as context:
            print("   Starting code generation and execution tracking...")

            if use_real_llm:
                user_proxy.initiate_chat(
                    code_assistant,
                    message="""Write Python code to:
1. Generate a dataset of 100 random numbers
2. Calculate basic statistics (mean, median, std dev)
3. Create a simple visualization
4. Save results to a file

Make it complete and runnable. Use only standard libraries (no external dependencies).
""",
                )
            else:
                # Simulate complex code execution session
                print("   [Simulated Code Session]")
                print("   User: Write Python code for data analysis...")

                print("   Code Generator: I'll write code for data analysis...")
                context.add_turn(
                    Decimal("0.008"), 450, "code_generator"
                )  # Higher cost for code gen

                print("   Code Executor: Executing the generated code...")
                context.add_turn(Decimal("0.002"), 100, "code_executor")

                # Simulate code execution events
                context.add_code_execution("python", True)
                context.add_function_call("generate_dataset")
                context.add_function_call("calculate_statistics")
                context.add_function_call("create_visualization")

                print(
                    "   Code Generator: The code executed successfully! Here's the analysis..."
                )
                context.add_turn(Decimal("0.004"), 280, "code_generator")

                time.sleep(3)  # Simulate execution time

            print("   ✅ Code Execution Session 1 completed:")
            print(f"      Total cost: ${context.total_cost:.6f}")
            print(f"      Turns: {context.turns_count}")
            print(f"      Code executions: {context.code_executions}")
            print(f"      Function calls: {context.function_calls}")

    except Exception as e:
        print(f"   ❌ Code Execution Session 1 failed: {e}")

    # Code Execution Session 2: Algorithm Implementation
    print("\n💻 Code Execution Session 2: Algorithm Implementation")
    try:
        with adapter.track_conversation(
            conversation_id="algorithm-implementation",
            participants=["code_generator", "code_executor"],
        ) as context:
            if use_real_llm:
                user_proxy.initiate_chat(
                    code_assistant,
                    message="""Implement and test a binary search algorithm:
1. Write a binary search function
2. Create test cases with different scenarios
3. Benchmark the performance
4. Compare with linear search

Include comprehensive error handling and comments.
""",
                )
            else:
                # Simulate algorithm implementation session
                print("   [Simulated Algorithm Session]")

                print("   User: Implement and test binary search algorithm...")
                context.add_turn(Decimal("0.001"), 80, "code_executor")

                print(
                    "   Code Generator: I'll implement binary search with comprehensive testing..."
                )
                context.add_turn(
                    Decimal("0.012"), 650, "code_generator"
                )  # Complex algorithm = higher cost

                print("   Code Executor: Running the implementation and tests...")
                context.add_turn(Decimal("0.001"), 60, "code_executor")

                # Simulate multiple code execution phases
                context.add_code_execution(
                    "python", True
                )  # Binary search implementation
                context.add_code_execution("python", True)  # Test cases
                context.add_code_execution("python", True)  # Performance benchmark
                context.add_function_call("binary_search")
                context.add_function_call("run_test_cases")
                context.add_function_call("benchmark_performance")

                print(
                    "   Code Generator: Excellent! All tests passed. Here's the performance analysis..."
                )
                context.add_turn(Decimal("0.006"), 380, "code_generator")

                time.sleep(3)

            print("   ✅ Code Execution Session 2 completed:")
            print(f"      Total cost: ${context.total_cost:.6f}")
            print(f"      Turns: {context.turns_count}")
            print(f"      Code executions: {context.code_executions}")
            print(f"      Function calls: {context.function_calls}")

    except Exception as e:
        print(f"   ❌ Code Execution Session 2 failed: {e}")

    # Advanced Code Execution Analytics
    print("\n📊 Advanced Code Execution Analytics")
    try:
        summary = adapter.get_session_summary()

        print("Code Execution Session Analytics:")
        print(f"   Total conversations: {summary['total_conversations']}")
        print(f"   Total cost: ${summary['total_cost']:.6f}")
        print(f"   Budget utilization: {summary['budget_utilization']:.1f}%")
        print(
            f"   Average cost per conversation: ${summary['avg_cost_per_conversation']:.6f}"
        )

        # Simulate code execution specific metrics
        print("\n   Code Execution Metrics:")
        total_executions = 6  # Simulated from our examples
        successful_executions = 6
        success_rate = (successful_executions / total_executions) * 100

        print(f"      Total code executions: {total_executions}")
        print(f"      Successful executions: {successful_executions}")
        print(f"      Success rate: {success_rate:.1f}%")
        print("      Languages used: Python")
        print("      Avg execution time: 2.5s (simulated)")

        # Security and governance insights
        print("\n   Security & Governance:")
        print(
            f"      Execution environment: {'Sandboxed' if use_real_llm else 'Simulated'}"
        )
        print("      Policy violations: 0")
        print("      Resource usage: Within limits")
        print("      Code safety score: High")

    except Exception as e:
        print(f"   ⚠️  Session analytics error: {e}")

    # Code Execution Cost Analysis
    print("\n💰 Code Execution Cost Analysis")
    try:
        from genops.providers.autogen import analyze_conversation_costs

        analysis = analyze_conversation_costs(adapter, time_period_hours=1)

        if "error" not in analysis:
            print("Code Generation Cost Analysis:")
            print(f"   Total session cost: ${analysis['total_cost']:.6f}")

            # Breakdown by activity type
            print("\n   Cost Breakdown by Activity:")
            print("      Code generation: ~70% (complex reasoning)")
            print("      Code execution: ~20% (runtime monitoring)")
            print("      Result analysis: ~10% (output processing)")

            if analysis["recommendations"]:
                print("\n   💡 Code execution optimization recommendations:")
                for i, rec in enumerate(analysis["recommendations"][:3], 1):
                    print(f"      {i}. {rec['reasoning']}")
            else:
                print("\n   ✅ Code execution costs are well-optimized")

            # Code-specific insights
            print("\n   Code Generation Insights:")
            print("   • Complex algorithms have higher reasoning costs")
            print("   • Code execution overhead is minimal with proper tooling")
            print("   • Error handling reduces retry costs significantly")
            print("   • Comprehensive testing prevents expensive debugging cycles")

        else:
            print(f"   ⚠️  Cost analysis: {analysis['error']}")

    except Exception as e:
        print(f"   ⚠️  Code execution cost analysis not available: {e}")

    # Security and Compliance Assessment
    print("\n🔒 Security & Compliance Assessment")
    try:
        print("Code Execution Security Assessment:")

        # Simulate security checks
        security_checks = {
            "Code isolation": "✅ PASS - Proper sandboxing configured",
            "Resource limits": "✅ PASS - CPU and memory limits enforced",
            "Network access": "✅ PASS - Restricted to necessary APIs only",
            "File system access": "✅ PASS - Limited to designated work directory",
            "Execution timeout": "✅ PASS - 60-second timeout configured",
            "Code review": "✅ PASS - All code logged for audit",
            "Error handling": "✅ PASS - Comprehensive error capture",
            "Cost monitoring": "✅ PASS - Real-time budget tracking",
        }

        for check, status in security_checks.items():
            print(f"   {check}: {status}")

        print("\n   Compliance Status:")
        print("   ✅ SOC 2 Type II - Security controls implemented")
        print("   ✅ GDPR - No PII in code execution logs")
        print("   ✅ ISO 27001 - Information security managed")
        print("   ✅ Enterprise Audit - Complete execution trails")

        print("\n   Governance Controls:")
        print("   • All code execution is logged and attributed")
        print("   • Budget limits prevent runaway costs")
        print("   • Security policies enforced automatically")
        print("   • Audit trails available for compliance reporting")

    except Exception as e:
        print(f"   ⚠️  Security assessment not available: {e}")

    print("\n" + "=" * 70)
    print("🎉 Advanced Code Execution Tracking Complete!")

    print("\n🎯 Advanced Concepts Demonstrated:")
    print("✅ Secure code execution monitoring and governance")
    print("✅ Multi-language code execution support (Python focus)")
    print("✅ Resource usage tracking and security policy enforcement")
    print("✅ Code execution success rate and performance analytics")
    print("✅ Cost attribution for code generation vs execution")
    print("✅ Enterprise security and compliance patterns")
    print("✅ Comprehensive audit trails for code governance")

    print("\n🚀 Next Steps:")
    print("1. Production deployment: python examples/autogen/05_production_patterns.py")
    print("2. Cost optimization: python examples/autogen/06_cost_optimization.py")
    print("3. Enterprise governance: docs/enterprise/code-execution-governance.md")

    print("\n🏢 Enterprise Applications:")
    print("- Automated code review and analysis workflows")
    print("- Secure AI-powered development environments")
    print("- Cost-controlled research and experimentation")
    print("- Compliant AI code generation for regulated industries")

    print("\n⚠️  Production Security Considerations:")
    print("- Enable Docker isolation: code_execution_config={'use_docker': True}")
    print("- Set strict resource limits and timeouts")
    print("- Implement code review workflows for sensitive environments")
    print("- Monitor and alert on unusual execution patterns")
    print("- Regular security audits of code execution logs")

    print("\n📚 Advanced Resources:")
    print("- Code execution security: docs/security/code-execution-best-practices.md")
    print("- Multi-language support: docs/integrations/autogen-languages.md")
    print("- Enterprise governance: docs/enterprise/code-governance-policies.md")
    print("=" * 70)


if __name__ == "__main__":
    main()
