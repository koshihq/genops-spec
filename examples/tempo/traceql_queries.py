"""
TraceQL query examples for Grafana Tempo.

This example demonstrates:
- TraceQL query syntax for GenOps governance attributes
- Cost analysis queries
- Team and customer attribution queries
- Performance analysis with TraceQL

Prerequisites:
    - Tempo 2.0+ running at http://localhost:3200 (TraceQL support)
    - Sample traces already exported (run direct_export.py first)
"""

from typing import Any

import requests


def execute_traceql_query(query: str, limit: int = 10) -> dict[str, Any]:
    """
    Execute a TraceQL query against Tempo.

    Args:
        query: TraceQL query string
        limit: Maximum number of results

    Returns:
        Query results as dictionary
    """
    response = requests.get(
        "http://localhost:3200/api/search",
        params={"q": query, "limit": limit},
        timeout=10,
    )

    if response.status_code == 200:
        return response.json()
    else:
        return {"error": f"HTTP {response.status_code}", "message": response.text}


def print_query_results(title: str, query: str, results: dict[str, Any]):
    """Pretty print query results."""
    print(f"\n{'=' * 70}")
    print(f"Query: {title}")
    print(f"{'=' * 70}")
    print(f"TraceQL: {query}")
    print("-" * 70)

    if "error" in results:
        print(f"❌ Error: {results['error']}")
        print(f"   {results.get('message', '')}")
        return

    traces = results.get("traces", [])
    print(f"Found {len(traces)} traces")

    for i, trace in enumerate(traces[:5], 1):  # Show first 5
        trace_id = trace.get("traceID", "unknown")
        root_service = trace.get("rootServiceName", "unknown")
        duration_ms = trace.get("durationMs", 0)

        print(f"\n{i}. Trace ID: {trace_id[:16]}...")
        print(f"   Service: {root_service}")
        print(f"   Duration: {duration_ms}ms")

        # Show span attributes if available
        if "spanSet" in trace:
            spans = trace["spanSet"].get("spans", [])
            if spans:
                span = spans[0]
                attrs = span.get("attributes", [])
                if attrs:
                    print("   Attributes:")
                    for attr in attrs[:5]:  # Show first 5 attributes
                        print(f"     - {attr.get('key')}: {attr.get('value')}")


def main():
    """
    Run comprehensive TraceQL query examples.
    """
    print("=" * 70)
    print("Grafana Tempo TraceQL Query Examples")
    print("=" * 70)

    # ========================================================================
    # Basic Queries
    # ========================================================================

    print("\n" + "=" * 70)
    print("SECTION 1: Basic Queries")
    print("=" * 70)

    # Query 1: All traces
    results = execute_traceql_query("{}", limit=10)
    print_query_results("All Recent Traces", "{}", results)

    # Query 2: Traces by service name
    results = execute_traceql_query('{resource.service.name = "genops-ai"}', limit=10)
    print_query_results(
        "Traces by Service Name", '{resource.service.name = "genops-ai"}', results
    )

    # Query 3: Traces with duration > 500ms
    results = execute_traceql_query("{duration > 500ms}", limit=10)
    print_query_results("Slow Traces (>500ms)", "{duration > 500ms}", results)

    # ========================================================================
    # Governance Attribute Queries
    # ========================================================================

    print("\n" + "=" * 70)
    print("SECTION 2: Governance Attribute Queries")
    print("=" * 70)

    # Query 4: Traces by team
    results = execute_traceql_query('{.team = "platform-engineering"}', limit=10)
    print_query_results("Traces by Team", '{.team = "platform-engineering"}', results)

    # Query 5: Traces by customer
    results = execute_traceql_query('{.customer_id = "enterprise-123"}', limit=10)
    print_query_results(
        "Traces by Customer ID", '{.customer_id = "enterprise-123"}', results
    )

    # Query 6: Traces by project
    results = execute_traceql_query('{.project = "ai-assistant"}', limit=10)
    print_query_results("Traces by Project", '{.project = "ai-assistant"}', results)

    # Query 7: Traces by environment
    results = execute_traceql_query(
        '{.deployment.environment = "production"}', limit=10
    )
    print_query_results(
        "Production Traces", '{.deployment.environment = "production"}', results
    )

    # ========================================================================
    # Cost Analysis Queries
    # ========================================================================

    print("\n" + "=" * 70)
    print("SECTION 3: Cost Analysis Queries")
    print("=" * 70)

    # Query 8: High cost traces
    results = execute_traceql_query("{.genops.cost.total_cost > 0.10}", limit=10)
    print_query_results(
        "High Cost Traces (>$0.10)", "{.genops.cost.total_cost > 0.10}", results
    )

    # Query 9: High token usage
    results = execute_traceql_query("{.genops.cost.total_tokens > 2000}", limit=10)
    print_query_results(
        "High Token Usage (>2000 tokens)", "{.genops.cost.total_tokens > 2000}", results
    )

    # Query 10: Cost by provider
    results = execute_traceql_query('{.genops.provider = "openai"}', limit=10)
    print_query_results("OpenAI Traces", '{.genops.provider = "openai"}', results)

    # ========================================================================
    # Complex Queries
    # ========================================================================

    print("\n" + "=" * 70)
    print("SECTION 4: Complex Multi-Condition Queries")
    print("=" * 70)

    # Query 11: Expensive slow traces for specific customer
    complex_query = '{duration > 1s && .genops.cost.total_cost > 0.05 && .customer_id = "enterprise-123"}'
    results = execute_traceql_query(complex_query, limit=10)
    print_query_results("Expensive Slow Traces for Customer", complex_query, results)

    # Query 12: Production traces with errors
    error_query = '{.deployment.environment = "production" && status = error}'
    results = execute_traceql_query(error_query, limit=10)
    print_query_results("Production Errors", error_query, results)

    # Query 13: Multi-team query
    team_query = '{.team = "platform-engineering" || .team = "ml-research"}'
    results = execute_traceql_query(team_query, limit=10)
    print_query_results("Multiple Teams", team_query, results)

    # ========================================================================
    # Aggregation Examples (via API)
    # ========================================================================

    print("\n" + "=" * 70)
    print("SECTION 5: Tag Analysis")
    print("=" * 70)

    print("\nAvailable Trace Tags:")
    print("-" * 70)

    try:
        tags_response = requests.get("http://localhost:3200/api/search/tags", timeout=5)
        if tags_response.status_code == 200:
            tags_data = tags_response.json()

            if isinstance(tags_data, dict) and "tagNames" in tags_data:
                tag_names = tags_data["tagNames"]
                print(f"Found {len(tag_names)} tags:")

                for tag in sorted(tag_names)[:20]:  # Show first 20
                    print(f"  - {tag}")

                    # Get values for GenOps tags
                    if tag.startswith("genops") or tag in [
                        "team",
                        "customer_id",
                        "project",
                    ]:
                        try:
                            values_response = requests.get(
                                f"http://localhost:3200/api/search/tag/{tag}/values",
                                timeout=5,
                            )
                            if values_response.status_code == 200:
                                values_data = values_response.json()
                                if (
                                    isinstance(values_data, dict)
                                    and "tagValues" in values_data
                                ):
                                    values = values_data["tagValues"][:5]  # First 5
                                    if values:
                                        print(f"    Values: {', '.join(values)}")
                        except Exception:
                            pass
            else:
                print("No tags found (no traces ingested yet)")
        else:
            print(f"❌ Could not retrieve tags: HTTP {tags_response.status_code}")

    except Exception as e:
        print(f"❌ Error retrieving tags: {e}")

    # ========================================================================
    # Summary
    # ========================================================================

    print("\n" + "=" * 70)
    print("TraceQL Query Examples Summary")
    print("=" * 70)
    print("""
TraceQL provides powerful querying for:

1. **Governance Tracking**
   - Team attribution: {.team = "platform-engineering"}
   - Customer tracking: {.customer_id = "enterprise-123"}
   - Project filtering: {.project = "ai-assistant"}

2. **Cost Analysis**
   - High cost traces: {.genops.cost.total_cost > 0.10}
   - Token usage: {.genops.cost.total_tokens > 2000}
   - Provider breakdown: {.genops.provider = "openai"}

3. **Performance Analysis**
   - Slow traces: {duration > 1s}
   - Error traces: {status = error}
   - Complex conditions: {duration > 1s && .cost > 0.05}

4. **Multi-Dimensional Filtering**
   - Combine duration, cost, team, customer
   - Environment-specific queries
   - Provider-specific analysis

For more TraceQL syntax, see:
https://grafana.com/docs/tempo/latest/traceql/
    """)

    print("=" * 70)
    print("✅ TraceQL query examples completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
