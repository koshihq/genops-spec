#!/usr/bin/env python3
"""
Context Manager Patterns - 15 Minute Tutorial

Learn how to use context managers for unified governance across multi-step workflows.

Demonstrates:
- Governance context for workflows
- Automatic cost aggregation
- Multi-step operation tracking
- Error handling within contexts
- Nested context patterns

Prerequisites:
- export ANYSCALE_API_KEY='your-api-key'
- pip install genops-ai
"""

import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import List, Dict

from genops.providers.anyscale import (
    instrument_anyscale,
    calculate_completion_cost
)

# Check API key
if not os.getenv("ANYSCALE_API_KEY"):
    print("❌ ERROR: ANYSCALE_API_KEY not set")
    exit(1)

print("=" * 70)
print("GenOps Anyscale - Context Manager Patterns")
print("=" * 70 + "\n")


# Pattern 1: Basic Governance Context
print("=" * 70)
print("PATTERN 1: Basic Governance Context")
print("=" * 70 + "\n")

adapter = instrument_anyscale(
    team="ml-engineering",
    project="workflows"
)

print("Using governance context for a customer workflow...\n")

# All operations within context inherit governance attributes
with adapter.governance_context(
    customer_id="customer-abc-123",
    feature="document-processing",
    workflow_id="doc-proc-001"
) as context:

    print(f"📋 Context attributes: {list(context.keys())}\n")

    # Step 1: Classify document
    print("Step 1: Document classification...")
    response1 = adapter.completion_create(
        model="meta-llama/Llama-2-7b-chat-hf",
        messages=[{"role": "user", "content": "Classify: invoice document"}],
        max_tokens=20
    )
    print(f"   ✅ Classification: {response1['choices'][0]['message']['content'][:50]}...")

    # Step 2: Extract data
    print("\nStep 2: Data extraction...")
    response2 = adapter.completion_create(
        model="meta-llama/Llama-2-13b-chat-hf",
        messages=[{"role": "user", "content": "Extract invoice details"}],
        max_tokens=100
    )
    print(f"   ✅ Extraction: {response2['choices'][0]['message']['content'][:50]}...")

    # Step 3: Validate
    print("\nStep 3: Validation...")
    response3 = adapter.completion_create(
        model="meta-llama/Llama-2-7b-chat-hf",
        messages=[{"role": "user", "content": "Validate extracted data"}],
        max_tokens=50
    )
    print(f"   ✅ Validation: {response3['choices'][0]['message']['content'][:50]}...")

print("\n✅ Workflow complete - all operations tracked with unified governance\n")


# Pattern 2: Multi-Step Workflow with Cost Tracking
print("=" * 70)
print("PATTERN 2: Multi-Step Workflow with Cost Tracking")
print("=" * 70 + "\n")

@dataclass
class WorkflowTracker:
    """Track workflow execution and costs."""

    workflow_id: str
    steps: List[Dict] = field(default_factory=list)
    total_cost: float = 0.0
    total_tokens: int = 0

    def add_step(self, step_name: str, response: Dict, model: str):
        """Add completed step with cost calculation."""
        usage = response['usage']
        cost = calculate_completion_cost(
            model=model,
            input_tokens=usage['prompt_tokens'],
            output_tokens=usage['completion_tokens']
        )

        self.steps.append({
            'step': step_name,
            'model': model,
            'tokens': usage['total_tokens'],
            'cost': cost
        })

        self.total_cost += cost
        self.total_tokens += usage['total_tokens']

    def print_summary(self):
        """Print workflow summary."""
        print(f"\n{'='*70}")
        print(f"WORKFLOW SUMMARY: {self.workflow_id}")
        print(f"{'='*70}")

        for i, step in enumerate(self.steps, 1):
            print(f"\nStep {i}: {step['step']}")
            print(f"   Model: {step['model']}")
            print(f"   Tokens: {step['tokens']}")
            print(f"   Cost: ${step['cost']:.8f}")

        print(f"\n{'='*70}")
        print(f"Total Steps: {len(self.steps)}")
        print(f"Total Tokens: {self.total_tokens}")
        print(f"Total Cost: ${self.total_cost:.6f}")
        print(f"Avg Cost/Step: ${self.total_cost/len(self.steps):.8f}")
        print(f"{'='*70}\n")


workflow_tracker = WorkflowTracker(workflow_id="sentiment-analysis-001")

print("Executing multi-step sentiment analysis workflow...\n")

with adapter.governance_context(
    customer_id="analytics-customer",
    feature="sentiment-analysis",
    workflow_id=workflow_tracker.workflow_id
):

    # Step 1: Preprocessing
    print("Step 1: Text preprocessing...")
    response = adapter.completion_create(
        model="meta-llama/Llama-2-7b-chat-hf",
        messages=[{"role": "user", "content": "Clean and normalize: Customer feedback text"}],
        max_tokens=50
    )
    workflow_tracker.add_step("Preprocessing", response, "meta-llama/Llama-2-7b-chat-hf")
    print("   ✅ Preprocessing complete")

    # Step 2: Sentiment classification
    print("\nStep 2: Sentiment classification...")
    response = adapter.completion_create(
        model="meta-llama/Llama-2-13b-chat-hf",
        messages=[{"role": "user", "content": "Classify sentiment: This product is amazing!"}],
        max_tokens=30
    )
    workflow_tracker.add_step("Classification", response, "meta-llama/Llama-2-13b-chat-hf")
    print("   ✅ Classification complete")

    # Step 3: Entity extraction
    print("\nStep 3: Entity extraction...")
    response = adapter.completion_create(
        model="meta-llama/Llama-2-13b-chat-hf",
        messages=[{"role": "user", "content": "Extract entities: product names, features"}],
        max_tokens=50
    )
    workflow_tracker.add_step("Entity Extraction", response, "meta-llama/Llama-2-13b-chat-hf")
    print("   ✅ Entity extraction complete")

    # Step 4: Summary generation
    print("\nStep 4: Summary generation...")
    response = adapter.completion_create(
        model="meta-llama/Llama-2-7b-chat-hf",
        messages=[{"role": "user", "content": "Summarize sentiment analysis results"}],
        max_tokens=80
    )
    workflow_tracker.add_step("Summary", response, "meta-llama/Llama-2-7b-chat-hf")
    print("   ✅ Summary complete")

workflow_tracker.print_summary()


# Pattern 3: Error Handling with Context
print("=" * 70)
print("PATTERN 3: Error Handling with Context")
print("=" * 70 + "\n")

@contextmanager
def safe_workflow_context(adapter, **governance_attrs):
    """Context manager with error handling."""
    print(f"🚀 Starting workflow with governance: {list(governance_attrs.keys())}")

    try:
        with adapter.governance_context(**governance_attrs) as ctx:
            yield ctx
        print("✅ Workflow completed successfully")

    except Exception as e:
        print(f"❌ Workflow failed: {e}")
        print("   Governance tracking preserved for debugging")
        # In production: log error with governance context for debugging
        raise


print("Testing error handling in workflow context...\n")

try:
    with safe_workflow_context(
        adapter,
        customer_id="error-test-customer",
        workflow_id="error-workflow-001"
    ) as ctx:

        # Successful operation
        print("Step 1: Successful operation...")
        response = adapter.completion_create(
            model="meta-llama/Llama-2-7b-chat-hf",
            messages=[{"role": "user", "content": "Test successful operation"}],
            max_tokens=20
        )
        print("   ✅ Operation successful\n")

        # This would raise an error in real scenario
        # raise Exception("Simulated error")

except Exception as e:
    print(f"Caught exception: {e}")

print()


# Pattern 4: Nested Context for Complex Workflows
print("=" * 70)
print("PATTERN 4: Nested Context for Complex Workflows")
print("=" * 70 + "\n")

print("Executing nested workflow: Document processing with sub-workflows...\n")

# Outer workflow: Document processing
with adapter.governance_context(
    customer_id="enterprise-customer",
    feature="document-processing",
    workflow_id="doc-master-001"
) as outer_ctx:

    print("📄 Main workflow: Document processing")
    print(f"   Context: {list(outer_ctx.keys())}\n")

    # Sub-workflow 1: Text extraction
    print("  → Sub-workflow 1: Text extraction")
    with adapter.governance_context(
        sub_workflow="text-extraction",
        workflow_step="1"
    ) as sub_ctx1:

        response = adapter.completion_create(
            model="meta-llama/Llama-2-7b-chat-hf",
            messages=[{"role": "user", "content": "Extract text from PDF"}],
            max_tokens=50
        )
        print(f"     ✅ Extracted {response['usage']['total_tokens']} tokens\n")

    # Sub-workflow 2: Translation
    print("  → Sub-workflow 2: Translation")
    with adapter.governance_context(
        sub_workflow="translation",
        workflow_step="2"
    ) as sub_ctx2:

        response = adapter.completion_create(
            model="meta-llama/Llama-2-13b-chat-hf",
            messages=[{"role": "user", "content": "Translate to Spanish"}],
            max_tokens=100
        )
        print(f"     ✅ Translated {response['usage']['total_tokens']} tokens\n")

    # Sub-workflow 3: Summarization
    print("  → Sub-workflow 3: Summarization")
    with adapter.governance_context(
        sub_workflow="summarization",
        workflow_step="3"
    ) as sub_ctx3:

        response = adapter.completion_create(
            model="meta-llama/Llama-2-7b-chat-hf",
            messages=[{"role": "user", "content": "Summarize document"}],
            max_tokens=80
        )
        print(f"     ✅ Summarized {response['usage']['total_tokens']} tokens\n")

print("✅ Nested workflow complete - all sub-workflows tracked under main workflow\n")


# Pattern 5: Batch Processing with Context
print("=" * 70)
print("PATTERN 5: Batch Processing with Context")
print("=" * 70 + "\n")

documents = [
    "Document 1: Product review",
    "Document 2: Customer feedback",
    "Document 3: Support ticket",
    "Document 4: Sales inquiry",
    "Document 5: Feature request",
]

print(f"Processing batch of {len(documents)} documents...\n")

batch_costs = []

with adapter.governance_context(
    customer_id="batch-processing-customer",
    feature="batch-analysis",
    workflow_id="batch-001"
):

    for i, doc in enumerate(documents, 1):
        print(f"Processing document {i}/{len(documents)}...")

        response = adapter.completion_create(
            model="meta-llama/Llama-2-7b-chat-hf",
            messages=[{"role": "user", "content": f"Analyze: {doc}"}],
            max_tokens=50,
            document_id=f"doc-{i}"  # Additional tracking per document
        )

        usage = response['usage']
        cost = calculate_completion_cost(
            model="meta-llama/Llama-2-7b-chat-hf",
            input_tokens=usage['prompt_tokens'],
            output_tokens=usage['completion_tokens']
        )

        batch_costs.append(cost)
        print(f"   ✅ Processed: ${cost:.8f}\n")

print("=" * 70)
print("BATCH PROCESSING SUMMARY")
print("=" * 70)
print(f"Documents processed: {len(documents)}")
print(f"Total cost: ${sum(batch_costs):.6f}")
print(f"Avg cost/document: ${sum(batch_costs)/len(batch_costs):.8f}")
print(f"Min cost: ${min(batch_costs):.8f}")
print(f"Max cost: ${max(batch_costs):.8f}")
print("=" * 70 + "\n")


# Summary
print("=" * 70)
print("✅ Context manager patterns demonstration complete!")
print("=" * 70)

print("\n🎯 KEY BENEFITS OF CONTEXT MANAGERS:")
print("   ✅ Unified governance for multi-step workflows")
print("   ✅ Automatic cost aggregation across steps")
print("   ✅ Consistent attribute propagation")
print("   ✅ Error handling with context preservation")
print("   ✅ Nested workflows with hierarchical tracking")
print("   ✅ Clean code structure for complex operations")
print()

print("💡 WHEN TO USE CONTEXT MANAGERS:")
print("   • Multi-step workflows (RAG pipelines, document processing)")
print("   • Customer-specific operations (unified attribution)")
print("   • Batch processing (consistent governance)")
print("   • Error-prone operations (preserve context for debugging)")
print("   • Complex nested workflows (hierarchical tracking)")
print()

print("📚 BEST PRACTICES:")
print("   • Use outer context for customer/workflow-level attributes")
print("   • Use nested contexts for sub-workflow tracking")
print("   • Track costs within contexts for workflow-level billing")
print("   • Preserve governance attributes for error debugging")
print("   • Use workflow_id for end-to-end tracing")
print()

print("🔗 INTEGRATION:")
print("   • Query observability platform by workflow_id")
print("   • Aggregate costs by customer_id + workflow")
print("   • Trace errors through governance attributes")
print("   • Build workflow-level dashboards")
print()

print("📚 Next Steps:")
print("   • Combine patterns from all examples")
print("   • Integrate with your production workflows")
print("   • Set up observability dashboards")
print("   • Monitor costs and performance")
