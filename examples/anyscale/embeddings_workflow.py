#!/usr/bin/env python3
"""
Embeddings Workflow - 15 Minute Tutorial

Learn how to generate embeddings for RAG (Retrieval-Augmented Generation) pipelines.

Demonstrates:
- Embedding generation with cost tracking
- Batch processing optimization
- Vector database integration patterns
- Governance for embedding operations

Prerequisites:
- export ANYSCALE_API_KEY='your-api-key'
- pip install genops-ai
"""

import os
from genops.providers.anyscale import (
    instrument_anyscale,
    calculate_embedding_cost,
    get_model_pricing
)

# Check API key
if not os.getenv("ANYSCALE_API_KEY"):
    print("❌ ERROR: ANYSCALE_API_KEY not set")
    exit(1)

print("=" * 70)
print("GenOps Anyscale - Embeddings Workflow")
print("=" * 70 + "\n")

# Create adapter with governance
adapter = instrument_anyscale(
    team="ml-engineering",
    project="rag-pipeline",
    environment="development"
)

# Sample documents for RAG knowledge base
documents = [
    "GenOps AI provides governance for AI systems built on OpenTelemetry.",
    "Anyscale Endpoints offers managed LLM inference with OpenAI-compatible APIs.",
    "RAG (Retrieval-Augmented Generation) combines vector search with LLM generation.",
    "Embeddings convert text into high-dimensional vectors for semantic search.",
    "Cost tracking is essential for production AI systems at scale.",
]

print("📚 Sample Knowledge Base:")
for i, doc in enumerate(documents, 1):
    print(f"   {i}. {doc}")
print()

# Get embedding model info
embedding_model = "thenlper/gte-large"
pricing = get_model_pricing(embedding_model)

print(f"🔧 Embedding Model: {embedding_model}")
print(f"Pricing: ${pricing.input_cost_per_million}/M tokens")
print(f"Dimension: 1024 (standard for gte-large)")
print()

# Example 1: Single document embedding
print("=" * 70)
print("EXAMPLE 1: Single Document Embedding")
print("=" * 70 + "\n")

single_doc = documents[0]
print(f"Document: \"{single_doc}\"")

response = adapter.embeddings_create(
    model=embedding_model,
    input=single_doc,
    customer_id="knowledge-base-v1"  # Track by knowledge base version
)

embedding = response['data'][0]['embedding']
tokens_used = response['usage']['total_tokens']

cost = calculate_embedding_cost(
    model=embedding_model,
    tokens=tokens_used
)

print(f"✅ Embedding generated:")
print(f"   Dimension: {len(embedding)}")
print(f"   First 5 values: {embedding[:5]}")
print(f"   Tokens: {tokens_used}")
print(f"   Cost: ${cost:.8f}")
print()

# Example 2: Batch embedding
print("=" * 70)
print("EXAMPLE 2: Batch Embedding (Optimized)")
print("=" * 70 + "\n")

print("Processing 5 documents in single batch request...")

batch_response = adapter.embeddings_create(
    model=embedding_model,
    input=documents,  # List of documents
    customer_id="knowledge-base-v1"
)

batch_embeddings = [item['embedding'] for item in batch_response['data']]
batch_tokens = batch_response['usage']['total_tokens']
batch_cost = calculate_embedding_cost(
    model=embedding_model,
    tokens=batch_tokens
)

print(f"✅ Batch processing complete:")
print(f"   Documents processed: {len(batch_embeddings)}")
print(f"   Total tokens: {batch_tokens}")
print(f"   Total cost: ${batch_cost:.8f}")
print(f"   Average cost per doc: ${batch_cost / len(documents):.8f}")
print()

# Cost comparison: batch vs individual
individual_cost_estimate = cost * len(documents)
savings = individual_cost_estimate - batch_cost
savings_pct = (savings / individual_cost_estimate) * 100 if individual_cost_estimate > 0 else 0

print("💡 Batch Processing Benefits:")
print(f"   Individual requests (5x): ${individual_cost_estimate:.8f}")
print(f"   Batch request (1x): ${batch_cost:.8f}")
print(f"   Savings: {savings_pct:.1f}% (${savings:.8f})")
print()

# Example 3: Semantic search simulation
print("=" * 70)
print("EXAMPLE 3: Semantic Search Simulation")
print("=" * 70 + "\n")

query = "How do I track costs for my AI system?"
print(f"Query: \"{query}\"\n")

# Generate query embedding
query_response = adapter.embeddings_create(
    model=embedding_model,
    input=query,
    feature="semantic-search"  # Track by feature
)

query_embedding = query_response['data'][0]['embedding']
query_tokens = query_response['usage']['total_tokens']
query_cost = calculate_embedding_cost(embedding_model, query_tokens)

print(f"Query embedding generated:")
print(f"   Tokens: {query_tokens}")
print(f"   Cost: ${query_cost:.8f}")
print()

# Simulate cosine similarity calculation
def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors."""
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    magnitude1 = sum(a * a for a in vec1) ** 0.5
    magnitude2 = sum(b * b for b in vec2) ** 0.5
    return dot_product / (magnitude1 * magnitude2) if magnitude1 and magnitude2 else 0

print("🔍 Finding most relevant documents:")
similarities = []
for i, doc_embedding in enumerate(batch_embeddings):
    similarity = cosine_similarity(query_embedding, doc_embedding)
    similarities.append((i, similarity, documents[i]))

# Sort by similarity (descending)
similarities.sort(key=lambda x: x[1], reverse=True)

print("\nTop 3 most relevant documents:")
for rank, (idx, similarity, doc) in enumerate(similarities[:3], 1):
    print(f"{rank}. [Score: {similarity:.4f}] {doc}")

print()

# Total cost summary
total_cost = batch_cost + query_cost
print("=" * 70)
print("COST SUMMARY")
print("=" * 70)
print(f"Knowledge base embedding (5 docs): ${batch_cost:.8f}")
print(f"Query embedding (1 query): ${query_cost:.8f}")
print(f"Total workflow cost: ${total_cost:.8f}")
print()

# Scale projection
print("📈 AT SCALE:")
kb_sizes = [100, 1000, 10000]
queries_per_day = 1000

for kb_size in kb_sizes:
    kb_cost = (batch_cost / len(documents)) * kb_size
    daily_query_cost = query_cost * queries_per_day
    monthly_total = (kb_cost + daily_query_cost * 30)

    print(f"\nKnowledge base: {kb_size:,} documents")
    print(f"   One-time indexing: ${kb_cost:.4f}")
    print(f"   Daily queries ({queries_per_day:,}/day): ${daily_query_cost:.4f}")
    print(f"   Monthly total: ${monthly_total:.2f}")

print()
print("=" * 70)
print("✅ Embeddings workflow complete!")
print("=" * 70)

print("\n🎯 BEST PRACTICES:")
print("   • Use batch processing for multiple documents (more efficient)")
print("   • Track embeddings by knowledge base version (customer_id)")
print("   • Use feature tags for different search types")
print("   • Cache embeddings - regenerate only when documents change")
print()

print("💡 INTEGRATION PATTERNS:")
print("   • Store embeddings in vector DB (Pinecone, Weaviate, Chroma)")
print("   • Use cosine similarity for semantic search")
print("   • Combine with chat completions for full RAG pipeline")
print("   • Track costs per knowledge base for chargeback")
print()

print("📚 Next Steps:")
print("   • Try context_manager_patterns.py for complex workflows")
print("   • See production_deployment.py for high-volume patterns")
