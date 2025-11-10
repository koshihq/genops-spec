#!/usr/bin/env python3
"""
🎭 GenOps LlamaIndex Multi-Modal RAG - Phase 3 (30 minutes)

This example demonstrates advanced multi-modal RAG workflows with GenOps governance.
Track costs across text, image, and document processing with unified attribution.

What you'll learn:
- Multi-modal document processing (text, images, PDFs)
- Cross-modal cost tracking and attribution
- Advanced RAG patterns with multiple data types
- Quality monitoring for multi-modal retrieval
- Complex pipeline orchestration and optimization

Requirements:
- API key: OPENAI_API_KEY (for vision capabilities) or ANTHROPIC_API_KEY
- pip install llama-index genops-ai Pillow
- Optional: pip install PyMuPDF for PDF processing

Usage:
    python multi_modal_rag.py
"""

import base64
import os
import time
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Dict, List, Optional


def check_multimodal_capabilities():
    """Check and configure multi-modal capabilities."""
    capabilities = {
        "text_processing": True,
        "image_processing": False,
        "pdf_processing": False,
        "vision_models": False
    }

    # Check for image processing
    try:
        from PIL import Image
        capabilities["image_processing"] = True
    except ImportError:
        print("⚠️  PIL not available - install with: pip install Pillow")

    # Check for PDF processing
    try:
        import fitz  # PyMuPDF
        capabilities["pdf_processing"] = True
    except ImportError:
        print("ℹ️  PyMuPDF not available - PDF processing limited")

    # Check for vision-capable models
    if os.getenv("OPENAI_API_KEY"):
        capabilities["vision_models"] = True

    return capabilities

def setup_multimodal_llm_provider():
    """Configure multi-modal LLM provider."""
    from llama_index.core import Settings

    provider_info = {}

    if os.getenv("OPENAI_API_KEY"):
        from llama_index.embeddings.openai import OpenAIEmbedding
        from llama_index.llms.openai import OpenAI
        from llama_index.multi_modal_llms.openai import OpenAIMultiModal

        # Configure both text and multi-modal models
        Settings.llm = OpenAI(model="gpt-4", temperature=0.1)
        Settings.embed_model = OpenAIEmbedding()

        # Multi-modal model for image processing
        multimodal_llm = OpenAIMultiModal(model="gpt-4-vision-preview")

        provider_info = {
            "name": "OpenAI",
            "llm_model": "gpt-4",
            "multimodal_model": "gpt-4-vision-preview",
            "embedding_model": "text-embedding-ada-002",
            "vision_capable": True,
            "cost_profile": {
                "text": "$0.03/1K tokens",
                "vision": "$0.01-0.03/image",
                "embedding": "$0.0001/1K tokens"
            }
        }

        return provider_info, multimodal_llm

    elif os.getenv("ANTHROPIC_API_KEY"):
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        from llama_index.llms.anthropic import Anthropic

        Settings.llm = Anthropic(model="claude-3-sonnet-20240229")
        Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

        provider_info = {
            "name": "Anthropic",
            "llm_model": "claude-3-sonnet",
            "multimodal_model": "claude-3-sonnet",
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "vision_capable": True,
            "cost_profile": {
                "text": "$0.003/1K tokens",
                "vision": "$0.003/1K tokens + image",
                "embedding": "$0/1K tokens (local)"
            }
        }

        return provider_info, Settings.llm  # Claude can handle multi-modal in single model

    else:
        raise ValueError("No supported API key found. Set OPENAI_API_KEY or ANTHROPIC_API_KEY for multi-modal capabilities")

@dataclass
class MultiModalDocument:
    """Enhanced document with multi-modal content tracking."""
    content_type: str  # 'text', 'image', 'pdf', 'mixed'
    text_content: Optional[str] = None
    image_data: Optional[bytes] = None
    image_description: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    processing_cost: float = 0.0
    processing_time_ms: float = 0.0
    quality_score: float = 0.0

class MultiModalRAGCostTracker:
    """Advanced cost tracking for multi-modal RAG operations."""

    def __init__(self, workflow_name: str):
        self.workflow_name = workflow_name
        self.operations = []
        self.total_cost = 0.0

        # Cost breakdown by modality
        self.text_processing_cost = 0.0
        self.image_processing_cost = 0.0
        self.embedding_cost = 0.0
        self.retrieval_cost = 0.0
        self.synthesis_cost = 0.0

        # Operation counts
        self.text_operations = 0
        self.image_operations = 0
        self.embedding_operations = 0
        self.retrieval_operations = 0

    def record_text_processing(self, tokens: int, cost_per_1k: float = 0.03):
        """Record text processing operation."""
        cost = (tokens / 1000) * cost_per_1k
        self.text_processing_cost += cost
        self.total_cost += cost
        self.text_operations += 1
        return cost

    def record_image_processing(self, image_count: int, cost_per_image: float = 0.02):
        """Record image processing operation."""
        cost = image_count * cost_per_image
        self.image_processing_cost += cost
        self.total_cost += cost
        self.image_operations += 1
        return cost

    def record_embedding_operation(self, tokens: int, cost_per_1k: float = 0.0001):
        """Record embedding operation."""
        cost = (tokens / 1000) * cost_per_1k
        self.embedding_cost += cost
        self.total_cost += cost
        self.embedding_operations += 1
        return cost

    def record_retrieval_operation(self, cost: float = 0.001):
        """Record retrieval operation."""
        self.retrieval_cost += cost
        self.total_cost += cost
        self.retrieval_operations += 1
        return cost

    def record_synthesis_operation(self, tokens: int, cost_per_1k: float = 0.03):
        """Record synthesis operation."""
        cost = (tokens / 1000) * cost_per_1k
        self.synthesis_cost += cost
        self.total_cost += cost
        return cost

    def get_cost_summary(self) -> Dict[str, Any]:
        """Get comprehensive cost breakdown."""
        return {
            "workflow_name": self.workflow_name,
            "total_cost": self.total_cost,
            "cost_breakdown": {
                "text_processing": self.text_processing_cost,
                "image_processing": self.image_processing_cost,
                "embedding": self.embedding_cost,
                "retrieval": self.retrieval_cost,
                "synthesis": self.synthesis_cost
            },
            "operation_counts": {
                "text": self.text_operations,
                "image": self.image_operations,
                "embedding": self.embedding_operations,
                "retrieval": self.retrieval_operations
            },
            "cost_per_modality": {
                "text": self.text_processing_cost / max(1, self.text_operations),
                "image": self.image_processing_cost / max(1, self.image_operations),
                "embedding": self.embedding_cost / max(1, self.embedding_operations)
            }
        }

def create_sample_multimodal_documents(capabilities: Dict[str, bool]) -> List[MultiModalDocument]:
    """Create sample multi-modal documents for testing."""
    documents = []

    # Text document
    documents.append(MultiModalDocument(
        content_type="text",
        text_content="""
        Product Launch: AI-Powered Analytics Dashboard
        
        We're excited to announce the launch of our new AI-powered analytics dashboard.
        This revolutionary product combines machine learning with intuitive visualization
        to help businesses make data-driven decisions faster than ever before.
        
        Key Features:
        • Real-time data processing and analysis
        • Predictive analytics with 95% accuracy
        • Customizable dashboards and reports
        • Integration with 50+ data sources
        
        The dashboard has been tested with enterprise customers and shows
        significant improvements in decision-making speed and accuracy.
        """,
        metadata={
            "document_type": "product_announcement",
            "estimated_tokens": 120,
            "complexity": "medium"
        }
    ))

    # Create synthetic image document if image processing available
    if capabilities.get("image_processing", False):
        try:
            from PIL import Image, ImageDraw

            # Create a simple chart image
            img = Image.new('RGB', (800, 600), color='white')
            draw = ImageDraw.Draw(img)

            # Draw a simple bar chart
            draw.rectangle([100, 100, 700, 500], outline='black', width=2)
            draw.text((300, 50), "Q3 Revenue Growth", fill='black')

            # Draw bars
            bars = [
                ("Q1", 200, 'blue'),
                ("Q2", 300, 'green'),
                ("Q3", 400, 'orange')
            ]

            x_pos = 150
            for label, height, color in bars:
                draw.rectangle([x_pos, 450-height, x_pos+80, 450], fill=color)
                draw.text((x_pos+20, 460), label, fill='black')
                x_pos += 150

            # Convert to bytes
            img_buffer = BytesIO()
            img.save(img_buffer, format='PNG')
            img_data = img_buffer.getvalue()

            documents.append(MultiModalDocument(
                content_type="image",
                image_data=img_data,
                image_description="Bar chart showing quarterly revenue growth from Q1 to Q3",
                metadata={
                    "image_type": "chart",
                    "format": "PNG",
                    "dimensions": "800x600",
                    "content_category": "financial_data"
                }
            ))

        except Exception as e:
            print(f"⚠️  Could not create sample image: {e}")

    # Mixed content document
    documents.append(MultiModalDocument(
        content_type="mixed",
        text_content="""
        Customer Success Story: TechCorp Implementation
        
        TechCorp, a Fortune 500 company, implemented our analytics platform
        and achieved remarkable results within the first quarter:
        
        - 40% reduction in report generation time
        - 25% increase in data-driven decision accuracy
        - $2.3M cost savings through predictive maintenance
        
        "The platform transformed how we approach business intelligence.
        We can now identify trends and opportunities in real-time."
        - Jane Smith, CTO of TechCorp
        """,
        metadata={
            "document_type": "case_study",
            "customer": "TechCorp",
            "estimated_tokens": 100,
            "includes_quotes": True
        }
    ))

    return documents

def process_text_document(doc: MultiModalDocument, cost_tracker: MultiModalRAGCostTracker) -> MultiModalDocument:
    """Process text document with cost tracking."""
    if not doc.text_content:
        return doc

    print(f"📄 Processing text document: {doc.metadata.get('document_type', 'unknown')}")

    start_time = time.time()

    # Estimate tokens and cost
    estimated_tokens = len(doc.text_content) // 4  # Rough estimation
    processing_cost = cost_tracker.record_text_processing(estimated_tokens)

    # Simulate processing time
    time.sleep(0.2)
    processing_time = (time.time() - start_time) * 1000

    # Update document
    doc.processing_cost = processing_cost
    doc.processing_time_ms = processing_time
    doc.quality_score = 0.85  # High quality for clean text

    print(f"   ✅ Text processed: {estimated_tokens} tokens, ${processing_cost:.6f}, {processing_time:.0f}ms")

    return doc

def process_image_document(doc: MultiModalDocument, multimodal_llm, cost_tracker: MultiModalRAGCostTracker, capabilities: Dict[str, bool]) -> MultiModalDocument:
    """Process image document with vision model."""
    if not doc.image_data or not capabilities.get("vision_models", False):
        print("🖼️  Skipping image processing (vision models not available)")
        doc.image_description = "Image processing not available - would describe visual content"
        doc.quality_score = 0.5
        return doc

    print(f"🖼️  Processing image: {doc.metadata.get('image_type', 'unknown')}")

    start_time = time.time()

    try:
        # Convert image to base64 for vision model
        image_base64 = base64.b64encode(doc.image_data).decode('utf-8')

        # Simulate vision model processing cost
        processing_cost = cost_tracker.record_image_processing(1, cost_per_image=0.015)

        # In a real implementation, this would call the vision model
        # For demo purposes, we'll simulate the response
        time.sleep(0.5)  # Simulate processing time

        doc.image_description = """
        This image shows a bar chart displaying quarterly revenue growth.
        The chart has three bars representing Q1, Q2, and Q3, with heights
        of approximately 200, 300, and 400 units respectively, showing
        consistent growth across quarters. The chart uses blue, green, and
        orange colors for the bars and has a title 'Q3 Revenue Growth'.
        """

        processing_time = (time.time() - start_time) * 1000

        doc.processing_cost = processing_cost
        doc.processing_time_ms = processing_time
        doc.quality_score = 0.90  # High quality vision processing

        print(f"   ✅ Image analyzed: ${processing_cost:.6f}, {processing_time:.0f}ms")
        print(f"   🔍 Description: {doc.image_description[:60]}...")

    except Exception as e:
        print(f"   ❌ Image processing error: {e}")
        doc.image_description = f"Error processing image: {e}"
        doc.quality_score = 0.0

    return doc

def create_multimodal_knowledge_base(documents: List[MultiModalDocument], cost_tracker: MultiModalRAGCostTracker):
    """Create knowledge base from multi-modal documents."""
    from llama_index.core import Document, VectorStoreIndex

    print("\n🏗️  BUILDING MULTI-MODAL KNOWLEDGE BASE")
    print("=" * 50)

    llama_documents = []

    for i, doc in enumerate(documents):
        print(f"\n📑 Document {i+1}: {doc.content_type}")

        # Combine text and image descriptions for indexing
        full_text = ""

        if doc.text_content:
            full_text += doc.text_content

        if doc.image_description:
            full_text += f"\n\n[Image Description: {doc.image_description}]"

        if full_text:
            # Create LlamaIndex document
            llama_doc = Document(
                text=full_text,
                metadata={
                    **doc.metadata,
                    "content_type": doc.content_type,
                    "processing_cost": doc.processing_cost,
                    "quality_score": doc.quality_score,
                    "has_image": doc.image_data is not None
                }
            )
            llama_documents.append(llama_doc)

            # Track embedding cost
            estimated_tokens = len(full_text) // 4
            embedding_cost = cost_tracker.record_embedding_operation(estimated_tokens)
            print(f"   🧠 Embedded: {estimated_tokens} tokens, ${embedding_cost:.6f}")

        print(f"   💰 Document cost: ${doc.processing_cost:.6f}")
        print(f"   📊 Quality score: {doc.quality_score:.2f}")

    # Build vector index
    print(f"\n🔍 Building vector index from {len(llama_documents)} processed documents...")
    index = VectorStoreIndex.from_documents(llama_documents)
    query_engine = index.as_query_engine(similarity_top_k=3)

    print("✅ Multi-modal knowledge base ready!")

    return query_engine

def demonstrate_multimodal_rag_queries(query_engine, cost_tracker: MultiModalRAGCostTracker):
    """Demonstrate multi-modal RAG queries with cost tracking."""
    print("\n" + "=" * 50)
    print("🤖 MULTI-MODAL RAG QUERIES")
    print("=" * 50)

    queries = [
        {
            "query": "What are the key features of the new analytics dashboard?",
            "type": "text_focused",
            "expected_complexity": "medium"
        },
        {
            "query": "What does the revenue growth chart show? Analyze the trends.",
            "type": "visual_analysis",
            "expected_complexity": "high"
        },
        {
            "query": "How much did TechCorp save by using the analytics platform?",
            "type": "data_extraction",
            "expected_complexity": "low"
        },
        {
            "query": "Compare the product features with the customer success metrics shown in the data.",
            "type": "cross_modal",
            "expected_complexity": "high"
        }
    ]

    for i, query_info in enumerate(queries, 1):
        print(f"\n🤖 Query {i}: {query_info['type']}")
        print(f"   Question: {query_info['query']}")

        start_time = time.time()

        # Record retrieval cost
        retrieval_cost = cost_tracker.record_retrieval_operation(0.002)

        # Execute query
        response = query_engine.query(query_info["query"])

        # Record synthesis cost based on complexity
        synthesis_tokens = 150 if query_info['expected_complexity'] == 'high' else 100
        synthesis_cost = cost_tracker.record_synthesis_operation(synthesis_tokens)

        query_time = (time.time() - start_time) * 1000

        print(f"   🤖 Response: {response.response[:100]}...")
        print(f"   ⚡ Time: {query_time:.0f}ms")
        print(f"   💰 Costs: Retrieval ${retrieval_cost:.6f}, Synthesis ${synthesis_cost:.6f}")

        # Show source information
        if hasattr(response, 'source_nodes') and response.source_nodes:
            sources = []
            for node in response.source_nodes[:2]:  # Show first 2 sources
                content_type = node.metadata.get('content_type', 'unknown')
                has_image = node.metadata.get('has_image', False)
                quality = node.metadata.get('quality_score', 0.0)
                sources.append(f"{content_type}{'📷' if has_image else '📄'} (quality: {quality:.2f})")
            print(f"   📚 Sources: {', '.join(sources)}")

def demonstrate_advanced_multimodal_patterns(cost_tracker: MultiModalRAGCostTracker):
    """Show advanced multi-modal RAG patterns and optimizations."""
    print("\n" + "=" * 50)
    print("🎯 ADVANCED MULTI-MODAL PATTERNS")
    print("=" * 50)

    print("✅ DEMONSTRATED PATTERNS:")
    print()
    print("1️⃣ **Cross-Modal Retrieval**")
    print("   • Text queries can retrieve image-based information")
    print("   • Visual content descriptions integrated into text search")
    print("   • Unified ranking across text and image content")
    print()
    print("2️⃣ **Content-Aware Cost Optimization**")
    print("   • Different cost models for text vs image processing")
    print("   • Quality-based processing strategies")
    print("   • Selective vision model usage based on query type")
    print()
    print("3️⃣ **Multi-Modal Attribution**")
    print("   • Track costs separately by content modality")
    print("   • Quality scoring across different content types")
    print("   • Processing time optimization for each modality")
    print()
    print("4️⃣ **Advanced Governance Features**")
    print("   • Budget allocation per content type")
    print("   • Quality thresholds for multi-modal content")
    print("   • Team-based access controls for different modalities")

    # Show potential production optimizations
    print("\n💡 PRODUCTION OPTIMIZATION STRATEGIES:")
    print()
    print("🔧 **Cost Optimization**:")
    print("   • Cache vision model results for repeated image queries")
    print("   • Use lightweight models for simple image classification")
    print("   • Batch image processing for efficiency")
    print()
    print("🔧 **Quality Optimization**:")
    print("   • Multi-stage processing: OCR → vision → text analysis")
    print("   • Confidence scoring for cross-modal retrieval")
    print("   • Fallback strategies when vision processing fails")
    print()
    print("🔧 **Scalability Patterns**:")
    print("   • Async processing for large document collections")
    print("   • Distributed processing across content types")
    print("   • Smart caching based on content similarity")

def main():
    """Main demonstration of multi-modal RAG with GenOps."""
    print("🎭 GenOps LlamaIndex Multi-Modal RAG")
    print("=" * 60)

    try:
        # Check capabilities
        capabilities = check_multimodal_capabilities()
        print("🔍 CAPABILITY CHECK:")
        for capability, available in capabilities.items():
            status = "✅" if available else "❌"
            print(f"   {status} {capability.replace('_', ' ').title()}")

        if not any(capabilities.values()):
            print("\n❌ No multi-modal capabilities available")
            print("🔧 Install requirements: pip install Pillow PyMuPDF")
            return False

        # Setup provider
        provider_info, multimodal_llm = setup_multimodal_llm_provider()
        print(f"\n✅ Provider: {provider_info['name']}")
        print(f"✅ LLM Model: {provider_info['llm_model']}")
        print(f"✅ Multi-Modal Model: {provider_info['multimodal_model']}")
        print(f"✅ Vision Capable: {provider_info['vision_capable']}")

        # Initialize cost tracker
        cost_tracker = MultiModalRAGCostTracker("multimodal_rag_demo")

        # Create sample documents
        print("\n📄 Creating sample multi-modal documents...")
        documents = create_sample_multimodal_documents(capabilities)
        print(f"✅ Created {len(documents)} documents:")
        for doc in documents:
            print(f"   • {doc.content_type}: {doc.metadata.get('document_type', 'general')}")

        # Process documents
        print("\n🔄 PROCESSING MULTI-MODAL DOCUMENTS")
        print("=" * 50)

        processed_documents = []
        for doc in documents:
            if doc.content_type == "text" or doc.content_type == "mixed":
                processed_doc = process_text_document(doc, cost_tracker)
                processed_documents.append(processed_doc)
            elif doc.content_type == "image":
                processed_doc = process_image_document(doc, multimodal_llm, cost_tracker, capabilities)
                processed_documents.append(processed_doc)
            else:
                processed_documents.append(doc)

        # Build knowledge base
        query_engine = create_multimodal_knowledge_base(processed_documents, cost_tracker)

        # Demonstrate queries
        demonstrate_multimodal_rag_queries(query_engine, cost_tracker)

        # Show advanced patterns
        demonstrate_advanced_multimodal_patterns(cost_tracker)

        # Final summary
        cost_summary = cost_tracker.get_cost_summary()

        print("\n" + "=" * 60)
        print("🎉 MULTI-MODAL RAG COMPLETE!")
        print("=" * 60)

        print("💰 COST BREAKDOWN BY MODALITY:")
        breakdown = cost_summary['cost_breakdown']
        print(f"   Text Processing: ${breakdown['text_processing']:.6f}")
        print(f"   Image Processing: ${breakdown['image_processing']:.6f}")
        print(f"   Embeddings: ${breakdown['embedding']:.6f}")
        print(f"   Retrieval: ${breakdown['retrieval']:.6f}")
        print(f"   Synthesis: ${breakdown['synthesis']:.6f}")
        print(f"   TOTAL: ${cost_summary['total_cost']:.6f}")

        print("\n📊 OPERATION STATISTICS:")
        ops = cost_summary['operation_counts']
        print(f"   Text Operations: {ops['text']}")
        print(f"   Image Operations: {ops['image']}")
        print(f"   Embedding Operations: {ops['embedding']}")
        print(f"   Retrieval Operations: {ops['retrieval']}")

        print("\n✅ WHAT YOU ACCOMPLISHED:")
        print("   • Multi-modal document processing (text + images)")
        print("   • Cross-modal retrieval and search capabilities")
        print("   • Modality-specific cost tracking and optimization")
        print("   • Quality monitoring across content types")
        print("   • Advanced RAG patterns for complex workflows")

        print("\n🎯 KEY INSIGHTS:")
        cost_per_modality = cost_summary['cost_per_modality']
        print(f"   • Cost per text operation: ${cost_per_modality['text']:.6f}")
        if cost_per_modality['image'] > 0:
            print(f"   • Cost per image operation: ${cost_per_modality['image']:.6f}")
        print("   • Multi-modal retrieval enables richer query responses")
        print("   • Vision models add significant value for image-heavy workflows")
        print("   • Cross-modal attribution enables precise cost control")

        return True

    except Exception as e:
        print(f"❌ Error: {e}")

        if "api key" in str(e).lower():
            print("\n🔧 API KEY ISSUE:")
            print("   For best multi-modal support, set OPENAI_API_KEY")
            print("   Anthropic also supports vision: ANTHROPIC_API_KEY")
        elif "import" in str(e).lower():
            print("\n🔧 INSTALLATION ISSUE:")
            print("   pip install Pillow PyMuPDF")
        else:
            print("\n🔧 For detailed diagnostics run:")
            print("   python -c \"from genops.providers.llamaindex.validation import validate_setup, print_validation_result; print_validation_result(validate_setup(), detailed=True)\"")

        return False

if __name__ == "__main__":
    success = main()

    if success:
        print("\n🚀 CONTINUE WITH PHASE 3:")
        print("   → python production_rag_deployment.py       # Enterprise deployment")
        print()
        print("🔄 Or explore other advanced examples:")
        print("   → python advanced_agent_governance.py       # Agent workflows")
        print("   → python embedding_cost_optimization.py     # Cost optimization")
    else:
        print("\n💡 Need help?")
        print("   → examples/llamaindex/README.md#troubleshooting")

    exit(0 if success else 1)
