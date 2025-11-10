#!/usr/bin/env python3
"""
⚡ GenOps LlamaIndex Minimal Example - Phase 1 (30 seconds)

This is the absolute simplest way to prove GenOps LlamaIndex integration works.
Perfect for first-time users - instant confidence builder!

Requirements: 
- API key: OPENAI_API_KEY, ANTHROPIC_API_KEY, or GOOGLE_API_KEY
- pip install llama-index genops-ai

Usage:
    python hello_genops_minimal.py
    
Expected result: "✅ SUCCESS! GenOps is now tracking your RAG pipeline!"
"""

def main():
    print("🚀 Testing GenOps with LlamaIndex RAG...")

    try:
        # Step 1: Enable GenOps tracking (universal CLAUDE.md standard)
        from genops.providers.llamaindex import auto_instrument
        auto_instrument()
        print("✅ GenOps auto-instrumentation enabled")

        # Step 2: Configure LlamaIndex (check for available API keys)
        import os

        from llama_index.core import Settings

        # Detect which LLM provider is available
        llm_configured = False
        embed_configured = False

        if os.getenv("OPENAI_API_KEY"):
            try:
                from llama_index.embeddings.openai import OpenAIEmbedding
                from llama_index.llms.openai import OpenAI
                Settings.llm = OpenAI(model="gpt-3.5-turbo")
                Settings.embed_model = OpenAIEmbedding()
                print("✅ OpenAI models configured")
                llm_configured = True
                embed_configured = True
            except ImportError:
                print("❌ OpenAI package not installed: pip install openai")

        elif os.getenv("ANTHROPIC_API_KEY"):
            try:
                from llama_index.llms.anthropic import Anthropic
                # Use OpenAI embeddings as fallback (most common)
                if os.getenv("OPENAI_API_KEY"):
                    from llama_index.embeddings.openai import OpenAIEmbedding
                    Settings.embed_model = OpenAIEmbedding()
                    embed_configured = True
                else:
                    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
                    Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
                    embed_configured = True

                Settings.llm = Anthropic(model="claude-3-haiku-20240307")
                print("✅ Anthropic LLM configured")
                llm_configured = True
            except ImportError:
                print("❌ Anthropic package not installed: pip install anthropic")

        elif os.getenv("GOOGLE_API_KEY"):
            try:
                from llama_index.embeddings.huggingface import HuggingFaceEmbedding
                from llama_index.llms.gemini import Gemini
                Settings.llm = Gemini(model="gemini-pro")
                Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
                print("✅ Google Gemini configured")
                llm_configured = True
                embed_configured = True
            except ImportError:
                print("❌ Google AI package not installed: pip install google-generativeai")

        if not llm_configured:
            print("❌ No API key found. Set one of:")
            print("   export OPENAI_API_KEY='sk-your-openai-key-here'")
            print("   export ANTHROPIC_API_KEY='sk-ant-your-anthropic-key-here'")
            print("   export GOOGLE_API_KEY='your-google-api-key-here'")
            print()
            print("🔧 QUICK FIX:")
            print("   1. Get API key from your preferred provider")
            print("   2. Set environment variable")
            print("   3. python hello_genops_minimal.py")
            return False

        if not embed_configured:
            print("⚠️ Using fallback embedding model")

        # Step 3: Create a simple RAG pipeline with GenOps tracking
        from llama_index.core import Document, VectorStoreIndex

        print("📄 Creating sample documents...")

        # Sample documents about GenOps and RAG
        documents = [
            Document(text="""
            GenOps is an open-source framework for AI governance and cost tracking.
            It provides comprehensive observability for RAG pipelines including
            embedding costs, retrieval performance, and synthesis quality metrics.
            GenOps integrates seamlessly with LlamaIndex for production-ready AI applications.
            """),
            Document(text="""
            RAG (Retrieval-Augmented Generation) is a technique that combines
            document retrieval with language model generation. It allows AI systems
            to access and use specific information from documents to provide
            more accurate and contextual responses.
            """),
            Document(text="""
            LlamaIndex is a framework for building RAG applications. It provides
            tools for document indexing, vector storage, query processing,
            and response synthesis. LlamaIndex supports multiple LLM providers
            and vector stores for flexible deployment options.
            """)
        ]

        print("🔍 Building vector index (this will use embeddings)...")
        index = VectorStoreIndex.from_documents(documents)

        print("🤖 Creating query engine...")
        query_engine = index.as_query_engine()

        print("💬 Running test query...")
        response = query_engine.query("What is GenOps and how does it help with RAG applications?")

        print("✅ SUCCESS! GenOps is now tracking your RAG pipeline!")
        print("💰 Cost tracking, team attribution, and governance are active.")
        print("📊 Your RAG operations are now visible in your observability platform.")
        print()
        print(f"🤖 RAG Response: {response.response[:200] if response.response else 'Success'}...")
        print()
        print("🎯 PHASE 1 COMPLETE - You now have GenOps working with LlamaIndex!")

        return True

    except ImportError as e:
        error_str = str(e).lower()
        if "llama_index" in error_str or "llama-index" in error_str:
            print("❌ LlamaIndex not installed")
            print("🔧 QUICK FIX: pip install llama-index>=0.10.0")
        elif "genops" in error_str:
            print("❌ GenOps not installed")
            print("🔧 QUICK FIX: pip install genops-ai[llamaindex]")
        else:
            print(f"❌ Import error: {e}")
            print("🔧 QUICK FIX: pip install llama-index openai anthropic")
        return False

    except Exception as e:
        error_str = str(e).lower()
        print(f"❌ Error: {e}")
        print()

        # Provide specific guidance for common errors
        if "api key" in error_str or "authentication" in error_str:
            print("🔧 API KEY ISSUE:")
            print("   1. Check your API key is set correctly")
            print("   2. Verify the key format (OpenAI: sk-..., Anthropic: sk-ant-...)")
            print("   3. Ensure the key has sufficient permissions/credits")
        elif "quota" in error_str or "rate limit" in error_str:
            print("🔧 RATE LIMIT:")
            print("   1. Wait 1-2 minutes and try again")
            print("   2. Check your API usage limits")
            print("   3. Consider using a different provider")
        elif "model" in error_str:
            print("🔧 MODEL ISSUE:")
            print("   1. Verify model name is correct")
            print("   2. Check if model is available for your API key")
            print("   3. Try with a different model")
        else:
            print("🔧 DETAILED DIAGNOSIS:")
            print("   python -c \"from genops.providers.llamaindex.validation import validate_setup, print_validation_result; print_validation_result(validate_setup(), detailed=True)\"")

        return False

if __name__ == "__main__":
    success = main()

    if success:
        print("🚀 READY FOR PHASE 2? (RAG Pipeline Optimization)")
        print("   → python rag_pipeline_tracking.py     # Complete RAG monitoring")
        print("   → python auto_instrumentation.py      # Zero-code existing apps")
        print()
        print("📚 Or explore the complete learning path:")
        print("   → examples/llamaindex/README.md")
    else:
        print()
        print("💡 Need help? Check the troubleshooting guide:")
        print("   → examples/llamaindex/README.md#troubleshooting")

    exit(0 if success else 1)
