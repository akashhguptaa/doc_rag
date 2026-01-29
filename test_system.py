"""
Test Script for RAG System Components
Run this to verify all components are working correctly
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

print("=" * 70)
print("RAG SYSTEM COMPONENT TEST")
print("=" * 70)

# Test 1: Check environment variables
print("\n[1/5] Checking environment variables...")
gemini_key = os.getenv("GEMINI_API_KEY")
if gemini_key:
    print(f"  ✓ GEMINI_API_KEY is set (length: {len(gemini_key)})")
else:
    print("  ✗ GEMINI_API_KEY is not set")
    print("    Please set it in .env file or environment")

# Test 2: Check data files
print("\n[2/5] Checking data files...")
files_to_check = [
    "financial_data.index",
    "financial_data_metadata.pkl",
    "holdings.csv",
    "trades.csv",
]

for file in files_to_check:
    if os.path.exists(file):
        size = os.path.getsize(file) / 1024  # KB
        print(f"  ✓ {file} exists ({size:.2f} KB)")
    else:
        print(f"  ✗ {file} not found")

# Test 3: Import vector search module
print("\n[3/5] Testing vector_search module...")
try:
    from vector_search import VectorSearchEngine

    print("  ✓ vector_search module imported successfully")

    # Try to initialize
    try:
        search_engine = VectorSearchEngine(
            index_path="financial_data.index",
            metadata_path="financial_data_metadata.pkl",
        )
        stats = search_engine.get_stats()
        print(f"  ✓ Search engine initialized")
        print(f"    - Total vectors: {stats['total_vectors']}")
        print(f"    - Holdings: {stats['holdings_count']}")
        print(f"    - Trades: {stats['trades_count']}")
        print(
            f"    - Reranking: {'enabled' if stats['reranking_enabled'] else 'disabled'}"
        )

        # Test search
        print("\n  Testing search functionality...")
        results = search_engine.search("IBM trades", top_k=2)
        print(f"  ✓ Search executed: {len(results)} results found")
        if results:
            print(f"    - Top result score: {results[0]['score']:.4f}")

    except Exception as e:
        print(f"  ✗ Search engine initialization failed: {e}")
        search_engine = None

except Exception as e:
    print(f"  ✗ Failed to import vector_search: {e}")
    search_engine = None

# Test 4: Import RAG agent module
print("\n[4/5] Testing rag_agent module...")
try:
    from rag_agent import FinancialRAGAgent

    print("  ✓ rag_agent module imported successfully")

    if search_engine and gemini_key:
        try:
            rag_agent = FinancialRAGAgent(
                vector_search_engine=search_engine, gemini_api_key=gemini_key
            )
            print("  ✓ RAG agent initialized")

            # Test query
            print("\n  Testing RAG agent query...")
            result = rag_agent.query("How many IBM trades?", thread_id="test")
            print(f"  ✓ Query executed")
            print(f"    - Query type: {'RAG' if result['requires_rag'] else 'General'}")
            print(f"    - Response length: {len(result['response'])} chars")

        except Exception as e:
            print(f"  ✗ RAG agent initialization failed: {e}")
    else:
        print("  ⚠ Skipping RAG agent test (missing dependencies)")

except Exception as e:
    print(f"  ✗ Failed to import rag_agent: {e}")

# Test 5: Import FastAPI main module
print("\n[5/5] Testing main.py module...")
try:
    import main

    print("  ✓ main.py imported successfully")
    print("  ✓ FastAPI app created")

except Exception as e:
    print(f"  ✗ Failed to import main.py: {e}")

# Summary
print("\n" + "=" * 70)
print("TEST SUMMARY")
print("=" * 70)

all_passed = (
    gemini_key is not None
    and os.path.exists("financial_data.index")
    and os.path.exists("financial_data_metadata.pkl")
    and search_engine is not None
)

if all_passed:
    print("✓ All critical tests passed!")
    print("\nYou can now start the server with:")
    print("  python main.py")
    print("\nOr run the example client:")
    print("  python client_example.py demo")
else:
    print("⚠ Some tests failed. Please check the errors above.")
    print("\nCommon fixes:")
    print("1. Set GEMINI_API_KEY in .env file")
    print("2. Run the notebook to generate FAISS index files")
    print("3. Install missing dependencies: pip install -r requirements.txt")

print("=" * 70)
