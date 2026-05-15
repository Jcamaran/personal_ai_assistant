#!/usr/bin/env python3
"""Test script for RAG pipeline"""
import os
from dotenv import load_dotenv
from core.rag_pipeline import ingest_document, query_documents

# Load environment variables
load_dotenv()

def test_ingestion():
    """Test ingesting a document"""
    print("=" * 50)
    print("TEST 1: Document Ingestion")
    print("=" * 50)
    
    # Create a test markdown file
    test_file = "test_note.md"
    with open(test_file, 'w', encoding='utf-8') as f:
        f.write("""# Python Async Programming

Async programming in Python allows concurrent execution of tasks without blocking.

## Key Concepts
- async/await keywords for defining coroutines
- Event loops manage concurrent operations
- Coroutines are functions that can pause and resume
- Ideal for I/O-bound operations

## Example Usage
```python
import asyncio

async def fetch_data():
    await asyncio.sleep(1)
    return "Data fetched"

async def main():
    result = await fetch_data()
    print(result)
```

## Benefits
- Better performance for I/O operations
- More efficient resource usage
- Cleaner code than threading for many use cases

This is particularly useful for network requests and database operations.
""")
    
    print(f"Created test file: {test_file}")
    
    # Test ingestion
    try:
        result = ingest_document(test_file)
        
        print(f"\n✓ Success: {result['success']}")
        print(f"✓ Chunks created: {result['chunks_created']}")
        print(f"✓ Document ID: {result['document_id']}")
        print(f"✓ Message: {result['message']}")
        
        # Cleanup
        os.remove(test_file)
        print(f"\n✓ Cleaned up test file")
        
        return result['success']
    except Exception as e:
        print(f"\n✗ Error during ingestion: {e}")
        import traceback
        traceback.print_exc()
        # Cleanup on error
        if os.path.exists(test_file):
            os.remove(test_file)
        return False

def test_query():
    """Test querying documents"""
    print("\n" + "=" * 50)
    print("TEST 2: Document Query")
    print("=" * 50)
    
    # Test multiple queries
    queries = [
        "What is async programming?",
        "How do you use async/await?",
        "What are the benefits?"
    ]
    
    all_success = True
    
    for query in queries:
        print(f"\n--- Query: '{query}' ---")
        try:
            result = query_documents(query, top_k=3)
            
            print(f"✓ Context length: {len(result['context'])} characters")
            print(f"✓ Sources found: {len(result['sources'])}")
            
            if len(result['context']) > 0:
                print(f"\nContext preview:")
                preview = result['context'][:200]
                print(f"  {preview}..." if len(result['context']) > 200 else f"  {preview}")
            
            if result['sources']:
                print(f"\nTop source:")
                source = result['sources'][0]
                print(f"  File: {source['file_path']}")
                print(f"  Similarity: {source['similarity_score']:.3f}")
                print(f"  Content: {source['content'][:100]}...")
            else:
                print("  ⚠ No sources found")
                all_success = False
                
        except Exception as e:
            print(f"✗ Error during query: {e}")
            import traceback
            traceback.print_exc()
            all_success = False
    
    return all_success

if __name__ == "__main__":
    print("🧪 Testing RAG Pipeline")
    print("=" * 50)
    print()
    
    # Test 1: Ingestion
    print("Starting tests...\n")
    ingest_success = test_ingestion()
    
    if ingest_success:
        # Test 2: Query
        query_success = test_query()
        
        print("\n" + "=" * 50)
        print("TEST RESULTS")
        print("=" * 50)
        print(f"✓ Ingestion: {'PASSED' if ingest_success else 'FAILED'}")
        print(f"✓ Query: {'PASSED' if query_success else 'FAILED'}")
        
        if ingest_success and query_success:
            print("\n🎉 All tests passed!")
            print("\n✅ Your RAG pipeline is working correctly!")
            print("✅ Ready to implement FastAPI endpoints!")
        else:
            print("\n⚠ Some tests failed - check logs above")
    else:
        print("\n❌ Ingestion test failed - cannot proceed with query tests")
        print("Check the error messages above for details")
