#!/usr/bin/env python3
"""Manual test script for the RAG pipeline.

Creates a temporary markdown file, ingests it, runs a few queries against
the collection, then cleans up.
"""
from pathlib import Path
import asyncio
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from brain_server.core.rag_pipeline import ingest_document, query_documents

TEST_FILE = "test_note.md"

TEST_CONTENT = """# Python Async Programming

Async programming in Python allows concurrent execution of tasks without blocking.

## Key Concepts
- async/await keywords for defining coroutines
- Event loops manage concurrent operations
- Coroutines are functions that can pause and resume
- Ideal for I/O-bound operations

## Benefits
- Better performance for I/O operations
- More efficient resource usage
- Cleaner code than threading for many use cases

This is particularly useful for network requests and database operations.
"""


async def test_ingestion() -> bool:
    print("--- Test 1: Document ingestion ---")

    with open(TEST_FILE, "w", encoding="utf-8") as f:
        f.write(TEST_CONTENT)

    try:
        result = await ingest_document(TEST_FILE)
        print(f"Success: {result['success']}")
        print(f"Chunks added: {result['chunks_added']}")
        print(f"Document ID: {result['document_id']}")
        print(f"Message: {result['message']}")
        return result["success"]
    except Exception as e:
        print(f"Error during ingestion: {e}")
        return False
    finally:
        if os.path.exists(TEST_FILE):
            os.remove(TEST_FILE)


async def test_query() -> bool:
    print("\n--- Test 2: Document query ---")

    queries = [
        "What is async programming?",
        "How do you use async/await?",
        "What are the benefits?",
    ]
    all_success = True

    for query in queries:
        print(f"\nQuery: '{query}'")
        try:
            result = await query_documents(query, top_k=3)
            print(f"Context length: {len(result['context'])} characters")
            print(f"Sources found: {len(result['sources'])}")

            if result["sources"]:
                source = result["sources"][0]
                print(f"Top source: {source['metadata'].get('file_name', 'unknown')}")
                print(f"Similarity: {source['similarity_score']:.3f}")
            else:
                print("No sources found")
                all_success = False

        except Exception as e:
            print(f"Error during query: {e}")
            all_success = False

    return all_success


async def main():
    ingest_ok = await test_ingestion()
    query_ok = await test_query() if ingest_ok else False

    print("\n--- Results ---")
    print(f"Ingestion: {'PASSED' if ingest_ok else 'FAILED'}")
    print(f"Query: {'PASSED' if query_ok else 'FAILED'}")


if __name__ == "__main__":
    asyncio.run(main())
