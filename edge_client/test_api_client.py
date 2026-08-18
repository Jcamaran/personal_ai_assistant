"""Manual test script for the brain server connection.

Run this to verify the edge client can reach the brain server and that
queries return sensible results.
"""
from api_client import BrainServerClient


def main():
    print("--- Edge client -> brain server connection test ---")

    client = BrainServerClient()

    print("\nStep 1: health check")
    if not client.check_health():
        print("\nBrain server is not accessible.")
        print("Make sure the Docker containers are running:")
        print("  cd brain_server && docker-compose up -d")
        return

    print("\nStep 2: queries")
    test_queries = [
        "What is a RAG application?",
        "Tell me about my notes",
    ]

    for query in test_queries:
        print(f"\nQuery: '{query}'")
        response = client.query(query, top_k=3)

        if response:
            print(f"\nAnswer:\n{response.answer}\n")
            print(f"Retrieved {len(response.sources)} sources:")
            for i, source in enumerate(response.sources, 1):
                file_name = source.metadata.get('file_name', 'unknown')
                print(f"  {i}. {file_name} (score: {source.similarity_score:.3f})")
                print(f"     {source.content_snippet[:80]}...")
        else:
            print("Query failed.")

    print("\nStep 3: collection stats")
    stats = client.get_stats()
    if stats:
        print(f"Collection: {stats.collection_name}")
        print(f"Documents:  {stats.total_documents}")
        print(f"Chunks:     {stats.total_chunks}")

    client.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
