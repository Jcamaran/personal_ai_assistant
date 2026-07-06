"""
Test script for API client
Run this to verify your edge client can communicate with the brain server
"""
from api_client import BrainServerClient


def main():
    print("=" * 60)
    print("🧪 Testing Edge Client → Brain Server Connection")
    print("=" * 60)
    
    # Initialize client
    client = BrainServerClient()
    
    # Step 1: Health Check
    print("\n📡 Step 1: Checking brain server health...")
    if not client.check_health():
        print("\n❌ Brain server is not accessible!")
        print("   Make sure Docker containers are running:")
        print("   cd brain_server && docker-compose up -d")
        return
    
    # Step 2: Test Query
    print("\n🧠 Step 2: Testing query functionality...")
    test_queries = [
        "What is a RAG application?",
        "Tell me about my notes"
    ]
    
    for query in test_queries:
        print(f"\n📝 Query: '{query}'")
        response = client.query(query, top_k=3)
        
        if response:
            print(f"\n💬 Answer:\n{response.answer}\n")
            print(f"📚 Retrieved {len(response.sources)} sources:")
            for i, source in enumerate(response.sources, 1):
                file_name = source.metadata.get('file_name', 'unknown')
                score = source.similarity_score
                print(f"   {i}. {file_name} (score: {score:.3f})")
                print(f"      {source.content_snippet[:80]}...")
        else:
            print("❌ Query failed!")
        
        print("-" * 60)
    
    # Step 3: Test Stats
    print("\n📊 Step 3: Testing health endpoint again...")
    client.check_health()
    
    # Close client
    client.close()
    
    print("\n" + "=" * 60)
    print("✅ All tests complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
