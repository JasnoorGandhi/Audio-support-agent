"""
Knowledge Base Testing Utility

This script helps students test their RAG implementation by setting up
the knowledge base and providing example queries to test against.
"""

import asyncio
import sys
import os

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.llm.agent import CustomerSupportAgent


async def test_knowledge_base_setup():
    """Test the knowledge base setup and ingestion."""
    print("Testing Knowledge Base Setup...")
    print("=" * 50)
    
    try:
        # Initialize agent (this will set up the knowledge base)
        config = {
            "model": "test",  # We're just testing KB setup, not the LLM
            "temperature": 0.7
        }
        
        agent = CustomerSupportAgent(config)
        
        # Set up knowledge base
        await agent._setup_knowledge_base()
        
        print(f"Knowledge base setup successful!")
        print(f"ChromaDB collection: {agent.collection.name}")
        print(f"Total documents: {agent.collection.count()}")
        
        return agent
        
    except Exception as e:
        print(f"Error setting up knowledge base: {str(e)}")
        return None


async def test_sample_queries(agent):
    """Provide sample queries for students to test their RAG implementation."""
    
    sample_queries = [
        "What is your return policy?",
        "How long does shipping take?",
        "How can I contact customer support?",
        "What payment methods do you accept?",
        "Can I track my order?",
        "Do you ship internationally?",
        "What is covered under warranty?",
        "How do I return an item?",
        "What are your business hours?",
        "Can I cancel my order?"
    ]
    
    print("\n" + "=" * 50)
    print("Sample Queries for Testing RAG Implementation")
    print("=" * 50)
    print("Use these queries to test your _rag_search() implementation:")
    print()
    
    for i, query in enumerate(sample_queries, 1):
        print(f"{i:2d}. {query}")
        
        # Call the RAG search method (students need to implement this)
        result = await agent._rag_search(query)
        print(f"    Result: {result[:100]}{'...' if len(result) > 100 else ''}")
        print()


def show_knowledge_base_structure(agent):
    """Show the structure of the knowledge base to help students understand the data."""
    
    print("\n" + "=" * 50)
    print("Knowledge Base Structure")
    print("=" * 50)
    
    try:
        # Get a sample of documents to show structure
        sample_results = agent.collection.query(
            query_texts=["return policy"],
            n_results=2,
            include=['documents', 'metadatas', 'distances']
        )
        
        print("Sample Document Structure:")
        print("-" * 30)
        
        if sample_results['documents'] and sample_results['documents'][0]:
            for i, (doc, meta, distance) in enumerate(zip(
                sample_results['documents'][0],
                sample_results['metadatas'][0],
                sample_results['distances'][0]
            )):
                print(f"Document {i + 1}:")
                print(f"  Title: {meta['title']}")
                print(f"  Category: {meta['category']}")
                print(f"  Distance: {distance:.4f}")
                print(f"  Content: {doc[:150]}{'...' if len(doc) > 150 else ''}")
                print()
        
        # Show available categories
        all_results = agent.collection.query(
            query_texts=[""],
            n_results=agent.collection.count(),
            include=['metadatas']
        )
        
        categories = set()
        if all_results['metadatas'] and all_results['metadatas'][0]:
            categories = {meta['category'] for meta in all_results['metadatas'][0]}
        
        print(f"Available Categories: {', '.join(sorted(categories))}")
        print(f"Total Documents: {len(all_results['metadatas'][0]) if all_results['metadatas'] else 0}")
        
    except Exception as e:
        print(f"Error exploring knowledge base: {str(e)}")


async def main():
    """Main function to run the knowledge base tests."""
    
    print("Customer Support Agent - Knowledge Base Test")
    print("=" * 60)
    print()
    
    # Test knowledge base setup
    agent = await test_knowledge_base_setup()
    
    if agent is None:
        print("Cannot proceed without successful knowledge base setup.")
        return
    
    # Show knowledge base structure
    show_knowledge_base_structure(agent)
    
    # Test sample queries
    await test_sample_queries(agent)
    
    print("\n" + "=" * 60)
    print("Next Steps:")
    print("1. Implement the _rag_search() method in CustomerSupportAgent")
    print("2. Use ChromaDB's query() method to search for relevant documents")
    print("3. Format the results into readable responses")
    print("4. Test with the sample queries above")
    print("5. Run this script again to see your implementation in action!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())