"""
Example usage and testing for BeastLife AI Customer Care System.

Run: python examples.py
"""

import asyncio
import json
from dotenv import load_dotenv
from src.core import validate_llm_configuration, setup_logger, get_llm_provider
from src.database import get_db
from src.rag import get_rag_system
from src.agents import build_graph, execute_query

# Load environment variables
load_dotenv()

logger = setup_logger(__name__)

# ================================
# EXAMPLE QUERIES
# ================================

EXAMPLE_QUERIES = [
    # Path A: API Resolution
    {
        "query": "Can you track my order #12345?",
        "description": "Order tracking - should route to Path A",
    },
    {
        "query": "I want a refund for my last purchase. I'm very upset!",
        "description": "Refund request with critical sentiment - may escalate to Path C",
    },
    {
        "query": "My payment was declined five times. I'm frustrated!",
        "description": "Payment issue with negative sentiment",
    },
    
    # Path B: RAG-Based
    {
        "query": "What are the membership options available on BeastLife?",
        "description": "General question - should route to Path B (RAG)",
    },
    {
        "query": "Tell me about your nutrition guidance services",
        "description": "Information request - RAG lookup",
    },
    
    # Path C: Escalation
    {
        "query": "I need immediate help. Something is critically wrong with my account!",
        "description": "Critical sentiment - should route to Path C",
    },
]

async def example_single_query():
    """Example 1: Process a single query."""
    print("\n" + "="*60)
    print("EXAMPLE 1: Single Query Processing")
    print("="*60)
    
    try:
        # Initialize components
        logger.info("Initializing system...")
        settings = validate_llm_configuration()
        logger.info(f"LLM Provider: {settings.llm_provider}")
        
        db = await get_db()
        rag = await get_rag_system()
        graph = await build_graph()
        
        # Process query
        query = "Can you track my order #12345?"
        logger.info(f"Processing query: {query}")
        
        result = await execute_query(query, graph)
        
        # Display results
        print(f"\nQuery: {result.get('query')}")
        print(f"Category: {result.get('category')}")
        print(f"Sentiment: {result.get('sentiment')}")
        print(f"Confidence: {result.get('confidence', 0):.2%}")
        print(f"Path: {result.get('path')}")
        print(f"Response: {result.get('response')}")
        print(f"Execution Time: {result.get('execution_time_ms', 0):.2f}ms")
    
    except Exception as e:
        logger.error(f"Error in example: {e}")

async def example_batch_queries():
    """Example 2: Process multiple queries."""
    print("\n" + "="*60)
    print("EXAMPLE 2: Batch Query Processing")
    print("="*60)
    
    try:
        # Initialize graph
        graph = await build_graph()
        
        queries = [
            "What are member benefits?",
            "How do I cancel my subscription?",
            "I can't log in to my account",
        ]
        
        # Process all queries concurrently
        logger.info(f"Processing {len(queries)} queries in parallel...")
        results = await asyncio.gather(
            *[execute_query(q, graph) for q in queries]
        )
        
        # Display results
        for i, result in enumerate(results, 1):
            print(f"\nQuery {i}: {result.get('query')}")
            print(f"  Category: {result.get('category')}")
            print(f"  Sentiment: {result.get('sentiment')}")
            print(f"  Response: {result.get('response')[:100]}...")
    
    except Exception as e:
        logger.error(f"Error in example: {e}")

async def example_rag_retrieval():
    """Example 3: RAG System Retrieval."""
    print("\n" + "="*60)
    print("EXAMPLE 3: RAG Document Retrieval")
    print("="*60)
    
    try:
        # Initialize RAG
        rag = await get_rag_system()
        llm = get_llm_provider()
        
        query = "What membership plans does BeastLife offer?"
        
        # Retrieve relevant documents
        logger.info(f"Retrieving documents for: {query}")
        result = await rag.retrieve(query, top_k=3)
        
        print(f"\nQuery: {query}")
        print(f"Retrieved {len(result.documents)} documents:")
        
        for i, doc in enumerate(result.documents, 1):
            score = result.scores[i-1] if i <= len(result.scores) else 0
            print(f"\n  Document {i}: {doc.id}")
            print(f"  Relevance Score: {score:.2%}")
            print(f"  Content: {doc.content[:150]}...")
        
        # Generate answer
        logger.info("Generating answer from retrieved documents...")
        answer = await rag.generate_answer(query, result.documents, llm)
        print(f"\nGenerated Answer:")
        print(answer)
    
    except Exception as e:
        logger.error(f"Error in example: {e}")

async def example_llm_tasks():
    """Example 4: Direct LLM Tasks."""
    print("\n" + "="*60)
    print("EXAMPLE 4: Direct LLM Processing")
    print("="*60)
    
    try:
        llm = get_llm_provider()
        query = "I need to return my order because the product is damaged."
        
        print(f"Query: {query}\n")
        
        # Classification
        logger.info("Classifying query...")
        categories = ["order_tracking", "refund_request", "payment_issue", "general_question"]
        classification = await llm.classify(query, categories)
        print(f"Classification: {classification}")
        
        # Entity extraction
        logger.info("Extracting entities...")
        entities = await llm.extract_entities(query)
        print(f"Entities: {entities}")
        
        # Sentiment analysis
        logger.info("Analyzing sentiment...")
        sentiment = await llm.analyze_sentiment(query)
        print(f"Sentiment: {sentiment}")
    
    except Exception as e:
        logger.error(f"Error in example: {e}")

async def example_database_logging():
    """Example 5: Database Logging."""
    print("\n" + "="*60)
    print("EXAMPLE 5: Database Operations")
    print("="*60)
    
    try:
        db = await get_db()
        graph = await build_graph()
        
        # Process query
        query = "Track my fitness progress"
        logger.info(f"Processing query: {query}")
        result = await execute_query(query, graph)
        
        # Get logs
        logger.info("Retrieving logs...")
        logs = await db.get_logs(limit=5)
        print(f"\nLast 5 queries in database:")
        for log in logs:
            print(f"  - {log['query'][:50]}... ({log['category']})")
        
        # Get stats
        logger.info("Retrieving statistics...")
        stats = await db.get_stats()
        print(f"\nDatabase Statistics:")
        print(f"  Total Queries: {stats['total_queries']}")
        print(f"  Resolved: {stats['resolved']}")
        print(f"  Escalated: {stats['escalated']}")
        print(f"  Resolution Rate: {stats['resolution_rate']:.1f}%")
    
    except Exception as e:
        logger.error(f"Error in example: {e}")

async def example_routing_paths():
    """Example 6: Demonstrate routing logic."""
    print("\n" + "="*60)
    print("EXAMPLE 6: Routing Path Analysis")
    print("="*60)
    
    try:
        graph = await build_graph()
        
        print("\nProcessing different query types to demonstrate routing:\n")
        
        for example in EXAMPLE_QUERIES:
            query = example["query"]
            description = example["description"]
            
            logger.info(f"Processing: {query}")
            result = await execute_query(query, graph)
            
            print(f"Query: {query}")
            print(f"  Expected: {description}")
            print(f"  Actual Path: {result.get('path')}")
            print(f"  Category: {result.get('category')}")
            print(f"  Sentiment: {result.get('sentiment')}")
            print(f"  Confidence: {result.get('confidence', 0):.0%}")
            print()
    
    except Exception as e:
        logger.error(f"Error in example: {e}")

async def main():
    """Run all examples."""
    print("\n" + "="*60)
    print("BeastLife AI Customer Care System - Examples")
    print("="*60)
    
    try:
        # Uncomment examples to run
        await example_single_query()
        # await example_batch_queries()
        # await example_rag_retrieval()
        # await example_llm_tasks()
        # await example_database_logging()
        # await example_routing_paths()
        
        print("\n" + "="*60)
        print("✅ Examples completed successfully!")
        print("="*60 + "\n")
    
    except Exception as e:
        logger.error(f"Example execution failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
