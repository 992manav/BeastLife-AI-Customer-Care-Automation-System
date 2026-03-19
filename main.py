#!/usr/bin/env python3
"""
Main entry point for BeastLife AI Customer Care System.

This script provides CLI commands to run different components of the system.
"""

import asyncio
import sys
import argparse
import subprocess
from pathlib import Path
import os
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.core import validate_llm_configuration, setup_logger, get_settings
from src.database import get_db
from src.rag import get_rag_system
from src.agents import build_graph, execute_query

logger = setup_logger(__name__)

def run_api():
    """Run FastAPI server."""
    logger.info("Starting FastAPI server...")
    subprocess.run([
        sys.executable, "-m", "uvicorn",
        "src.api.main:app",
        "--host", "0.0.0.0",
        "--port", "8000",
        "--reload"
    ])

def run_dashboard():
    """Run Streamlit dashboard."""
    logger.info("Starting Streamlit dashboard...")
    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        "src/dashboard/app.py",
        "--logger.level=info"
    ])

async def test_query(query_text: str):
    """Test a single query through the system."""
    logger.info(f"Testing query: {query_text}")
    
    try:
        # Initialize components
        settings = validate_llm_configuration()
        logger.info(f"Using LLM Provider: {settings.llm_provider}")
        
        db = await get_db()
        logger.info("Database initialized")
        
        rag = await get_rag_system()
        logger.info("RAG system initialized")
        
        graph = await build_graph()
        logger.info("LangGraph built")
        
        # Execute query
        result = await execute_query(query_text, graph)
        
        # Display results
        print("\n" + "="*60)
        print("QUERY RESULT")
        print("="*60)
        print(f"Query: {result.get('query', '')}")
        print(f"Category: {result.get('category', '')}")
        print(f"Sentiment: {result.get('sentiment', '')}")
        print(f"Confidence: {result.get('confidence', '')}")
        print(f"Path: {result.get('path', '')}")
        print(f"Response: {result.get('response', '')}")
        print(f"Execution Time: {result.get('execution_time_ms', 0):.2f}ms")
        print("="*60 + "\n")
    
    except Exception as e:
        logger.error(f"Error testing query: {e}")
        sys.exit(1)

async def init_system():
    """Initialize all system components."""
    logger.info("Initializing BeastLife AI Customer Care System...")
    
    try:
        # Validate configuration
        settings = validate_llm_configuration()
        logger.info(f"✓ LLM Provider validated: {settings.llm_provider}")
        
        # Initialize database
        db = await get_db()
        logger.info("✓ Database initialized")
        
        # Initialize RAG system
        rag = await get_rag_system()
        logger.info("✓ RAG system initialized")
        
        # Build LangGraph
        graph = await build_graph()
        logger.info("✓ LangGraph built")
        
        logger.info("✅ System initialization complete!")
        logger.info("You can now run the API server: python main.py api")
    
    except Exception as e:
        logger.error(f"✗ System initialization failed: {e}")
        sys.exit(1)

def main():
    """Main CLI entry point."""
    
    # Load environment variables
    load_dotenv()
    
    parser = argparse.ArgumentParser(
        description="BeastLife AI Customer Care System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py init          # Initialize the system
  python main.py api           # Start FastAPI server
  python main.py dashboard     # Start Streamlit dashboard
  python main.py test "Hello"  # Test a query
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # API command
    subparsers.add_parser("api", help="Start FastAPI server")
    
    # Dashboard command
    subparsers.add_parser("dashboard", help="Start Streamlit dashboard")
    
    # Init command
    subparsers.add_parser("init", help="Initialize the system")
    
    # Test command
    test_parser = subparsers.add_parser("test", help="Test a query")
    test_parser.add_argument("query", help="Query to test")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    if args.command == "api":
        run_api()
    elif args.command == "dashboard":
        run_dashboard()
    elif args.command == "init":
        asyncio.run(init_system())
    elif args.command == "test":
        asyncio.run(test_query(args.query))

if __name__ == "__main__":
    main()
