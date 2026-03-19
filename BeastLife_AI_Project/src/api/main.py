import asyncio
import time
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from src.core import validate_llm_configuration, get_settings, logger, QueryRequest, QueryResponse
from src.agents import build_graph, execute_query
from src.database import get_db
from src.rag import get_rag_system

# ================================
# GLOBAL STATE
# ================================

_graph = None

async def initialize_app():
    """Initialize app components."""
    global _graph
    logger.info("Initializing FastAPI application...")
    
    try:
        # Validate LLM configuration
        settings = validate_llm_configuration()
        logger.info(f"Using LLM Provider: {settings.llm_provider}")
        
        # Initialize database
        db = await get_db()
        logger.info("Database initialized")
        
        # Initialize RAG system
        rag = await get_rag_system()
        logger.info("RAG system initialized")
        
        # Build LangGraph
        _graph = await build_graph()
        logger.info("LangGraph built")
        
        logger.info("Application initialization complete")
    except Exception as e:
        logger.error(f"Failed to initialize application: {e}")
        raise

async def shutdown_app():
    """Cleanup on shutdown."""
    logger.info("Shutting down FastAPI application...")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for app startup and shutdown."""
    await initialize_app()
    yield
    await shutdown_app()

# ================================
# FASTAPI APP
# ================================

app = FastAPI(
    title="BeastLife AI Customer Care System",
    description="Production-grade AI system for handling customer queries with multi-agent orchestration",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================================
# ENDPOINTS
# ================================

@app.get("/health", tags=["System"])
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "BeastLife AI Customer Care",
        "timestamp": time.time()
    }

@app.post("/query", response_model=QueryResponse, tags=["Queries"])
async def process_query(request: QueryRequest):
    """
    Process customer query through AI system.
    
    Args:
        request: QueryRequest with customer query and optional metadata
    
    Returns:
        QueryResponse with AI-generated response and metadata
    """
    global _graph
    
    if not _graph:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        start_time = time.time()
        
        # Add metadata to request
        initial_state = {
            "query": request.query,
            "metadata": {
                "customer_id": request.customer_id,
                "session_id": request.session_id,
            }
        }
        
        # Execute through LangGraph
        result = await execute_query(request.query, _graph)
        
        execution_time = (time.time() - start_time) * 1000
        
        # Build response
        response = QueryResponse(
            query=request.query,
            category=result.get("category", "unknown"),
            sentiment=result.get("sentiment", "neutral"),
            confidence=result.get("confidence", 0.0),
            response=result.get("response", ""),
            path=result.get("path", ""),
            entities=result.get("entities", {}),
            intents=result.get("all_intents", []),
            execution_time_ms=execution_time,
        )
        
        logger.info(f"Query processed successfully in {execution_time:.2f}ms")
        return response
    
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/logs", tags=["Analytics"])
async def get_logs(
    customer_id: str = Query(None),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0)
):
    """
    Retrieve query logs.
    
    Args:
        customer_id: Optional customer ID filter
        limit: Number of logs to return
        offset: Number of logs to skip
    
    Returns:
        List of query logs
    """
    try:
        db = await get_db()
        logs = await db.get_logs(customer_id=customer_id, limit=limit, offset=offset)
        return {
            "logs": logs,
            "total": len(logs),
            "limit": limit,
            "offset": offset,
        }
    except Exception as e:
        logger.error(f"Error retrieving logs: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats", tags=["Analytics"])
async def get_stats():
    """
    Get system statistics.
    
    Returns:
        Statistics about queries, resolutions, and escalations
    """
    try:
        db = await get_db()
        stats = await db.get_stats()
        return {
            "stats": stats,
            "timestamp": time.time(),
        }
    except Exception as e:
        logger.error(f"Error retrieving stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query/batch", tags=["Queries"])
async def process_batch_queries(requests: list[QueryRequest]):
    """
    Process multiple queries in parallel.
    
    Args:
        requests: List of QueryRequest objects
    
    Returns:
        List of QueryResponse objects
    """
    global _graph
    
    if not _graph:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        # Process all queries concurrently
        tasks = [
            process_query(request) for request in requests
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle errors
        responses = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Error in batch processing: {result}")
                responses.append({"error": str(result)})
            else:
                responses.append(result)
        
        return {
            "queries": len(requests),
            "processed": len([r for r in results if not isinstance(r, Exception)]),
            "results": responses,
        }
    
    except Exception as e:
        logger.error(f"Error processing batch: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/config", tags=["System"])
async def get_config():
    """Get application configuration (non-sensitive data)."""
    settings = get_settings()
    return {
        "llm_provider": settings.llm_provider,
        "database_url": "***" if settings.database_url else None,
        "api_port": settings.api_port,
        "api_env": settings.api_env,
    }

# ================================
# ERROR HANDLERS
# ================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions."""
    logger.error(f"HTTP error: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error"}
    )

# ================================
# ROOT ENDPOINT
# ================================

@app.get("/", tags=["System"])
async def root():
    """Root endpoint with API documentation."""
    return {
        "service": "BeastLife AI Customer Care System",
        "version": "1.0.0",
        "docs_url": "/docs",
        "endpoints": {
            "health": "GET /health",
            "process_query": "POST /query",
            "batch_queries": "POST /query/batch",
            "logs": "GET /logs",
            "stats": "GET /stats",
            "config": "GET /config",
        }
    }

if __name__ == "__main__":
    import uvicorn
    settings = get_settings()
    uvicorn.run(
        app,
        host=settings.api_host,
        port=settings.api_port,
    )
