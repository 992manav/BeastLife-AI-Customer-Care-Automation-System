import asyncio
from datetime import datetime
from typing import List, Optional, Any, Dict
from sqlalchemy import create_engine, Column, String, Float, DateTime, Boolean, JSON, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from src.core.config import get_settings
from src.core.logger import setup_logger
from src.core.models import QueryLog

logger = setup_logger(__name__)

Base = declarative_base()

class QueryLogTable(Base):
    """SQLAlchemy model for query logs."""
    __tablename__ = "query_logs"
    
    id = Column(String, primary_key=True)
    query = Column(String, nullable=False)
    sanitized_query = Column(String, nullable=False)
    category = Column(String, nullable=False)
    sentiment = Column(String, nullable=False)
    confidence = Column(Float, nullable=False)
    response = Column(Text, nullable=False)
    path = Column(String, nullable=False)
    entities = Column(JSON, nullable=True)
    intents = Column(JSON, nullable=True)
    customer_id = Column(String, nullable=True)
    session_id = Column(String, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    resolved = Column(Boolean, default=False)
    escalated = Column(Boolean, default=False)
    feedback_score = Column(Float, nullable=True)

class Database:
    """Database connection and operations manager."""
    
    def __init__(self):
        settings = get_settings()
        self.database_url = settings.database_url
        
        # Determine if async or sync
        if self.database_url.startswith("sqlite"):
            self.engine = create_engine(
                self.database_url,
                connect_args={"check_same_thread": False}
            )
            self.SessionLocal = sessionmaker(
                autocommit=False,
                autoflush=False,
                bind=self.engine
            )
            self.is_async = False
        else:
            # For async PostgreSQL
            self.engine = None
            self.async_engine = None
            self.SessionLocal = None
            self.is_async = True
        
        self._initialized = False
    
    async def init(self):
        """Initialize database."""
        try:
            if self.is_async:
                self.async_engine = create_async_engine(self.database_url, echo=False)
                self.SessionLocal = async_sessionmaker(
                    self.async_engine,
                    class_=AsyncSession,
                    expire_on_commit=False
                )
            
            self._create_tables()
            self._initialized = True
            logger.info("Database initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise
    
    def _create_tables(self):
        """Create database tables."""
        try:
            Base.metadata.create_all(self.engine)
            logger.info("Database tables created")
        except Exception as e:
            logger.error(f"Error creating database tables: {e}")
            raise
    
    async def log_query(self, log_data: QueryLog) -> str:
        """Log a query to the database."""
        try:
            import uuid
            query_id = str(uuid.uuid4())
            
            new_log = QueryLogTable(
                id=query_id,
                query=log_data.query,
                sanitized_query=log_data.sanitized_query,
                category=log_data.category,
                sentiment=log_data.sentiment,
                confidence=log_data.confidence,
                response=log_data.response,
                path=log_data.path,
                entities=log_data.entities,
                intents=log_data.intents,
                customer_id=log_data.customer_id,
                session_id=log_data.session_id,
                timestamp=log_data.timestamp,
                resolved=log_data.resolved,
                escalated=log_data.escalated,
                feedback_score=log_data.feedback_score,
            )
            
            if self.is_async:
                async with self.SessionLocal() as session:
                    session.add(new_log)
                    await session.commit()
            else:
                db = self.SessionLocal()
                db.add(new_log)
                db.commit()
                db.close()
            
            logger.info(f"Query logged with ID: {query_id}")
            return query_id
        except Exception as e:
            logger.error(f"Error logging query: {e}")
            raise
    
    async def get_logs(
        self,
        customer_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """Retrieve logs from database."""
        try:
            if self.is_async:
                from sqlalchemy import select
                async with self.SessionLocal() as session:
                    query = select(QueryLogTable)
                    if customer_id:
                        query = query.where(QueryLogTable.customer_id == customer_id)
                    query = query.limit(limit).offset(offset)
                    result = await session.execute(query)
                    logs = result.scalars().all()
            else:
                query = self.SessionLocal().query(QueryLogTable)
                if customer_id:
                    query = query.filter(QueryLogTable.customer_id == customer_id)
                logs = query.limit(limit).offset(offset).all()
            
            return [
                {
                    "id": log.id,
                    "query": log.query,
                    "category": log.category,
                    "sentiment": log.sentiment,
                    "response": log.response,
                    "path": log.path,
                    "timestamp": log.timestamp.isoformat() if log.timestamp else None,
                    "resolved": log.resolved,
                    "escalated": log.escalated,
                }
                for log in logs
            ]
        except Exception as e:
            logger.error(f"Error retrieving logs: {e}")
            return []
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get statistics from logs."""
        try:
            if self.is_async:
                from sqlalchemy import func, select
                async with self.SessionLocal() as session:
                    total = await session.execute(select(func.count(QueryLogTable.id)))
                    total_count = total.scalar()
                    
                    resolved = await session.execute(
                        select(func.count(QueryLogTable.id)).where(QueryLogTable.resolved == True)
                    )
                    resolved_count = resolved.scalar()
                    
                    escalated = await session.execute(
                        select(func.count(QueryLogTable.id)).where(QueryLogTable.escalated == True)
                    )
                    escalated_count = escalated.scalar()
            else:
                db = self.SessionLocal()
                total_count = db.query(QueryLogTable).count()
                resolved_count = db.query(QueryLogTable).filter(QueryLogTable.resolved == True).count()
                escalated_count = db.query(QueryLogTable).filter(QueryLogTable.escalated == True).count()
                db.close()
            
            return {
                "total_queries": total_count,
                "resolved": resolved_count,
                "escalated": escalated_count,
                "resolution_rate": (resolved_count / total_count * 100) if total_count > 0 else 0,
            }
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {
                "total_queries": 0,
                "resolved": 0,
                "escalated": 0,
                "resolution_rate": 0,
            }

# Global database instance
db = Database()

async def get_db() -> Database:
    """Get database instance."""
    if not db._initialized:
        await db.init()
    return db
