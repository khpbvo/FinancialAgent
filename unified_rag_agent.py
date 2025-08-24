#!/usr/bin/env python3
"""
Unified RAG-enabled Financial Agent with PostgreSQL + PGVector
Integrates document ingestion, embedding storage, and automatic RAG retrieval
"""

import os
import json
import asyncio
import numpy as np
from pathlib import Path
from typing import Any, List, Dict, Optional, Tuple
from datetime import datetime
import hashlib

from agents import Agent, Runner, function_tool, SQLiteSession, RunContextWrapper
from agents.tool import WebSearchTool, FileSearchTool
from openai import AsyncOpenAI
from pydantic import BaseModel
from pypdf import PdfReader
import pandas as pd

# Database imports
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, JSON, Boolean, desc
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.sql import text
from pgvector.sqlalchemy import Vector
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class UnifiedSettings:
    """Unified configuration for RAG system"""
    
    OPENAI_API_KEY: str | None = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL: str = "gpt-4o"
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-large"
    EMBEDDING_DIMENSION: int = 3072
    
    # PostgreSQL with PGVector
    DATABASE_URL: str = os.getenv("DATABASE_URL", 
        "postgresql://financial_user:financial_pass@localhost:5432/financial_agent")
    
    # Paths
    BASE_DIR: Path = Path(__file__).parent
    DOCUMENTS_DIR: Path = BASE_DIR / "documents"
    UPLOADS_DIR: Path = BASE_DIR / "uploads"
    
    # RAG Configuration
    CHUNK_SIZE: int = 1000  # Characters per chunk
    CHUNK_OVERLAP: int = 200  # Overlap between chunks
    MAX_SEARCH_RESULTS: int = 5  # Max results from vector search
    MIN_SIMILARITY_SCORE: float = 0.7  # Minimum similarity for retrieval

settings = UnifiedSettings()

# OpenAI async client
openai_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

# ---------------------------------------------------------------------------
# Database Models
# ---------------------------------------------------------------------------

Base = declarative_base()

class DocumentChunk(Base):
    """Store document chunks with embeddings for RAG"""
    __tablename__ = 'document_chunks'
    
    id = Column(Integer, primary_key=True)
    document_id = Column(Integer, nullable=False)
    chunk_text = Column(Text, nullable=False)
    chunk_index = Column(Integer, nullable=False)
    embedding = Column(Vector(settings.EMBEDDING_DIMENSION))
    metadata = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)

class RagDocument(Base):
    """Track ingested documents"""
    __tablename__ = 'rag_documents'
    
    id = Column(Integer, primary_key=True)
    file_path = Column(String(500), nullable=False)
    file_name = Column(String(255), nullable=False)
    file_hash = Column(String(64), unique=True)  # Prevent duplicates
    document_type = Column(String(100))
    total_chunks = Column(Integer, default=0)
    processed = Column(Boolean, default=False)
    ingested_at = Column(DateTime, default=datetime.utcnow)
    metadata = Column(JSON)

class QueryLog(Base):
    """Log RAG queries for analysis"""
    __tablename__ = 'query_logs'
    
    id = Column(Integer, primary_key=True)
    query_text = Column(Text, nullable=False)
    retrieved_chunks = Column(JSON)
    response = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

# Database setup
try:
    engine = create_engine(settings.DATABASE_URL)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)
except Exception as e:
    print(f"Database connection error: {e}")

# ---------------------------------------------------------------------------
# Text Processing Utilities
# ---------------------------------------------------------------------------

class TextChunker:
    """Split documents into overlapping chunks for better retrieval"""
    
    @staticmethod
    def split_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Split text into overlapping chunks"""
        chunks = []
        start = 0
        text_len = len(text)
        
        while start < text_len:
            end = start + chunk_size
            chunk = text[start:end]
            
            # Try to break at sentence boundary
            if end < text_len:
                last_period = chunk.rfind('.')
                if last_period > chunk_size * 0.8:
                    chunk = chunk[:last_period + 1]
                    end = start + last_period + 1
            
            chunks.append(chunk.strip())
            start = end - overlap
        
        return chunks
    
    @staticmethod
    def extract_text_from_pdf(file_path: str) -> str:
        """Extract text from PDF"""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                return text
        except Exception as e:
            return f"Error extracting PDF: {str(e)}"
    
    @staticmethod
    def extract_text_from_csv(file_path: str) -> str:
        """Extract text representation from CSV"""
        try:
            df = pd.read_csv(file_path)
            # Create a text representation
            text = f"CSV Data with {len(df)} rows and columns: {', '.join(df.columns)}\n\n"
            text += "Sample entries:\n"
            text += df.head(20).to_string()
            return text
        except Exception as e:
            return f"Error reading CSV: {str(e)}"

# ---------------------------------------------------------------------------
# Unified RAG Store
# ---------------------------------------------------------------------------

class UnifiedRAGStore:
    """PostgreSQL + PGVector based RAG store"""
    
    def __init__(self):
        self.chunker = TextChunker()
    
    async def create_embedding(self, text: str) -> np.ndarray:
        """Create embedding using OpenAI"""
        if not settings.OPENAI_API_KEY or settings.OPENAI_API_KEY == "test-key-placeholder":
            return np.zeros(settings.EMBEDDING_DIMENSION)
        
        resp = await openai_client.embeddings.create(
            model=settings.OPENAI_EMBEDDING_MODEL,
            input=text,
        )
        return np.array(resp.data[0].embedding)
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate file hash to detect duplicates"""
        hasher = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    
    async def ingest_document(self, file_path: str, metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """Ingest a document into the RAG system"""
        path = Path(file_path)
        if not path.exists():
            return {"status": "error", "message": f"File not found: {file_path}"}
        
        db = SessionLocal()
        try:
            # Check for duplicate
            file_hash = self._calculate_file_hash(file_path)
            existing = db.query(RagDocument).filter_by(file_hash=file_hash).first()
            if existing:
                return {
                    "status": "exists",
                    "message": f"Document already ingested: {path.name}",
                    "document_id": existing.id
                }
            
            # Extract text based on file type
            suffix = path.suffix.lower()
            if suffix == '.pdf':
                text = self.chunker.extract_text_from_pdf(file_path)
                doc_type = 'pdf'
            elif suffix == '.csv':
                text = self.chunker.extract_text_from_csv(file_path)
                doc_type = 'csv'
            elif suffix in ['.txt', '.md']:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                doc_type = 'text'
            else:
                return {"status": "error", "message": f"Unsupported file type: {suffix}"}
            
            # Create document record
            document = RagDocument(
                file_path=str(path.absolute()),
                file_name=path.name,
                file_hash=file_hash,
                document_type=doc_type,
                metadata=metadata or {}
            )
            db.add(document)
            db.flush()  # Get document ID
            
            # Split into chunks
            chunks = self.chunker.split_text(text, settings.CHUNK_SIZE, settings.CHUNK_OVERLAP)
            
            # Create embeddings and store chunks
            for idx, chunk_text in enumerate(chunks):
                if chunk_text.strip():  # Skip empty chunks
                    embedding = await self.create_embedding(chunk_text)
                    
                    chunk = DocumentChunk(
                        document_id=document.id,
                        chunk_text=chunk_text,
                        chunk_index=idx,
                        embedding=embedding.tolist(),
                        metadata={
                            "file_name": path.name,
                            "chunk_number": idx + 1,
                            "total_chunks": len(chunks)
                        }
                    )
                    db.add(chunk)
            
            # Update document with chunk count
            document.total_chunks = len(chunks)
            document.processed = True
            
            db.commit()
            
            return {
                "status": "success",
                "message": f"Ingested {path.name}",
                "document_id": document.id,
                "chunks_created": len(chunks)
            }
            
        except Exception as e:
            db.rollback()
            return {"status": "error", "message": f"Ingestion error: {str(e)}"}
        finally:
            db.close()
    
    async def search(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Search for relevant document chunks using vector similarity"""
        db = SessionLocal()
        try:
            # Create query embedding
            query_embedding = await self.create_embedding(query)
            
            # Perform vector similarity search using pgvector
            # Using cosine similarity (1 - cosine distance)
            results = db.execute(
                text("""
                    SELECT 
                        dc.id,
                        dc.chunk_text,
                        dc.metadata,
                        rd.file_name,
                        1 - (dc.embedding <=> :query_embedding::vector) as similarity
                    FROM document_chunks dc
                    JOIN rag_documents rd ON dc.document_id = rd.id
                    WHERE 1 - (dc.embedding <=> :query_embedding::vector) > :min_similarity
                    ORDER BY similarity DESC
                    LIMIT :max_results
                """),
                {
                    "query_embedding": query_embedding.tolist(),
                    "min_similarity": settings.MIN_SIMILARITY_SCORE,
                    "max_results": max_results
                }
            ).fetchall()
            
            # Format results
            formatted_results = []
            for row in results:
                formatted_results.append({
                    "chunk_id": row[0],
                    "text": row[1],
                    "metadata": row[2],
                    "file_name": row[3],
                    "similarity": float(row[4])
                })
            
            # Log query
            log_entry = QueryLog(
                query_text=query,
                retrieved_chunks=[r["chunk_id"] for r in formatted_results]
            )
            db.add(log_entry)
            db.commit()
            
            return formatted_results
            
        except Exception as e:
            print(f"Search error: {str(e)}")
            return []
        finally:
            db.close()
    
    def list_documents(self) -> List[Dict[str, Any]]:
        """List all ingested documents"""
        db = SessionLocal()
        try:
            documents = db.query(RagDocument).order_by(desc(RagDocument.ingested_at)).all()
            return [
                {
                    "id": doc.id,
                    "file_name": doc.file_name,
                    "document_type": doc.document_type,
                    "chunks": doc.total_chunks,
                    "ingested_at": doc.ingested_at.isoformat(),
                    "processed": doc.processed
                }
                for doc in documents
            ]
        finally:
            db.close()
    
    def delete_document(self, document_id: int) -> bool:
        """Delete a document and its chunks"""
        db = SessionLocal()
        try:
            # Delete chunks first
            db.query(DocumentChunk).filter_by(document_id=document_id).delete()
            # Delete document
            db.query(RagDocument).filter_by(id=document_id).delete()
            db.commit()
            return True
        except Exception as e:
            db.rollback()
            print(f"Delete error: {str(e)}")
            return False
        finally:
            db.close()

# Initialize global store
rag_store = UnifiedRAGStore()

# ---------------------------------------------------------------------------
# Enhanced Function Tools
# ---------------------------------------------------------------------------

@function_tool
async def ingest_document_tool(file_path: str) -> str:
    """Ingest a document into the RAG system for future retrieval"""
    result = await rag_store.ingest_document(file_path)
    
    if result["status"] == "success":
        return f"✅ Successfully ingested '{result['message']}' with {result['chunks_created']} chunks"
    elif result["status"] == "exists":
        return f"ℹ️ {result['message']}"
    else:
        return f"❌ {result['message']}"

@function_tool
async def rag_search(query: str) -> str:
    """Automatically search RAG database for relevant information"""
    results = await rag_store.search(query, max_results=settings.MAX_SEARCH_RESULTS)
    
    if not results:
        return "No relevant documents found in the knowledge base."
    
    # Format results
    response = f"Found {len(results)} relevant sections:\n\n"
    for i, result in enumerate(results, 1):
        response += f"**[{i}] From: {result['file_name']} (Score: {result['similarity']:.2f})**\n"
        response += f"{result['text'][:500]}...\n\n"
    
    return response

@function_tool
async def list_knowledge_base() -> str:
    """List all documents in the knowledge base"""
    documents = rag_store.list_documents()
    
    if not documents:
        return "Knowledge base is empty. Use 'ingest_document_tool' to add documents."
    
    response = f"📚 Knowledge Base ({len(documents)} documents):\n\n"
    for doc in documents:
        status = "✅" if doc["processed"] else "⏳"
        response += f"{status} **{doc['file_name']}**\n"
        response += f"   Type: {doc['document_type']} | Chunks: {doc['chunks']} | Added: {doc['ingested_at']}\n"
    
    return response

@function_tool
async def bulk_ingest_directory(directory_path: str, file_pattern: str = "*") -> str:
    """Ingest all matching files from a directory"""
    path = Path(directory_path)
    if not path.exists() or not path.is_dir():
        return f"Directory not found: {directory_path}"
    
    files = list(path.glob(file_pattern))
    if not files:
        return f"No files matching pattern '{file_pattern}' in {directory_path}"
    
    results = {"success": 0, "exists": 0, "error": 0}
    
    for file_path in files:
        if file_path.is_file():
            result = await rag_store.ingest_document(str(file_path))
            results[result["status"]] = results.get(result["status"], 0) + 1
    
    return (f"Bulk ingestion complete:\n"
            f"✅ New documents: {results['success']}\n"
            f"ℹ️ Already existed: {results.get('exists', 0)}\n"
            f"❌ Errors: {results.get('error', 0)}")

# ---------------------------------------------------------------------------
# Create Enhanced RAG Agent
# ---------------------------------------------------------------------------

def create_unified_rag_agent() -> Agent:
    """Create the unified RAG-enabled financial agent"""
    
    instructions = """You are an intelligent financial assistant with a comprehensive knowledge base.

IMPORTANT: Always follow this workflow:
1. **FIRST** - Use rag_search to query the knowledge base for ANY financial question
2. **ANALYZE** - Review the retrieved information from the RAG system
3. **SYNTHESIZE** - Combine RAG results with your analysis to provide comprehensive answers
4. **CITE** - Reference specific documents when using information from the knowledge base

Key capabilities:
- Automatic document ingestion from files or directories
- Vector-based semantic search across all documents
- Intelligent chunking for optimal retrieval
- Deduplication to prevent redundant storage

Available commands:
- Search knowledge: Use rag_search for any query
- Add document: Use ingest_document_tool with file path
- Bulk add: Use bulk_ingest_directory for multiple files
- View knowledge base: Use list_knowledge_base

Remember: ALWAYS search the knowledge base FIRST before answering financial questions!"""
    
    tools = [
        rag_search,
        ingest_document_tool,
        bulk_ingest_directory,
        list_knowledge_base,
        WebSearchTool(),  # For current market data
    ]
    
    return Agent(
        name="unified_rag_agent",
        model=settings.OPENAI_MODEL,
        instructions=instructions,
        tools=tools,
    )

# ---------------------------------------------------------------------------
# Interactive CLI Interface
# ---------------------------------------------------------------------------

async def interactive_rag_chat():
    """Enhanced interactive chat with automatic RAG retrieval"""
    agent = create_unified_rag_agent()
    
    # Use SQLite for conversation history
    db_path = str((settings.DOCUMENTS_DIR / "rag_conversation_history.db").resolve())
    session = SQLiteSession("unified_rag_session", db_path)
    
    print("\n" + "="*60)
    print("🤖 Unified RAG Financial Assistant")
    print("="*60)
    print("\nCommands:")
    print("  'add <file_path>' - Add a document to knowledge base")
    print("  'add-dir <directory>' - Add all documents from directory")
    print("  'list' - Show all documents in knowledge base")
    print("  'clear' - Clear conversation memory")
    print("  'exit' - Quit")
    print("\n💡 The assistant will automatically search the knowledge base for answers!")
    print("="*60 + "\n")
    
    loop = asyncio.get_event_loop()
    while True:
        try:
            user_input = await loop.run_in_executor(None, lambda: input("You: ").strip())
        except (KeyboardInterrupt, EOFError):
            print("\n👋 Goodbye!")
            break
        
        if not user_input:
            continue
        
        # Handle special commands
        if user_input.lower() in {"exit", "quit", "bye"}:
            print("👋 Goodbye!")
            break
        
        if user_input.lower() == "clear":
            await session.clear_session()
            print("✅ Memory cleared\n")
            continue
        
        if user_input.lower() == "list":
            docs = rag_store.list_documents()
            if docs:
                print("\n📚 Knowledge Base:")
                for doc in docs:
                    print(f"  - {doc['file_name']} ({doc['chunks']} chunks)")
                print()
            else:
                print("📭 Knowledge base is empty\n")
            continue
        
        if user_input.lower().startswith("add "):
            file_path = user_input[4:].strip()
            print(f"📎 Adding document: {file_path}")
            result = await rag_store.ingest_document(file_path)
            if result["status"] == "success":
                print(f"✅ Added with {result['chunks_created']} chunks\n")
            else:
                print(f"❌ {result['message']}\n")
            continue
        
        if user_input.lower().startswith("add-dir "):
            dir_path = user_input[8:].strip()
            print(f"📁 Adding directory: {dir_path}")
            result = await bulk_ingest_directory(dir_path)
            print(f"{result}\n")
            continue
        
        # Regular chat with automatic RAG
        try:
            result = Runner.run_streamed(agent, input=user_input, session=session)
            print("Assistant:", end=" ", flush=True)
            async for event in result.stream_events():
                if event.type == "raw_response_event":
                    if hasattr(event.data, 'delta'):
                        print(event.data.delta, end="", flush=True)
            print("\n")
        except (KeyboardInterrupt, asyncio.CancelledError):
            print("\n(Interrupted)\n")
            break

# ---------------------------------------------------------------------------
# Command Line Interface
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "init":
            print("🔧 Initializing database...")
            Base.metadata.create_all(bind=engine)
            print("✅ Database initialized with RAG tables")
        
        elif command == "ingest" and len(sys.argv) > 2:
            file_path = sys.argv[2]
            print(f"📎 Ingesting: {file_path}")
            result = asyncio.run(rag_store.ingest_document(file_path))
            print(f"Result: {result}")
        
        elif command == "search" and len(sys.argv) > 2:
            query = " ".join(sys.argv[2:])
            print(f"🔍 Searching: {query}")
            results = asyncio.run(rag_store.search(query))
            for i, result in enumerate(results, 1):
                print(f"\n[{i}] {result['file_name']} (Score: {result['similarity']:.2f})")
                print(f"{result['text'][:200]}...")
        
        elif command == "list":
            docs = rag_store.list_documents()
            print(f"\n📚 Knowledge Base ({len(docs)} documents):")
            for doc in docs:
                print(f"  [{doc['id']}] {doc['file_name']} - {doc['chunks']} chunks")
        
        elif command == "chat":
            try:
                asyncio.run(interactive_rag_chat())
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
        
        else:
            print("Usage:")
            print("  python unified_rag_agent.py init              - Initialize database")
            print("  python unified_rag_agent.py ingest <file>     - Add document to RAG")
            print("  python unified_rag_agent.py search <query>    - Search knowledge base")
            print("  python unified_rag_agent.py list              - List all documents")
            print("  python unified_rag_agent.py chat              - Interactive chat mode")
    else:
        # Default to chat mode
        try:
            asyncio.run(interactive_rag_chat())
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
