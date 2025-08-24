# Unified RAG System - Improvements Documentation

## 🚀 Overview of Improvements

I've created a **Unified RAG System** that addresses all the issues you mentioned and provides a seamless, production-ready solution for your FinancialAgent project.

## ✨ Key Improvements

### 1. **Unified Storage with PostgreSQL + PGVector**
- Replaced the simple JSON file storage with PostgreSQL database
- Uses PGVector extension for efficient vector similarity search
- All embeddings stored in proper database tables with metadata
- Automatic deduplication using file hashing

### 2. **Automatic RAG-First Approach**
- The agent **ALWAYS** searches the knowledge base first for any financial question
- No need to manually call retrieval functions
- Automatic context injection into responses

### 3. **Multiple Document Ingestion Methods**

#### Chat-Based Commands:
- `add <file_path>` - Add single document while chatting
- `add-dir <directory>` - Bulk add all documents from a directory
- `list` - View all documents in knowledge base

#### Command Line Interface:
```bash
python unified_rag_agent.py ingest <file>     # Add single document
python unified_rag_agent.py search <query>    # Search knowledge base
python unified_rag_agent.py list              # List all documents
python unified_rag_agent.py chat              # Start interactive chat
```

#### Convenience Scripts:
```bash
./add_document.sh <file>       # Quick document addition
./search_knowledge.sh <query>  # Quick search
./start_chat.sh                # Start chat interface
```

### 4. **Intelligent Document Processing**
- **Smart Chunking**: Documents split into overlapping chunks for better retrieval
- **Multi-Format Support**: PDF, CSV, TXT, and Markdown files
- **Semantic Search**: Uses cosine similarity with configurable threshold
- **Query Logging**: All searches logged for analysis

### 5. **Database Schema**

The new system uses three main tables:

- **`rag_documents`**: Tracks all ingested documents with metadata
- **`document_chunks`**: Stores text chunks with embeddings
- **`query_logs`**: Logs all RAG searches for optimization

## 📦 Installation & Setup

### Prerequisites
1. PostgreSQL installed with PGVector extension
2. Python 3.8+
3. OpenAI API key

### Quick Setup
```bash
# Run the automated setup script
./setup_unified_rag.sh
```

This script will:
1. ✅ Setup PostgreSQL with PGVector
2. ✅ Install Python dependencies
3. ✅ Create database tables
4. ✅ Migrate existing documents
5. ✅ Create convenience scripts

### Manual Setup
```bash
# 1. Install dependencies
pip install pgvector psycopg2-binary sqlalchemy

# 2. Initialize database
python unified_rag_agent.py init

# 3. Start chatting
python unified_rag_agent.py chat
```

## 🎯 Usage Examples

### Interactive Chat Mode
```python
# Start the chat
python unified_rag_agent.py chat

# In chat, you can:
You: What are my recent expenses?
# Agent automatically searches RAG first, then responds

You: add documents/new_statement.pdf
# Document added to knowledge base

You: add-dir documents/
# All documents in directory added

You: list
# Shows all documents in knowledge base
```

### Programmatic Usage
```python
from unified_rag_agent import rag_store

# Ingest document
result = await rag_store.ingest_document("path/to/document.pdf")

# Search knowledge base
results = await rag_store.search("What were my expenses last month?")

# List documents
docs = rag_store.list_documents()
```

## 🔧 Configuration

Edit settings in `unified_rag_agent.py`:

```python
class UnifiedSettings:
    CHUNK_SIZE: int = 1000           # Characters per chunk
    CHUNK_OVERLAP: int = 200         # Overlap between chunks
    MAX_SEARCH_RESULTS: int = 5      # Max RAG results
    MIN_SIMILARITY_SCORE: float = 0.7 # Minimum similarity threshold
```

## 🏗️ Architecture

```
┌─────────────────┐
│   User Input    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   RAG Agent     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐     ┌──────────────────┐
│  Auto RAG Search│────►│  PostgreSQL      │
└─────────────────┘     │  + PGVector      │
         │              │                  │
         ▼              │  - Documents     │
┌─────────────────┐     │  - Chunks        │
│  Context Merge  │     │  - Embeddings    │
└────────┬────────┘     └──────────────────┘
         │
         ▼
┌─────────────────┐
│  LLM Response   │
└─────────────────┘
```

## 🎨 Features Comparison

| Feature | Old System | New Unified System |
|---------|------------|-------------------|
| Storage | JSON file | PostgreSQL + PGVector |
| Embeddings | In-memory | Database with indexing |
| Deduplication | None | SHA-256 hash checking |
| Document Formats | Limited | PDF, CSV, TXT, MD |
| Ingestion | Manual function | Chat, CLI, Scripts |
| RAG Search | Manual call | Automatic |
| Chunking | Basic | Smart with overlap |
| Query Logging | None | Full logging |
| Bulk Import | No | Yes |

## 🚦 Migration from Old System

The setup script automatically migrates:
1. Existing documents in `/documents` folder
2. Previous transaction data remains intact
3. Conversation history preserved

## 🔍 How It Works

1. **Document Ingestion**:
   - File uploaded → Text extracted → Split into chunks
   - Each chunk gets embedded using OpenAI
   - Stored in PostgreSQL with metadata

2. **RAG Retrieval**:
   - User query → Embedded → Vector similarity search
   - Top K chunks retrieved based on similarity
   - Context provided to LLM

3. **Response Generation**:
   - LLM receives: User query + RAG context
   - Generates response citing sources
   - Response streamed to user

## 📈 Performance Tips

1. **Optimal Chunk Size**: 1000 characters works well for financial documents
2. **Similarity Threshold**: 0.7 balances precision and recall
3. **Index Creation**: PGVector automatically creates indexes for fast search
4. **Batch Ingestion**: Use `add-dir` for multiple files

## 🐛 Troubleshooting

### PostgreSQL Connection Issues
```bash
# Check if PostgreSQL is running
pg_isready -h localhost -p 5432

# Start PostgreSQL (macOS)
brew services start postgresql
```

### PGVector Not Found
```bash
# Install PGVector extension
CREATE EXTENSION vector;
```

### Slow Embeddings
- Ensure OPENAI_API_KEY is set correctly
- Check OpenAI API rate limits

## 🎯 Next Steps

1. **Add Web UI**: Create a Flask/FastAPI web interface
2. **Scheduled Ingestion**: Auto-import from email/cloud storage
3. **Advanced Analytics**: Add visualization of embedded documents
4. **Fine-tuning**: Optimize chunk size and overlap for your documents
5. **Caching**: Add Redis for frequently accessed embeddings

## 📝 Example Workflow

```bash
# 1. Setup the system
./setup_unified_rag.sh

# 2. Add your financial documents
python unified_rag_agent.py ingest documents/AFSCHRIFT.pdf
python unified_rag_agent.py ingest documents/NL22INGB0669807419_01-01-2025_23-08-2025.csv

# 3. Start chatting with automatic RAG
./start_chat.sh

# Now ask questions - the agent automatically searches the knowledge base!
You: What were my largest expenses last month?
You: Show me all restaurant transactions
You: Analyze my spending patterns
```

## 🤝 Integration with Existing Code

The unified system can work alongside your existing `main.py` and `enhanced_agents.py`. You can:

1. Use the same PostgreSQL database
2. Share the transaction data
3. Combine RAG search with transaction analysis

## 📊 Monitoring & Analytics

The system logs all queries for analysis:
```sql
-- View most common queries
SELECT query_text, COUNT(*) as frequency
FROM query_logs
GROUP BY query_text
ORDER BY frequency DESC;

-- Check retrieval performance
SELECT AVG(array_length(retrieved_chunks, 1)) as avg_chunks
FROM query_logs;
```

## 🔐 Security Considerations

1. **File Deduplication**: Prevents duplicate uploads
2. **Input Validation**: File type checking
3. **Database Credentials**: Use environment variables
4. **API Key Protection**: Never commit .env file

## 📚 Further Enhancements

Consider adding:
- **Reranking**: Use cross-encoder for better relevance
- **Hybrid Search**: Combine keyword and semantic search
- **Document Versioning**: Track document updates
- **User Management**: Multi-user support with permissions
- **Export/Import**: Backup and restore knowledge base

---

This unified RAG system provides a robust, scalable foundation for your FinancialAgent project with seamless document management and automatic retrieval!
