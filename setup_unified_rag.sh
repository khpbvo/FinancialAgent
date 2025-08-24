#!/bin/bash
# Migration and Setup Script for Unified RAG System

echo "=========================================="
echo "Unified RAG System Setup & Migration"
echo "=========================================="

# Step 1: Check PostgreSQL and create extensions
echo ""
echo "Step 1: Setting up PostgreSQL with PGVector..."
echo "----------------------------------------------"

# Check if PostgreSQL is running
if ! pg_isready -h localhost -p 5432 > /dev/null 2>&1; then
    echo "❌ PostgreSQL is not running. Please start it first."
    echo "   On macOS: brew services start postgresql"
    echo "   On Linux: sudo systemctl start postgresql"
    exit 1
fi

echo "✅ PostgreSQL is running"

# Create database and enable pgvector extension
echo "Creating database and enabling pgvector..."
psql -U postgres << EOF
-- Create database if not exists
SELECT 'CREATE DATABASE financial_agent'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'financial_agent');

-- Connect to the database
\c financial_agent;

-- Create user if not exists
DO
\$do\$
BEGIN
   IF NOT EXISTS (
      SELECT FROM pg_catalog.pg_roles
      WHERE  rolname = 'financial_user') THEN

      CREATE ROLE financial_user LOGIN PASSWORD 'financial_pass';
   END IF;
END
\$do\$;

-- Grant privileges
GRANT ALL PRIVILEGES ON DATABASE financial_agent TO financial_user;

-- Install pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create schema for RAG tables
CREATE SCHEMA IF NOT EXISTS rag;
GRANT ALL ON SCHEMA rag TO financial_user;
EOF

echo "✅ Database setup complete"

# Step 2: Install Python dependencies
echo ""
echo "Step 2: Installing Python dependencies..."
echo "----------------------------------------------"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install additional dependencies for unified RAG
pip install pgvector==0.2.5
pip install psycopg2-binary==2.9.9
pip install sqlalchemy==2.0.0

echo "✅ Dependencies installed"

# Step 3: Create necessary directories
echo ""
echo "Step 3: Creating directory structure..."
echo "----------------------------------------------"

mkdir -p documents
mkdir -p uploads
mkdir -p exports

echo "✅ Directories created"

# Step 4: Initialize the unified RAG database tables
echo ""
echo "Step 4: Initializing RAG database tables..."
echo "----------------------------------------------"

python3 << EOF
import os
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY', 'placeholder')

from unified_rag_agent import Base, engine

try:
    Base.metadata.create_all(bind=engine)
    print("✅ RAG tables created successfully")
except Exception as e:
    print(f"❌ Error creating tables: {e}")
EOF

# Step 5: Migrate existing documents
echo ""
echo "Step 5: Migrating existing documents..."
echo "----------------------------------------------"

python3 << 'EOF'
import os
import json
import asyncio
from pathlib import Path

# Set API key
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY', 'placeholder')

async def migrate_documents():
    from unified_rag_agent import rag_store
    
    # Check for existing JSON embeddings
    json_path = Path("documents/embeddings.json")
    if json_path.exists():
        print("Found existing embeddings.json")
        with open(json_path, 'r') as f:
            data = json.load(f)
        print(f"  - Found {len(data)} existing embeddings (will need re-ingestion)")
    
    # Migrate documents from documents folder
    docs_path = Path("documents")
    migrated = 0
    
    for file_path in docs_path.glob("*"):
        if file_path.suffix in ['.pdf', '.csv', '.txt']:
            print(f"  Migrating: {file_path.name}")
            result = await rag_store.ingest_document(str(file_path))
            if result["status"] == "success":
                migrated += 1
                print(f"    ✅ Added with {result['chunks_created']} chunks")
            elif result["status"] == "exists":
                print(f"    ℹ️ Already exists")
            else:
                print(f"    ❌ Error: {result['message']}")
    
    print(f"\n✅ Migration complete: {migrated} documents added")

# Run migration
asyncio.run(migrate_documents())
EOF

# Step 6: Create convenience scripts
echo ""
echo "Step 6: Creating convenience scripts..."
echo "----------------------------------------------"

# Create add_document.sh script
cat > add_document.sh << 'EOF'
#!/bin/bash
# Add a document to the RAG knowledge base

if [ $# -eq 0 ]; then
    echo "Usage: ./add_document.sh <file_path>"
    exit 1
fi

source venv/bin/activate
python3 unified_rag_agent.py ingest "$1"
EOF

chmod +x add_document.sh

# Create search_knowledge.sh script
cat > search_knowledge.sh << 'EOF'
#!/bin/bash
# Search the RAG knowledge base

if [ $# -eq 0 ]; then
    echo "Usage: ./search_knowledge.sh <query>"
    exit 1
fi

source venv/bin/activate
python3 unified_rag_agent.py search "$@"
EOF

chmod +x search_knowledge.sh

# Create start_chat.sh script
cat > start_chat.sh << 'EOF'
#!/bin/bash
# Start the unified RAG chat interface

source venv/bin/activate
python3 unified_rag_agent.py chat
EOF

chmod +x start_chat.sh

echo "✅ Convenience scripts created:"
echo "   - ./add_document.sh <file>  - Add document to RAG"
echo "   - ./search_knowledge.sh <query>  - Search knowledge base"
echo "   - ./start_chat.sh  - Start interactive chat"

# Step 7: Update .env file template
echo ""
echo "Step 7: Checking environment configuration..."
echo "----------------------------------------------"

if [ ! -f ".env" ]; then
    cat > .env << 'EOF'
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Database Configuration
DATABASE_URL=postgresql://financial_user:financial_pass@localhost:5432/financial_agent

# Optional: Existing OpenAI Vector Store IDs (comma-separated)
# OPENAI_VECTOR_STORE_IDS=vs_xxx,vs_yyy
EOF
    echo "⚠️  Created .env file - Please add your OPENAI_API_KEY"
else
    echo "✅ .env file exists"
fi

# Final summary
echo ""
echo "=========================================="
echo "✅ Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Ensure your OPENAI_API_KEY is set in .env file"
echo "2. Start the chat interface: ./start_chat.sh"
echo "3. In chat, you can:"
echo "   - Type questions (auto-searches knowledge base)"
echo "   - 'add <file>' to add documents"
echo "   - 'add-dir <directory>' to bulk add"
echo "   - 'list' to see all documents"
echo ""
echo "Command line usage:"
echo "   python3 unified_rag_agent.py chat     - Interactive mode"
echo "   python3 unified_rag_agent.py ingest <file>  - Add document"
echo "   python3 unified_rag_agent.py search <query> - Search"
echo "   python3 unified_rag_agent.py list     - List documents"
echo ""
echo "Happy analyzing! 🚀"
