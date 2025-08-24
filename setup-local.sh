#!/bin/bash

# FinancialExpertAgent Local Development Setup
echo "🚀 Setting up FinancialExpertAgent for local development..."

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed."
    exit 1
fi

# Check if PostgreSQL is running (optional for development)
if command -v pg_isready &> /dev/null; then
    if pg_isready -q; then
        echo "✅ PostgreSQL is running"
    else
        echo "⚠️  PostgreSQL is not running - you can use Docker or install locally"
    fi
else
    echo "⚠️  PostgreSQL not found - you can use Docker or install locally"
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating Python virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📚 Installing Python dependencies..."
pip install -r requirements.txt

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo "⚙️  Creating .env configuration file..."
    cp .env.example .env 2>/dev/null || cat > .env << EOF
# OpenAI Configuration
OPENAI_API_KEY=your-openai-api-key-here

# Database Configuration (for local PostgreSQL)
DATABASE_URL=postgresql://financial_user:financial_pass@localhost:5432/financial_agent

# Application Settings
ENVIRONMENT=development
EOF
fi

# Create necessary directories
echo "📁 Creating necessary directories..."
mkdir -p uploads exports documents

# Add .keep files to preserve empty directories
touch uploads/.keep exports/.keep documents/.keep

echo ""
echo "✅ Local development setup complete!"
echo ""
echo "📋 Next steps:"
echo "1. Set your OpenAI API key in .env file:"
echo "   OPENAI_API_KEY=your-actual-api-key"
echo ""
echo "2. Choose your database option:"
echo "   a) Use Docker (recommended): make up"
echo "   b) Install PostgreSQL locally and create database"
echo ""
echo "3. Test the setup:"
echo "   source venv/bin/activate"
echo "   python main.py status"
echo ""
echo "4. Start developing:"
echo "   python main.py --help"
echo ""
echo "🐳 For Docker setup: make help"
echo "💬 For local setup: source venv/bin/activate"
