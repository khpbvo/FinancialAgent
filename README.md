# 🚀 FinancialExpertAgent

AI-Powered Financial Analysis Tool with OpenAI GPT-5 and Vector Database

## 🌟 Features

- **Document Analysis**: Parse bank statements, credit card statements, CSV files
- **AI-Powered Insights**: Get personalized financial recommendations using OpenAI 5
- **Spending Categorization**: Automatic transaction categorization
- **Vector Search**: Find similar transactions using pgvector
- **Interactive Chat**: Ask questions about your finances
- **Export Reports**: Generate JSON/CSV reports
- **Docker & Local Support**: Run with Docker or local Python environment

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Document      │    │  Transaction     │    │   OpenAI        │
│   Parser        │───▶│  Analyzer        │───▶│   GPT-5         │
│   (PDF/CSV)     │    │  & Categorizer   │    │   Insights      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   PostgreSQL    │    │    PGVector      │    │   Rich CLI      │
│   Database      │───▶│   Embeddings     │───▶│   Interface     │
│                 │    │   & Search       │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 🚀 Quick Start

### Option 1: Docker Setup (Recommended)

```bash
# 1. Clone and navigate to the project
cd FinancialAgent

# 2. Set your OpenAI API key in .env
OPENAI_API_KEY=your-actual-api-key-here

# 3. Start with Docker
make up

# 4. Test the setup
make shell
python main.py status
```

### Option 2: Local Development Setup

```bash
# 1. Run the setup script
./setup-local.sh

# 2. Activate virtual environment
source venv/bin/activate

# 3. Set your OpenAI API key in .env
OPENAI_API_KEY=your-actual-api-key-here

# 4. Install PostgreSQL with PGVector
brew install postgresql pgvector
brew services start postgresql

# 5. Create database
createdb financial_agent
psql financial_agent -c "CREATE EXTENSION vector;"

# 6. Test the setup
python main.py status
```

## 📚 Usage Examples

### Analyze Financial Documents

```bash
# Analyze a bank statement PDF
python main.py analyze ~/Documents/bank_statement.pdf

# Analyze a CSV file
python main.py analyze ~/Documents/transactions.csv

# Using Docker
make analyze
```

### Get Financial Summary

```bash
# Last month summary
python main.py summary --period last_month

# Last quarter summary
python main.py summary --period last_quarter

# Using Docker
make summary
```

### Interactive Chat

```bash
# Start chat mode
python main.py chat

# Example questions:
# - "What's my biggest expense category?"
# - "How much did I spend on dining last month?"
# - "Find all transactions over $100"
# - "What are my savings opportunities?"

# Using Docker
make chat
```

### Export Reports

```bash
# Export as JSON
python main.py export --format json --period last_month

# Export as CSV
python main.py export --format csv --period last_quarter
```

## 🛠️ Make Commands

### Docker Commands
```bash
make help          # Show all available commands
make build          # Build Docker images
make up             # Start all services
make down           # Stop all services
make logs           # View logs
make shell          # Open shell in container
make clean          # Clean up volumes
```

### Local Development Commands
```bash
make dev-setup      # Run local development setup
make local-install  # Install dependencies in venv
make local-run      # Run status check locally
make local-chat     # Start chat mode locally
make local-analyze  # Analyze document locally
make local-summary  # Show financial summary locally
```

## 📁 Project Structure

```
FinancialAgent/
├── main.py                 # Main CLI application
├── requirements.txt        # Python dependencies
├── docker-compose.yml      # Docker services
├── Dockerfile             # Docker image definition
├── Makefile              # Build and run commands
├── init.sql              # Database initialization
├── .env                  # Environment variables
├── .env.example          # Environment template
├── setup-local.sh        # Local development setup
├── venv/                 # Python virtual environment
├── uploads/              # Document uploads
├── exports/              # Analysis exports
└── documents/            # Sample documents
    └── sample_transactions.csv
```

## 🔧 Configuration

### Environment Variables (.env)

```bash
# OpenAI Configuration
OPENAI_API_KEY=your-openai-api-key-here

# Database Configuration
DATABASE_URL=postgresql://financial_user:financial_pass@localhost:5432/financial_agent

# Application Settings
ENVIRONMENT=development
MAX_DOCUMENT_SIZE_MB=100
ANALYSIS_TIMEOUT_SECONDS=30
```

### Expense Categories

The system automatically categorizes transactions into:

- Housing
- Transportation  
- Food & Dining
- Utilities
- Healthcare
- Insurance
- Personal Care
- Entertainment
- Shopping
- Education
- Savings
- Investments
- Debt Payments
- Miscellaneous

## 🧪 Testing

Test with the included sample file:

```bash
# Activate environment
source venv/bin/activate

# Analyze sample transactions
python main.py analyze documents/sample_transactions.csv

# Get summary
python main.py summary --period last_month

# Start chat
python main.py chat
```

## 🔒 Security & Privacy

- All financial data is processed **locally**
- No data sent to external services except OpenAI for insights
- Database runs locally or in your Docker environment
- OpenAI API only receives aggregated, anonymized insights

## 🐛 Troubleshooting

### Database Connection Issues
```bash
# Check PostgreSQL status
pg_isready

# Restart PostgreSQL
brew services restart postgresql

# Check Docker services
docker-compose ps
```

### Missing Dependencies
```bash
# Reinstall dependencies
source venv/bin/activate
pip install -r requirements.txt

# Or rebuild Docker
make build
```

### OpenAI API Issues
```bash
# Verify API key is set
echo $OPENAI_API_KEY

# Test API connection
curl -H "Authorization: Bearer $OPENAI_API_KEY" \
  https://api.openai.com/v1/models
```

## 📝 Development

### Adding New Document Types

1. Extend `DocumentParser` class in `main.py`
2. Add parsing logic for new format
3. Update `supported_formats` list

### Adding New Categories

1. Update `category_keywords` in `TransactionAnalyzer`
2. Add to `EXPENSE_CATEGORIES` in settings

### Custom AI Prompts

Modify the `generate_insights` method to customize AI behavior.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- OpenAI for GPT-5 API
- PostgreSQL team for the database
- pgvector for vector similarity search
- Rich library for beautiful CLI interfaces

---

**Ready to take control of your finances with AI? Let's go, CHAMP! 🚀💰**
