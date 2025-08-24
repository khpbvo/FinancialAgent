# 🚀 FinancialExpertAgent Setup Guide

## Prerequisites

1. **Python 3.11+** installed
2. **PostgreSQL** with **PGVector extension**
3. **OpenAI API Key** with GPT-5 access

## Step 1: Install PostgreSQL with PGVector

### macOS:
```bash
brew install postgresql
brew services start postgresql
brew install pgvector
```

### Ubuntu/Debian:
```bash
sudo apt update
sudo apt install postgresql postgresql-contrib
sudo apt install postgresql-15-pgvector
```

### Windows:
Download PostgreSQL from https://www.postgresql.org/download/windows/
Then install PGVector extension separately.

## Step 2: Setup Database

```bash
# Create database
createdb financial_agent

# Connect and enable PGVector
psql financial_agent
CREATE EXTENSION vector;
\q
```

## Step 3: Environment Setup

Create a `.env` file in your project root:

```env
OPENAI_API_KEY=your-openai-api-key-here
DATABASE_URL=postgresql://username:password@localhost/financial_agent
```

## Step 4: Install Python Dependencies

```bash
# Create virtual environment
python -m venv venv

# Activate it
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install dependencies
pip install openai click psycopg2-binary pgvector sqlalchemy pypdf pandas python-dotenv rich tabulate pydantic numpy
```

## Step 5: Initialize the Application

```bash
# Make the script executable
chmod +x main.py

# Run initial setup (creates tables)
python main.py status
```

## 🎮 Usage Examples

### 1. Analyze a Financial Document
```bash
python main.py analyze ~/Documents/bank_statement.pdf
```

### 2. Get Financial Summary
```bash
python main.py summary --period last_month
python main.py summary --period last_quarter
```

### 3. View Spending Trends
```bash
python main.py trends
python main.py trends --category "Food & Dining"
```

### 4. Get Personalized Recommendations
```bash
python main.py recommendations
```

### 5. Interactive Chat Mode
```bash
python main.py chat
# Then ask questions like:
# "What's my biggest expense category?"
# "How can I save more money?"
# "Show me unusual transactions"
```

### 6. Export Analysis
```bash
python main.py export --format json --period last_month
python main.py export --format csv --period last_quarter
```

## 📁 Project Structure

```
financial-expert-agent/
├── main.py                 # CLI entry point
├── .env                    # Environment variables
├── requirements.txt        # Python dependencies
├── config/
│   └── settings.py        # Configuration
├── database/
│   ├── models.py          # SQLAlchemy models
│   └── vector_store.py    # PGVector operations
├── tools/
│   ├── document_parser.py # Document parsing
│   ├── transaction_analyzer.py # Analysis logic
│   └── insight_generator.py # GPT-5 insights
├── agents/
│   └── financial_expert.py # Main agent
├── uploads/               # Document uploads
└── exports/              # Export files
```

## 🔥 Pro Tips

1. **Process Multiple Documents**: 
   ```bash
   for file in ~/Documents/statements/*.pdf; do
     python main.py analyze "$file"
   done
   ```

2. **Schedule Regular Analysis**:
   ```bash
   # Add to crontab for weekly analysis
   0 9 * * MON python /path/to/main.py summary --period last_week
   ```

3. **Bulk Import CSV Transactions**:
   ```bash
   python main.py analyze transactions.csv
   ```

4. **Chat with Context**:
   The chat mode remembers your financial data, so you can ask complex questions:
   - "Compare my spending this month vs last month"
   - "What percentage of income goes to savings?"
   - "Find all Amazon transactions over $100"

## 🛠️ Troubleshooting

### Database Connection Error
- Check PostgreSQL is running: `pg_isready`
- Verify credentials in `.env` file
- Ensure database exists: `psql -l`

### OpenAI API Error
- Verify API key has GPT-5 access
- Check API key in `.env` file
- Ensure you have sufficient credits

### PGVector Not Found
- Install extension: `CREATE EXTENSION IF NOT EXISTS vector;`
- May need superuser privileges

### Document Parsing Issues
- Ensure PDF is text-based (not scanned image)
- For scanned docs, consider OCR preprocessing
- Check file permissions

## 🚀 Advanced Features

### Custom Categories
Edit `settings.py` to add custom expense categories:
```python
EXPENSE_CATEGORIES = [
    "Your Custom Category",
    # ... more categories
]
```

### Adjust AI Temperature
For more creative insights, increase temperature in `settings.py`:
```python
AGENT_TEMPERATURE = 0.3  # Default is 0.1
```

### Increase Analysis Window
Modify embedding dimensions for more context:
```python
EMBEDDING_DIMENSION = 3072  # Using text-embedding-3-large
```

## 📊 Performance Optimization

1. **Index your database**:
   ```sql
   CREATE INDEX idx_transactions_date ON transactions(date);
   CREATE INDEX idx_transactions_category ON transactions(category);
   ```

2. **Batch process documents**:
   ```python
   # Process multiple files efficiently
   python main.py analyze *.pdf
   ```

3. **Use connection pooling** for high-volume processing

## 🎯 Next Steps

1. Start by analyzing your most recent bank statement
2. Let the system categorize your transactions
3. Review the insights and recommendations
4. Use chat mode to ask specific questions
5. Export reports for record-keeping

## 💡 Remember

- All data is processed **locally** - your financial data never leaves your machine
- The system **learns** from your categorizations over time
- Regular use improves accuracy and insights
- GPT-5 provides state-of-the-art financial analysis

---

**You're all set, CHAMP! Time to take control of your finances with AI! 🚀💰**