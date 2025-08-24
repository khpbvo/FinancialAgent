#!/usr/bin/env python3
"""
FinancialExpertAgent - Refactored with OpenAI Agents SDK
Using the Responses API with built-in tools and agent orchestration
"""

import os
import sys
import click
import json
import asyncio
import numpy as np
import pandas as pd
import pypdf
import re
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from collections import defaultdict

# Rich console for beautiful CLI
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import track

# Database and vector store
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, JSON
from sqlalchemy.orm import declarative_base, sessionmaker
from pgvector.sqlalchemy import Vector

# OpenAI Agents SDK
from agents import Agent, Runner, RunContextWrapper, function_tool
from agents.tool import WebSearchTool, FileSearchTool
from agents.models.openai_responses import OpenAIResponsesModel
from pydantic import BaseModel
from typing_extensions import TypedDict

# OpenAI Client
from openai import AsyncOpenAI

# Environment
from dotenv import load_dotenv

# Initialize
load_dotenv()
console = Console()

# ================== Configuration ==================
class Settings:
    # OpenAI Configuration
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL = "gpt-4o"  # Using gpt-4o as it's the latest available
    OPENAI_EMBEDDING_MODEL = "text-embedding-3-large"
    
    # Database Configuration
    DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://financial_user:financial_pass@localhost:5432/financial_agent")
    
    # Application Settings
    MAX_DOCUMENT_SIZE_MB = 100
    ANALYSIS_TIMEOUT_SECONDS = 30
    EMBEDDING_DIMENSION = 3072
    
    # Paths
    BASE_DIR = Path(__file__).parent
    UPLOAD_DIR = BASE_DIR / "uploads"
    EXPORT_DIR = BASE_DIR / "exports"
    
    # Agent Configuration
    AGENT_TEMPERATURE = 0.7
    MAX_COMPLETION_TOKENS = 8196

    # Categories for expense classification
    EXPENSE_CATEGORIES = [
        "Housing", "Transportation", "Food & Dining", "Utilities",
        "Healthcare", "Insurance", "Personal Care", "Entertainment",
        "Shopping", "Education", "Savings", "Investments", "Debt Payments",
        "Miscellaneous"
    ]

settings = Settings()

# Initialize OpenAI client
openai_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

# ================== Database Models ==================
Base = declarative_base()

class Transaction(Base):
    __tablename__ = 'transactions'
    
    id = Column(Integer, primary_key=True)
    amount = Column(Float, nullable=False)
    date = Column(DateTime, nullable=False)
    category = Column(String(100))
    description = Column(Text)
    source_document = Column(String(255))
    merchant = Column(String(255))
    account = Column(String(100))
    transaction_type = Column(String(50))
    created_at = Column(DateTime, default=datetime.utcnow)
    embedding = Column(Vector(settings.EMBEDDING_DIMENSION))

class Document(Base):
    __tablename__ = 'documents'
    
    id = Column(Integer, primary_key=True)
    file_path = Column(String(500), nullable=False)
    file_name = Column(String(255), nullable=False)
    upload_date = Column(DateTime, default=datetime.utcnow)
    document_type = Column(String(100))
    processed = Column(Integer, default=0)
    doc_metadata = Column(JSON)
    embedding = Column(Vector(settings.EMBEDDING_DIMENSION))

# Database setup
try:
    engine = create_engine(settings.DATABASE_URL)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
except Exception as e:
    console.print(f"[red]Database connection error: {e}[/red]")
    console.print("[yellow]Please ensure PostgreSQL with PGVector is running[/yellow]")

def init_database():
    """Initialize database with PGVector extension"""
    try:
        Base.metadata.create_all(bind=engine)
        console.print("[green]✅ Database initialized successfully[/green]")
    except Exception as e:
        console.print(f"[red]Database initialization error: {e}[/red]")

# ================== Pydantic Models for Tool Outputs ==================
class DocumentParseResult(BaseModel):
    transactions: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    raw_text: Optional[str]

class TransactionAnalysis(BaseModel):
    total_spending: float
    average_monthly_spending: float
    category_totals: Dict[str, float]
    category_percentages: Dict[str, float]
    transaction_count: int

class FinancialInsights(BaseModel):
    insights: List[str]
    recommendations: List[str]
    risk_assessment: str
    savings_opportunities: List[str]

# ================== Document Parser with Function Tools ==================
class DocumentParser:
    """Parse various financial document formats"""
    
    def __init__(self):
        self.supported_formats = ['.pdf', '.csv', '.txt']
    
    def parse_document(self, file_path: str) -> Dict[str, Any]:
        """Main parsing method"""
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        suffix = path.suffix.lower()
        
        if suffix == '.pdf':
            return self.parse_pdf(file_path)
        elif suffix == '.csv':
            return self.parse_csv(file_path)
        elif suffix == '.txt':
            return self.parse_text(file_path)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")
    
    def parse_pdf(self, file_path: str) -> Dict[str, Any]:
        """Parse PDF financial documents"""
        transactions = []
        
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = pypdf.PdfReader(file)
                text = ""
                
                for page in pdf_reader.pages:
                    text += page.extract_text()
                
                # Extract transactions using regex
                transaction_pattern = r'(\d{1,2}/\d{1,2}/\d{2,4})\s+([^\$]+?)\s+\$?([\d,]+\.?\d*)'
                matches = re.findall(transaction_pattern, text)
                
                for match in matches:
                    date_str, description, amount = match
                    try:
                        parsed_date = self._parse_date(date_str)
                        parsed_amount = float(amount.replace(',', ''))
                        
                        transactions.append({
                            'date': parsed_date,
                            'description': description.strip(),
                            'amount': parsed_amount,
                            'transaction_type': 'expense' if parsed_amount > 0 else 'income'
                        })
                    except (ValueError, TypeError):
                        continue
                
                return {
                    'transactions': transactions,
                    'metadata': {'document_type': 'pdf_statement'},
                    'raw_text': text[:1000]
                }
        except Exception as e:
            console.print(f"[red]Error parsing PDF: {e}[/red]")
            return {'transactions': [], 'metadata': {}, 'raw_text': ''}
    
    def parse_csv(self, file_path: str) -> Dict[str, Any]:
        """Parse CSV financial data"""
        try:
            df = pd.read_csv(file_path)
            
            # Standardize column names
            column_mapping = {
                'date': ['date', 'transaction date', 'trans date', 'posted date', 'datum'],
                'description': ['description', 'merchant', 'payee', 'details', 'naam / omschrijving', 'naam/omschrijving'],
                'amount': ['amount', 'debit', 'credit', 'transaction amount', 'bedrag (eur)', 'bedrag'],
                'category': ['category', 'type', 'classification', 'mutatiesoort']
            }
            
            standardized_df = pd.DataFrame()
            
            for standard_col, possible_names in column_mapping.items():
                for col in df.columns:
                    if col.lower().strip() in [name.lower() for name in possible_names]:
                        standardized_df[standard_col] = df[col]
                        break
            
            # Convert to transactions
            transactions = []
            for _, row in standardized_df.iterrows():
                try:
                    # Handle date
                    date_val = row.get('date', datetime.now())
                    if isinstance(date_val, str) and len(date_val) == 8:  # Dutch format YYYYMMDD
                        date_val = datetime.strptime(date_val, '%Y%m%d')
                    else:
                        date_val = pd.to_datetime(date_val)
                    
                    # Handle amount and debit/credit
                    amount = float(str(row.get('amount', 0)).replace(',', '.'))
                    
                    # Check for Dutch debit/credit indicator
                    af_bij = df.get('Af Bij', [''])[row.name] if 'Af Bij' in df.columns else ''
                    if af_bij == 'Af':  # Debit (money going out)
                        amount = -abs(amount)
                    elif af_bij == 'Bij':  # Credit (money coming in)
                        amount = abs(amount)
                    
                    transaction = {
                        'date': date_val,
                        'description': str(row.get('description', 'Unknown')),
                        'amount': amount,
                        'category': str(row.get('category', 'Uncategorized')),
                        'transaction_type': 'expense' if amount < 0 else 'income'
                    }
                    transactions.append(transaction)
                except Exception as e:
                    continue
            
            return {
                'transactions': transactions,
                'metadata': {'document_type': 'csv_import'},
                'raw_text': None
            }
        except Exception as e:
            console.print(f"[red]Error parsing CSV: {e}[/red]")
            return {'transactions': [], 'metadata': {}, 'raw_text': ''}
    
    def parse_text(self, file_path: str) -> Dict[str, Any]:
        """Parse text financial documents"""
        try:
            with open(file_path, 'r') as file:
                text = file.read()
            
            transactions = []
            transaction_pattern = r'(\d{1,2}/\d{1,2}/\d{2,4})\s+([^\$]+?)\s+\$?([\d,]+\.?\d*)'
            matches = re.findall(transaction_pattern, text)
            
            for match in matches:
                date_str, description, amount = match
                try:
                    transactions.append({
                        'date': self._parse_date(date_str),
                        'description': description.strip(),
                        'amount': float(amount.replace(',', ''))
                    })
                except Exception:
                    continue
            
            return {
                'transactions': transactions,
                'metadata': {'document_type': 'text_document'},
                'raw_text': text[:1000]
            }
        except Exception as e:
            console.print(f"[red]Error parsing text file: {e}[/red]")
            return {'transactions': [], 'metadata': {}, 'raw_text': ''}
    
    def _parse_date(self, date_str: str) -> datetime:
        """Parse various date formats"""
        date_formats = ['%m/%d/%Y', '%m/%d/%y', '%Y-%m-%d', '%d/%m/%Y', '%Y%m%d']
        
        for fmt in date_formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        
        return datetime.now()

# ================== Transaction Analyzer ==================
class TransactionAnalyzer:
    """Analyze and categorize financial transactions"""
    
    def __init__(self):
        self.category_keywords = {
            'Food & Dining': ['restaurant', 'cafe', 'coffee', 'pizza', 'burger', 'grocery', 'food', 'dining'],
            'Transportation': ['uber', 'lyft', 'gas', 'parking', 'toll', 'transit', 'metro', 'taxi'],
            'Shopping': ['amazon', 'walmart', 'target', 'store', 'shop', 'mall', 'retail'],
            'Utilities': ['electric', 'water', 'gas', 'internet', 'phone', 'cable', 'utility'],
            'Entertainment': ['netflix', 'spotify', 'movie', 'theater', 'concert', 'game', 'entertainment'],
            'Healthcare': ['doctor', 'hospital', 'pharmacy', 'medical', 'health', 'dental'],
            'Housing': ['rent', 'mortgage', 'property', 'maintenance', 'repair', 'housing'],
            'Insurance': ['insurance', 'premium', 'coverage'],
            'Education': ['tuition', 'school', 'course', 'training', 'book', 'education'],
            'Savings': ['savings', 'deposit', 'investment', 'retirement', '401k']
        }
    
    def categorize_transaction(self, description: str, amount: float) -> str:
        """Categorize a transaction"""
        description_lower = description.lower()
        
        for category, keywords in self.category_keywords.items():
            for keyword in keywords:
                if keyword in description_lower:
                    return category
        
        # Default categorization by amount
        if amount > 1000:
            return 'Housing'
        elif amount > 100:
            return 'Shopping'
        else:
            return 'Miscellaneous'
    
    def analyze_spending_patterns(self, transactions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze spending patterns"""
        category_totals = defaultdict(float)
        monthly_totals = defaultdict(float)
        merchant_frequency = defaultdict(int)
        
        for transaction in transactions:
            category = transaction.get('category', 'Uncategorized')
            amount = abs(transaction['amount'])  # Use absolute value
            date = transaction['date']
            description = transaction.get('description', 'Unknown')
            
            category_totals[category] += amount
            
            month_key = f"{date.year}-{date.month:02d}"
            monthly_totals[month_key] += amount
            
            merchant_frequency[description] += 1
        
        total_spending = sum(category_totals.values())
        avg_monthly = sum(monthly_totals.values()) / max(len(monthly_totals), 1)
        
        top_merchants = sorted(merchant_frequency.items(), key=lambda x: x[1], reverse=True)[:10]
        
        category_percentages = {
            cat: (amount / total_spending * 100) if total_spending > 0 else 0
            for cat, amount in category_totals.items()
        }
        
        return {
            'total_spending': total_spending,
            'average_monthly_spending': avg_monthly,
            'category_totals': dict(category_totals),
            'category_percentages': category_percentages,
            'monthly_totals': dict(monthly_totals),
            'top_merchants': top_merchants,
            'transaction_count': len(transactions)
        }

# ================== Function Tools for Agents SDK ==================
document_parser = DocumentParser()
transaction_analyzer = TransactionAnalyzer()

@function_tool
async def parse_financial_document(file_path: str) -> str:
    """Parse and extract data from financial documents.
    
    Args:
        file_path: Path to the financial document to parse
    """
    try:
        parsed_data = document_parser.parse_document(file_path)
        
        if not parsed_data['transactions']:
            return json.dumps({
                'status': 'warning',
                'message': 'No transactions found in document',
                'transactions_processed': 0
            })
        
        db = SessionLocal()
        
        # Store document
        document = Document(
            file_path=file_path,
            file_name=Path(file_path).name,
            document_type=parsed_data['metadata'].get('document_type', 'unknown'),
            doc_metadata=parsed_data['metadata'],
            processed=1
        )
        db.add(document)
        
        # Process transactions
        transactions_processed = 0
        for transaction_data in parsed_data['transactions']:
            category = transaction_analyzer.categorize_transaction(
                transaction_data.get('description', ''),
                transaction_data.get('amount', 0)
            )
            
            transaction = Transaction(
                amount=transaction_data['amount'],
                date=transaction_data['date'],
                description=transaction_data.get('description', ''),
                category=category,
                source_document=file_path,
                transaction_type=transaction_data.get('transaction_type', 'expense')
            )
            
            db.add(transaction)
            transactions_processed += 1
        
        db.commit()
        db.close()
        
        return json.dumps({
            'status': 'success',
            'document_type': parsed_data['metadata'].get('document_type'),
            'transactions_processed': transactions_processed,
            'file_path': file_path
        })
    except Exception as e:
        return json.dumps({
            'status': 'error',
            'message': str(e),
            'transactions_processed': 0
        })

@function_tool
async def analyze_transactions(period: Optional[str] = None) -> str:
    """Analyze financial transactions for patterns and insights.
    
    Args:
        period: Analysis period - 'last_month', 'last_quarter', 'last_year', or None for all time
    """
    try:
        db = SessionLocal()
        
        query = db.query(Transaction)
        
        if period == 'last_month':
            start_date = datetime.now() - timedelta(days=30)
            query = query.filter(Transaction.date >= start_date)
        elif period == 'last_quarter':
            start_date = datetime.now() - timedelta(days=90)
            query = query.filter(Transaction.date >= start_date)
        elif period == 'last_year':
            start_date = datetime.now() - timedelta(days=365)
            query = query.filter(Transaction.date >= start_date)
        
        transactions = query.all()
        
        if not transactions:
            return json.dumps({
                'analysis': {'total_spending': 0, 'transaction_count': 0},
                'period': period or 'all_time'
            })
        
        # Convert to dict format
        transaction_dicts = [
            {
                'date': t.date,
                'amount': t.amount,
                'category': t.category,
                'description': t.description
            }
            for t in transactions
        ]
        
        # Analyze patterns
        analysis = transaction_analyzer.analyze_spending_patterns(transaction_dicts)
        
        db.close()
        
        return json.dumps(analysis, default=str)
    except Exception as e:
        return json.dumps({'error': str(e)})

@function_tool
async def search_transactions(query: str, limit: int = 10) -> str:
    """Search for specific transactions using text search.
    
    Args:
        query: Search query to find transactions
        limit: Maximum number of results to return
    """
    try:
        db = SessionLocal()
        
        # Simple text search - in production, you'd use vector similarity
        transactions = db.query(Transaction).filter(
            Transaction.description.ilike(f'%{query}%')
        ).limit(limit).all()
        
        results = [
            {
                'date': t.date.isoformat(),
                'amount': t.amount,
                'description': t.description,
                'category': t.category
            }
            for t in transactions
        ]
        
        db.close()
        
        return json.dumps(results)
    except Exception as e:
        return json.dumps({'error': str(e)})

@function_tool
async def get_spending_summary(category: Optional[str] = None) -> str:
    """Get a spending summary by category.
    
    Args:
        category: Optional specific category to analyze
    """
    try:
        db = SessionLocal()
        
        if category:
            transactions = db.query(Transaction).filter(
                Transaction.category == category
            ).all()
        else:
            transactions = db.query(Transaction).all()
        
        if not transactions:
            return json.dumps({'total': 0, 'count': 0, 'category': category})
        
        total = sum(abs(t.amount) for t in transactions)
        
        db.close()
        
        return json.dumps({
            'total': total,
            'count': len(transactions),
            'category': category or 'all',
            'average': total / len(transactions)
        })
    except Exception as e:
        return json.dumps({'error': str(e)})

@function_tool
async def read_file_content(file_path: str) -> str:
    """Read the contents of a file.
    
    Args:
        file_path: Path to the file to read
    """
    try:
        path = Path(file_path)
        if not path.exists():
            return f"File not found: {file_path}"
        
        with open(file_path, 'r') as f:
            content = f.read()
        
        return content[:5000]  # Return first 5000 chars
    except Exception as e:
        return f"Error reading file: {str(e)}"

# ================== Financial Expert Agents ==================
def create_document_processor_agent():
    """Create agent for processing financial documents"""
    return Agent(
        name="Document Processor",
        model=OpenAIResponsesModel(model="gpt-4o", openai_client=openai_client),
        instructions="""You are a document processing specialist. Your role is to:
        1. Parse financial documents (PDFs, CSVs, text files) with high accuracy
        2. Extract transaction data and categorize expenses
        3. Store processed data for analysis
        4. Handle various document formats and structures
        
        Use the parse_financial_document tool to process documents.""",
        tools=[parse_financial_document, read_file_content]
    )

def create_financial_analyst_agent():
    """Create agent for financial analysis"""
    return Agent(
        name="Financial Analyst",
        model=OpenAIResponsesModel(model="gpt-4o", openai_client=openai_client),
        instructions="""You are a financial analyst expert. Your role is to:
        1. Analyze spending patterns and trends
        2. Identify unusual transactions or potential issues
        3. Calculate key financial metrics
        4. Provide data-driven insights
        
        Use the analyze_transactions and search_transactions tools to gather data.""",
        tools=[analyze_transactions, search_transactions, get_spending_summary]
    )

def create_advisor_agent():
    """Create agent for financial advice and recommendations"""
    return Agent(
        name="Financial Advisor",
        model=OpenAIResponsesModel(model="gpt-4o", openai_client=openai_client),
        instructions="""You are a personal financial advisor. Your role is to:
        1. Provide personalized financial recommendations
        2. Identify savings opportunities
        3. Suggest budget optimizations
        4. Offer actionable advice for financial health
        5. Search for relevant financial information online when needed
        
        Base your advice on the user's actual financial data and current best practices.""",
        tools=[
            WebSearchTool(),
            analyze_transactions,
            get_spending_summary
        ]
    )

def create_main_orchestrator():
    """Create the main orchestrator agent"""
    document_processor = create_document_processor_agent()
    analyst = create_financial_analyst_agent()
    advisor = create_advisor_agent()
    
    return Agent(
        name="Financial Expert Orchestrator",
        model=OpenAIResponsesModel(model="gpt-4o", openai_client=openai_client),
        instructions="""You are the main financial expert orchestrator. Your role is to:
        1. Understand user requests and delegate to appropriate specialist agents
        2. Coordinate between document processing, analysis, and advisory tasks
        3. Provide comprehensive responses by combining insights from multiple agents
        4. Ensure accuracy and relevance in all financial guidance
        
        You have access to specialist agents for:
        - Document processing: For parsing and extracting data from financial documents
        - Financial analysis: For analyzing transactions and spending patterns
        - Financial advice: For providing recommendations and guidance
        
        Always provide clear, actionable insights based on the user's actual financial data.""",
        handoffs=[document_processor, analyst, advisor],
        tools=[analyze_transactions, search_transactions]
    )

# ================== CLI Implementation ==================
@click.group()
def cli():
    """FinancialExpertAgent - AI-Powered Financial Analysis CLI"""
    # Ensure directories exist
    settings.UPLOAD_DIR.mkdir(exist_ok=True)
    settings.EXPORT_DIR.mkdir(exist_ok=True)

@cli.command()
@click.argument('file_path', type=click.Path(exists=True))
def analyze(file_path):
    """Process and analyze a financial document"""
    console.print(f"\n[bold cyan]Processing document:[/bold cyan] {file_path}")
    
    async def run_analysis():
        orchestrator = create_main_orchestrator()
        
        prompt = f"""Please process the financial document at {file_path} and provide:
        1. A summary of transactions found
        2. Key insights about spending patterns
        3. Recommendations for financial improvement"""
        
        result = await Runner.run(orchestrator, prompt)
        return result.final_output
    
    with console.status("[bold green]Analyzing document..."):
        output = asyncio.run(run_analysis())
        console.print(Panel(output, title="Analysis Complete", border_style="green"))

@cli.command()
@click.option('--period', default='last_month', 
              type=click.Choice(['last_month', 'last_quarter', 'last_year', 'all']),
              help='Analysis period')
def summary(period):
    """Show financial overview and summary"""
    
    async def run_summary():
        orchestrator = create_main_orchestrator()
        
        period_str = period if period != 'all' else None
        prompt = f"""Provide a comprehensive financial summary for {period.replace('_', ' ')}. Include:
        1. Total spending by category
        2. Key insights and trends
        3. Areas of concern or opportunity"""
        
        result = await Runner.run(orchestrator, prompt)
        return result.final_output
    
    with console.status("[bold green]Generating financial summary..."):
        output = asyncio.run(run_summary())
        console.print(Panel(output, title=f"Financial Summary - {period.replace('_', ' ').title()}", border_style="blue"))

@cli.command()
def chat():
    """Interactive chat with the Financial Expert Agent"""
    
    console.print(Panel(
        "Welcome to Financial Expert Chat!\n"
        "Ask me anything about your finances.\n"
        "Type 'exit' or 'quit' to end the session.",
        title="💬 Chat Mode",
        border_style="cyan"
    ))
    
    orchestrator = create_main_orchestrator()
    
    async def chat_loop():
        while True:
            try:
                user_input = console.input("\n[bold cyan]You:[/bold cyan] ")
                
                if user_input.lower() in ['exit', 'quit', 'bye']:
                    console.print("[yellow]Goodbye! Stay financially healthy! 👋[/yellow]")
                    break
                
                with console.status("[dim]Thinking...[/dim]"):
                    result = await Runner.run(orchestrator, user_input)
                    response = result.final_output
                
                console.print(f"\n[bold green]Agent:[/bold green] {response}")
                
            except KeyboardInterrupt:
                console.print("\n[yellow]Chat ended.[/yellow]")
                break
            except Exception as e:
                console.print(f"[red]Error: {str(e)}[/red]")
    
    asyncio.run(chat_loop())

@cli.command()
def status():
    """Show system status and statistics"""
    try:
        db = SessionLocal()
        
        total_transactions = db.query(Transaction).count()
        total_documents = db.query(Document).count()
        
        console.print(Panel(
            f"[bold cyan]System Status[/bold cyan]\n\n"
            f"Database: [green]Connected[/green]\n"
            f"Model: [green]{settings.OPENAI_MODEL}[/green]\n"
            f"API Key: [green]{'✓ Set' if settings.OPENAI_API_KEY else '✗ Missing'}[/green]\n\n"
            f"[bold]Statistics:[/bold]\n"
            f"Total Transactions: {total_transactions}\n"
            f"Total Documents: {total_documents}",
            title="📊 FinancialExpertAgent Status",
            border_style="blue"
        ))
        
        db.close()
    except Exception as e:
        console.print(f"[red]Status check failed: {e}[/red]")
        console.print("[yellow]Please check your database connection and configuration[/yellow]")

@cli.command()
@click.option('--format', type=click.Choice(['json', 'csv']), default='json', help='Export format')
@click.option('--period', default='last_month', help='Export period')
def export(format, period):
    """Export analysis results"""
    
    async def run_export():
        orchestrator = create_main_orchestrator()
        
        prompt = f"Generate a detailed financial analysis for {period} that can be exported."
        
        result = await Runner.run(orchestrator, prompt)
        return result.final_output
    
    with console.status(f"[bold green]Exporting data as {format}..."):
        analysis_text = asyncio.run(run_export())
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"financial_analysis_{period}_{timestamp}.{format}"
        filepath = settings.EXPORT_DIR / filename
        
        try:
            if format == 'json':
                with open(filepath, 'w') as f:
                    json.dump({'analysis': analysis_text, 'period': period, 'timestamp': timestamp}, f, indent=2)
            elif format == 'csv':
                # For CSV, we'd need structured data - for now, save as text
                with open(filepath, 'w') as f:
                    f.write(analysis_text)
            
            console.print(f"[green]✅ Exported to:[/green] {filepath}")
        except Exception as e:
            console.print(f"[red]Export failed: {e}[/red]")

@cli.command()
def recommendations():
    """Get personalized financial recommendations"""
    
    async def get_recommendations():
        advisor = create_advisor_agent()
        
        prompt = """Based on my financial data, provide:
        1. Top 5 personalized recommendations for improving my financial health
        2. Specific savings opportunities
        3. Budget optimization suggestions
        4. Long-term financial planning advice"""
        
        result = await Runner.run(advisor, prompt)
        return result.final_output
    
    with console.status("[bold green]Generating personalized recommendations..."):
        output = asyncio.run(get_recommendations())
        console.print(Panel(output, title="Personalized Financial Recommendations", border_style="cyan"))

if __name__ == '__main__':
    # Initialize database on startup
    try:
        init_database()
    except Exception as e:
        console.print(f"[yellow]Warning: Database initialization failed: {e}[/yellow]")
    
    cli()