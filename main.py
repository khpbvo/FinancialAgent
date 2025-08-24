#!/usr/bin/env python3
"""
FinancialExpertAgent - AI-Powered Financial Analysis CLI
Full implementation with OpenAI GPT-5 and vector database
"""

import os
import sys
import click
import json
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

# OpenAI and environment
from openai import OpenAI
from dotenv import load_dotenv

# Initialize
load_dotenv()
console = Console()

# ================== Configuration ==================
class Settings:
    # OpenAI Configuration
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL = "gpt-5"  # Using GPT-5 for now
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
    AGENT_TEMPERATURE = 1
    
    MAX_COMPLETION_TOKENS = 8196

    # Categories for expense classification
    EXPENSE_CATEGORIES = [
        "Housing", "Transportation", "Food & Dining", "Utilities",
        "Healthcare", "Insurance", "Personal Care", "Entertainment",
        "Shopping", "Education", "Savings", "Investments", "Debt Payments",
        "Miscellaneous"
    ]

settings = Settings()

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
    doc_metadata = Column(JSON)  # Renamed from metadata to avoid SQLAlchemy conflict
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

# ================== Document Parser ==================
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
                'date': ['date', 'transaction date', 'trans date', 'posted date'],
                'description': ['description', 'merchant', 'payee', 'details'],
                'amount': ['amount', 'debit', 'credit', 'transaction amount'],
                'category': ['category', 'type', 'classification']
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
                    transaction = {
                        'date': pd.to_datetime(row.get('date', datetime.now())),
                        'description': str(row.get('description', 'Unknown')),
                        'amount': float(row.get('amount', 0)),
                        'category': str(row.get('category', 'Uncategorized'))
                    }
                    transactions.append(transaction)
                except Exception:
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
        date_formats = ['%m/%d/%Y', '%m/%d/%y', '%Y-%m-%d', '%d/%m/%Y']
        
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

# ================== Financial Expert Agent ==================
class FinancialExpertAgent:
    """Main Financial Expert Agent"""
    
    def __init__(self):
        if not settings.OPENAI_API_KEY:
            console.print("[red]Error: OPENAI_API_KEY not found in environment[/red]")
            console.print("[yellow]Please set your OpenAI API key in the .env file[/yellow]")
            sys.exit(1)
        
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
        self.document_parser = DocumentParser()
        self.transaction_analyzer = TransactionAnalyzer()
    
    def process_document(self, file_path: str) -> Dict[str, Any]:
        """Process a financial document"""
        parsed_data = self.document_parser.parse_document(file_path)
        
        if not parsed_data['transactions']:
            return {
                'status': 'warning',
                'message': 'No transactions found in document',
                'transactions_processed': 0
            }
        
        try:
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
                # Categorize transaction
                category = self.transaction_analyzer.categorize_transaction(
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
            
            return {
                'status': 'success',
                'document_type': parsed_data['metadata'].get('document_type'),
                'transactions_processed': transactions_processed,
                'file_path': file_path
            }
        except Exception as e:
            console.print(f"[red]Database error: {e}[/red]")
            return {
                'status': 'error',
                'message': str(e),
                'transactions_processed': 0
            }
    
    def analyze_financial_health(self, period: Optional[str] = None) -> Dict[str, Any]:
        """Analyze financial health"""
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
                return {
                    'analysis': {'total_spending': 0, 'transaction_count': 0},
                    'insights': {'insights': ['No transactions found'], 'recommendations': []},
                    'period': period or 'all_time'
                }
            
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
            analysis = self.transaction_analyzer.analyze_spending_patterns(transaction_dicts)
            
            # Generate insights using AI
            insights = self.generate_insights(analysis)
            
            db.close()
            
            return {
                'analysis': analysis,
                'insights': insights,
                'period': period or 'all_time',
                'generated_at': datetime.now().isoformat()
            }
        except Exception as e:
            console.print(f"[red]Analysis error: {e}[/red]")
            return {
                'analysis': {'total_spending': 0, 'transaction_count': 0},
                'insights': {'insights': [f'Error: {str(e)}'], 'recommendations': []},
                'period': period or 'all_time'
            }
    
    def generate_insights(self, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate insights using OpenAI"""
        try:
            context = f"""
            Financial Analysis Summary:
            - Total Spending: ${analysis_data['total_spending']:,.2f}
            - Average Monthly Spending: ${analysis_data['average_monthly_spending']:,.2f}
            - Number of Transactions: {analysis_data['transaction_count']}
            
            Category Breakdown:
            """
            
            for category, percentage in analysis_data['category_percentages'].items():
                context += f"- {category}: {percentage:.1f}%\n"
            
            response = self.client.chat.completions.create(
                model=settings.OPENAI_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a financial expert. Provide 3 key insights and 3 actionable recommendations based on the financial data."
                    },
                    {
                        "role": "user",
                        "content": f"Analyze this financial data:\n{context}"
                    }
                ],
                temperature=settings.AGENT_TEMPERATURE,
                max_completion_tokens=2000
            )
            
            content = response.choices[0].message.content
            
            if not content:
                return {
                    'insights': ['Financial analysis completed'],
                    'recommendations': ['Review your spending patterns regularly'],
                    'risk_assessment': 'Unknown'
                }
            
            # Parse response
            lines = content.split('\n')
            insights = []
            recommendations = []
            current_section = None
            
            for line in lines:
                line = line.strip()
                if 'insight' in line.lower() and len(insights) == 0:
                    current_section = 'insights'
                elif 'recommendation' in line.lower():
                    current_section = 'recommendations'
                elif line and line[0] in '•-123456789':
                    cleaned_line = line.lstrip('•-123456789. ')
                    if current_section == 'insights' and len(insights) < 3:
                        insights.append(cleaned_line)
                    elif current_section == 'recommendations' and len(recommendations) < 3:
                        recommendations.append(cleaned_line)
            
            return {
                'insights': insights if insights else ['Financial data processed successfully'],
                'recommendations': recommendations if recommendations else ['Continue monitoring your spending patterns'],
                'risk_assessment': 'Medium' if analysis_data['total_spending'] > 5000 else 'Low'
            }
        except Exception as e:
            console.print(f"[yellow]Warning: Could not generate AI insights: {e}[/yellow]")
            return {
                'insights': ['Financial analysis completed'],
                'recommendations': ['Review your spending patterns regularly'],
                'risk_assessment': 'Unknown'
            }
    
    def chat_with_agent(self, user_input: str) -> str:
        """Chat with the financial agent"""
        try:
            # Get recent transactions for context
            db = SessionLocal()
            recent_transactions = db.query(Transaction).order_by(Transaction.date.desc()).limit(10).all()
            
            context = "Recent transactions:\n"
            for t in recent_transactions:
                context += f"- {t.date.strftime('%Y-%m-%d')}: {t.description} ${t.amount:.2f} ({t.category})\n"
            
            response = self.client.chat.completions.create(
                model=settings.OPENAI_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": f"You are a helpful financial assistant. Here's the user's recent financial data:\n{context}\n\nProvide helpful financial advice based on their data."
                    },
                    {
                        "role": "user",
                        "content": user_input
                    }
                ],
                temperature=settings.AGENT_TEMPERATURE,
                max_completion_tokens=2000
            )
            
            db.close()
            return response.choices[0].message.content or "I apologize, but I couldn't generate a response. Please try again."
        except Exception as e:
            return f"I apologize, but I encountered an error: {str(e)}. Please try again."

# ================== CLI Commands ==================

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
    
    agent = FinancialExpertAgent()
    
    with console.status("[bold green]Analyzing document..."):
        result = agent.process_document(file_path)
        
        if result['status'] == 'success':
            console.print(Panel(
                f"✅ Document processed successfully!\n"
                f"Type: {result.get('document_type', 'Unknown')}\n"
                f"Transactions: {result['transactions_processed']}",
                title="Analysis Complete",
                border_style="green"
            ))
            
            # Generate analysis
            analysis = agent.analyze_financial_health('last_month')
            
            # Display insights
            if analysis['insights']['insights']:
                console.print("\n[bold cyan]Key Insights:[/bold cyan]")
                for i, insight in enumerate(analysis['insights']['insights'], 1):
                    console.print(f"  {i}. {insight}")
            
            if analysis['insights']['recommendations']:
                console.print("\n[bold cyan]Recommendations:[/bold cyan]")
                for i, rec in enumerate(analysis['insights']['recommendations'], 1):
                    console.print(f"  {i}. {rec}")
        else:
            console.print(f"[bold yellow]Warning:[/bold yellow] {result.get('message', 'Document processing incomplete')}")

@cli.command()
@click.option('--period', default='last_month', 
              type=click.Choice(['last_month', 'last_quarter', 'last_year', 'all']),
              help='Analysis period')
def summary(period):
    """Show financial overview and summary"""
    agent = FinancialExpertAgent()
    
    with console.status("[bold green]Generating financial summary..."):
        analysis = agent.analyze_financial_health(period if period != 'all' else None)
        
        if analysis['analysis']['transaction_count'] == 0:
            console.print("[yellow]No transactions found. Please analyze some documents first.[/yellow]")
            return
        
        # Create summary table
        table = Table(title=f"Financial Summary - {period.replace('_', ' ').title()}")
        table.add_column("Category", style="cyan")
        table.add_column("Amount", justify="right", style="green")
        table.add_column("Percentage", justify="right", style="yellow")
        
        for category, amount in analysis['analysis']['category_totals'].items():
            percentage = analysis['analysis']['category_percentages'][category]
            table.add_row(
                category,
                f"${amount:,.2f}",
                f"{percentage:.1f}%"
            )
        
        console.print(table)
        
        # Display totals
        console.print(Panel(
            f"Total Spending: ${analysis['analysis']['total_spending']:,.2f}\n"
            f"Average Monthly: ${analysis['analysis']['average_monthly_spending']:,.2f}\n"
            f"Transactions: {analysis['analysis']['transaction_count']}\n"
            f"Risk Level: {analysis['insights']['risk_assessment']}",
            title="Summary Statistics",
            border_style="blue"
        ))

@cli.command()
def chat():
    """Interactive chat with the Financial Expert Agent"""
    agent = FinancialExpertAgent()
    
    console.print(Panel(
        "Welcome to Financial Expert Chat!\n"
        "Ask me anything about your finances.\n"
        "Type 'exit' or 'quit' to end the session.",
        title="💬 Chat Mode",
        border_style="cyan"
    ))
    
    while True:
        try:
            user_input = console.input("\n[bold cyan]You:[/bold cyan] ")
            
            if user_input.lower() in ['exit', 'quit', 'bye']:
                console.print("[yellow]Goodbye! Stay financially healthy! 👋[/yellow]")
                break
            
            with console.status("[dim]Thinking...[/dim]"):
                response = agent.chat_with_agent(user_input)
            
            console.print(f"\n[bold green]Agent:[/bold green] {response}")
            
        except KeyboardInterrupt:
            console.print("\n[yellow]Chat ended.[/yellow]")
            break
        except Exception as e:
            console.print(f"[red]Error: {str(e)}[/red]")

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
    agent = FinancialExpertAgent()
    
    with console.status(f"[bold green]Exporting data as {format}..."):
        analysis = agent.analyze_financial_health(period if period != 'all' else None)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"financial_analysis_{period}_{timestamp}.{format}"
        filepath = settings.EXPORT_DIR / filename
        
        try:
            if format == 'json':
                with open(filepath, 'w') as f:
                    json.dump(analysis, f, indent=2, default=str)
            elif format == 'csv':
                import csv
                with open(filepath, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['Category', 'Amount', 'Percentage'])
                    for cat, amt in analysis['analysis']['category_totals'].items():
                        pct = analysis['analysis']['category_percentages'][cat]
                        writer.writerow([cat, amt, pct])
            
            console.print(f"[green]✅ Exported to:[/green] {filepath}")
        except Exception as e:
            console.print(f"[red]Export failed: {e}[/red]")

if __name__ == '__main__':
    # Initialize database on startup
    try:
        init_database()
    except Exception as e:
        console.print(f"[yellow]Warning: Database initialization failed: {e}[/yellow]")
    
    cli()
