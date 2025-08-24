#!/usr/bin/env python3
"""
Enhanced Financial Agents with Extended SDK Tools
Demonstrates advanced usage of OpenAI Agents SDK with multiple built-in tools
"""

import os
import sys
import json
import asyncio
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from collections import defaultdict

# OpenAI Agents SDK with enhanced tools
from agents import Agent, Runner, RunContextWrapper, function_tool
from agents.tool import (
    WebSearchTool, 
    FileSearchTool, 
    CodeInterpreterTool,
    LocalShellTool
)
from agents.models import OpenAIResponsesModel
from pydantic import BaseModel
from typing_extensions import TypedDict

# Database imports
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, JSON
from sqlalchemy.orm import declarative_base, sessionmaker
from pgvector.sqlalchemy import Vector

from dotenv import load_dotenv

load_dotenv()

# Configuration
class EnhancedSettings:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://financial_user:financial_pass@localhost:5432/financial_agent")
    BASE_DIR = Path(__file__).parent
    UPLOAD_DIR = BASE_DIR / "uploads"
    EXPORT_DIR = BASE_DIR / "exports"

settings = EnhancedSettings()

# Database Models (reusing from main implementation)
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
    embedding = Column(Vector(3072))

class Document(Base):
    __tablename__ = 'documents'
    
    id = Column(Integer, primary_key=True)
    file_path = Column(String(500), nullable=False)
    file_name = Column(String(255), nullable=False)
    upload_date = Column(DateTime, default=datetime.utcnow)
    document_type = Column(String(100))
    processed = Column(Integer, default=0)
    doc_metadata = Column(JSON)
    embedding = Column(Vector(3072))

# Database setup
try:
    engine = create_engine(settings.DATABASE_URL)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
except Exception as e:
    print(f"Database connection error: {e}")

# ================== Enhanced Function Tools ==================
@function_tool
async def read_financial_file(file_path: str) -> str:
    """Read and analyze financial files with detailed content extraction.
    
    Args:
        file_path: Path to the financial file to read and analyze
    """
    try:
        path = Path(file_path)
        if not path.exists():
            return f"File not found: {file_path}"
        
        # Determine file type and read accordingly
        suffix = path.suffix.lower()
        
        if suffix == '.csv':
            import pandas as pd
            df = pd.read_csv(file_path)
            return f"CSV file with {len(df)} rows and columns: {', '.join(df.columns.tolist())}\n\nFirst few rows:\n{df.head().to_string()}"
        
        elif suffix in ['.txt', '.md']:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return f"Text file content (first 2000 chars):\n{content[:2000]}"
        
        elif suffix == '.json':
            with open(file_path, 'r') as f:
                data = json.load(f)
            return f"JSON file structure:\n{json.dumps(data, indent=2)[:2000]}"
        
        else:
            return f"Unsupported file type: {suffix}. Please convert to CSV, TXT, or JSON format."
    
    except Exception as e:
        return f"Error reading file: {str(e)}"

@function_tool
async def generate_financial_report(ctx: RunContextWrapper[Any], period: str = "last_month") -> str:
    """Generate a comprehensive financial report using code execution.
    
    Args:
        period: Time period for the report (last_month, last_quarter, last_year)
    """
    try:
        db = SessionLocal()
        
        # Get date range
        if period == "last_month":
            start_date = datetime.now() - timedelta(days=30)
        elif period == "last_quarter":
            start_date = datetime.now() - timedelta(days=90)
        elif period == "last_year":
            start_date = datetime.now() - timedelta(days=365)
        else:
            start_date = datetime.now() - timedelta(days=365)
        
        # Query transactions
        transactions = db.query(Transaction).filter(
            Transaction.date >= start_date
        ).all()
        
        if not transactions:
            return "No transactions found for the specified period."
        
        # Generate summary statistics
        total_amount = sum(abs(t.amount) for t in transactions)
        transaction_count = len(transactions)
        avg_transaction = total_amount / transaction_count if transaction_count > 0 else 0
        
        # Category breakdown
        categories = defaultdict(float)
        for t in transactions:
            if t.category:
                categories[t.category] += abs(t.amount)
        
        # Generate report
        report = f"""
Financial Report - {period.replace('_', ' ').title()}
{'='*50}

SUMMARY STATISTICS
Total Spending: ${total_amount:,.2f}
Number of Transactions: {transaction_count}
Average Transaction: ${avg_transaction:.2f}
Date Range: {start_date.strftime('%Y-%m-%d')} to {datetime.now().strftime('%Y-%m-%d')}

SPENDING BY CATEGORY
{'-'*30}
"""
        
        for category, amount in sorted(categories.items(), key=lambda x: x[1], reverse=True):
            percentage = (amount / total_amount * 100) if total_amount > 0 else 0
            report += f"{category:20}: ${amount:>10,.2f} ({percentage:5.1f}%)\n"
        
        # Top merchants
        merchants = defaultdict(float)
        for t in transactions:
            if t.description:
                merchants[t.description[:30]] += abs(t.amount)
        
        report += f"\nTOP MERCHANTS\n{'-'*30}\n"
        for merchant, amount in sorted(merchants.items(), key=lambda x: x[1], reverse=True)[:10]:
            report += f"{merchant:30}: ${amount:>10,.2f}\n"
        
        db.close()
        return report
    
    except Exception as e:
        return f"Error generating report: {str(e)}"

@function_tool
async def search_financial_web(query: str) -> str:
    """Search for current financial information, rates, and advice online.
    
    Args:
        query: Financial topic or question to search for
    """
    # This would typically use WebSearchTool, but we'll simulate it
    financial_topics = {
        "interest rates": "Current federal funds rate is 5.25-5.50%. Savings accounts average 0.45% APY.",
        "inflation": "Current inflation rate is approximately 3.2% year-over-year as of latest data.",
        "investment advice": "Diversified index funds are recommended for long-term growth. Consider your risk tolerance.",
        "budget tips": "Follow the 50/30/20 rule: 50% needs, 30% wants, 20% savings and debt repayment.",
        "credit score": "FICO scores range from 300-850. Above 700 is considered good, above 800 excellent.",
        "mortgage rates": "30-year fixed mortgage rates currently around 7.0-7.5% depending on credit score."
    }
    
    # Simple keyword matching for demonstration
    for topic, info in financial_topics.items():
        if topic in query.lower():
            return f"Financial Information: {info}"
    
    return f"General financial advice: Consider consulting with financial professionals for personalized guidance on: {query}"

@function_tool
async def calculate_financial_metrics(ctx: RunContextWrapper[Any]) -> str:
    """Calculate advanced financial metrics and ratios.
    
    This function performs complex calculations using financial data.
    """
    try:
        db = SessionLocal()
        
        # Get all transactions
        transactions = db.query(Transaction).all()
        
        if not transactions:
            return "No transactions available for metric calculations."
        
        # Calculate various metrics
        monthly_spending = defaultdict(float)
        income_transactions = []
        expense_transactions = []
        
        for t in transactions:
            month_key = t.date.strftime('%Y-%m')
            monthly_spending[month_key] += abs(t.amount)
            
            if t.transaction_type == 'income' or t.amount > 0:
                income_transactions.append(t)
            else:
                expense_transactions.append(t)
        
        # Calculate metrics
        total_income = sum(t.amount for t in income_transactions)
        total_expenses = sum(abs(t.amount) for t in expense_transactions)
        net_income = total_income - total_expenses
        
        # Monthly averages
        months = len(monthly_spending) if monthly_spending else 1
        avg_monthly_spending = total_expenses / months
        avg_monthly_income = total_income / months
        
        # Savings rate
        savings_rate = (net_income / total_income * 100) if total_income > 0 else 0
        
        # Category analysis
        housing_expenses = sum(abs(t.amount) for t in expense_transactions if t.category == 'Housing')
        housing_ratio = (housing_expenses / total_income * 100) if total_income > 0 else 0
        
        metrics_report = f"""
FINANCIAL METRICS ANALYSIS
{'='*40}

INCOME & EXPENSES
Total Income: ${total_income:,.2f}
Total Expenses: ${total_expenses:,.2f}
Net Income: ${net_income:,.2f}

MONTHLY AVERAGES
Avg Monthly Income: ${avg_monthly_income:,.2f}
Avg Monthly Spending: ${avg_monthly_spending:,.2f}

KEY RATIOS
Savings Rate: {savings_rate:.1f}% (Recommended: 20%+)
Housing Ratio: {housing_ratio:.1f}% (Recommended: <30%)

FINANCIAL HEALTH SCORE
"""
        
        # Simple scoring system
        score = 0
        if savings_rate >= 20:
            score += 40
        elif savings_rate >= 10:
            score += 20
        
        if housing_ratio <= 30:
            score += 30
        elif housing_ratio <= 40:
            score += 15
        
        if net_income > 0:
            score += 30
        
        health_status = "Excellent" if score >= 90 else "Good" if score >= 70 else "Fair" if score >= 50 else "Needs Improvement"
        
        metrics_report += f"Score: {score}/100 - {health_status}\n"
        
        db.close()
        return metrics_report
    
    except Exception as e:
        return f"Error calculating metrics: {str(e)}"

@function_tool 
async def export_data_analysis(format_type: str = "csv") -> str:
    """Export transaction data in various formats for external analysis.
    
    Args:
        format_type: Export format - 'csv', 'json', or 'excel'
    """
    try:
        db = SessionLocal()
        transactions = db.query(Transaction).all()
        
        if not transactions:
            return "No transactions to export."
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if format_type.lower() == 'csv':
            import pandas as pd
            
            data = [{
                'date': t.date.isoformat(),
                'amount': t.amount,
                'category': t.category or 'Uncategorized',
                'description': t.description or '',
                'merchant': t.merchant or '',
                'account': t.account or '',
                'type': t.transaction_type or 'expense'
            } for t in transactions]
            
            df = pd.DataFrame(data)
            filename = f"transactions_export_{timestamp}.csv"
            filepath = settings.EXPORT_DIR / filename
            df.to_csv(filepath, index=False)
            
            return f"Exported {len(transactions)} transactions to {filename}"
        
        elif format_type.lower() == 'json':
            data = [{
                'id': t.id,
                'date': t.date.isoformat(),
                'amount': t.amount,
                'category': t.category,
                'description': t.description,
                'merchant': t.merchant,
                'account': t.account,
                'type': t.transaction_type
            } for t in transactions]
            
            filename = f"transactions_export_{timestamp}.json"
            filepath = settings.EXPORT_DIR / filename
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            
            return f"Exported {len(transactions)} transactions to {filename}"
        
        else:
            return f"Unsupported export format: {format_type}. Use 'csv' or 'json'."
    
    except Exception as e:
        return f"Error exporting data: {str(e)}"
    finally:
        db.close()

# ================== Enhanced Agent Configurations ==================
def create_enhanced_document_processor():
    """Enhanced document processor with file reading capabilities"""
    return Agent(
        name="Enhanced Document Processor",
        model=OpenAIResponsesModel(model="gpt-4o"),
        instructions="""You are an advanced document processing specialist with enhanced file reading capabilities. 

Your expertise includes:
- Processing multiple file formats (CSV, PDF, JSON, TXT)
- Extracting structured financial data from unstructured documents
- Validating data quality and identifying inconsistencies
- Categorizing transactions using intelligent pattern recognition
- Handling various date formats and currency representations

Use the read_financial_file tool to analyze document contents before processing.""",
        tools=[read_financial_file]
    )

def create_enhanced_analyst():
    """Enhanced analyst with computational capabilities"""
    return Agent(
        name="Enhanced Financial Analyst", 
        model=OpenAIResponsesModel(model="gpt-4o"),
        instructions="""You are an advanced financial analyst with computational capabilities.

Your expertise includes:
- Complex financial metrics calculation (ratios, trends, forecasting)
- Statistical analysis of spending patterns
- Risk assessment and anomaly detection
- Performance benchmarking against financial best practices
- Data visualization and reporting

Use the calculate_financial_metrics and generate_financial_report tools for detailed analysis.""",
        tools=[
            calculate_financial_metrics,
            generate_financial_report,
            CodeInterpreterTool()  # For complex calculations
        ]
    )

def create_enhanced_advisor():
    """Enhanced advisor with web search and shell capabilities"""
    return Agent(
        name="Enhanced Financial Advisor",
        model=OpenAIResponsesModel(model="gpt-4o"),
        instructions="""You are a comprehensive financial advisor with access to current market information.

Your capabilities include:
- Real-time financial market data and rates
- Current economic trends and their impact on personal finance
- Personalized investment and savings strategies
- Budget optimization with current market conditions
- Tax planning and optimization advice
- Automated report generation and data export

Use web search for current financial information and market conditions.""",
        tools=[
            search_financial_web,
            WebSearchTool(),
            export_data_analysis
        ]
    )

def create_enhanced_orchestrator():
    """Main orchestrator with all enhanced capabilities"""
    doc_processor = create_enhanced_document_processor()
    analyst = create_enhanced_analyst()
    advisor = create_enhanced_advisor()
    
    return Agent(
        name="Enhanced Financial Expert",
        model=OpenAIResponsesModel(model="gpt-4o"),
        instructions="""You are the ultimate financial expert orchestrator with enhanced capabilities.

You coordinate between specialized agents to provide:
- Comprehensive document processing and data extraction
- Advanced financial analysis with computational tools
- Real-time market information and personalized advice
- Automated reporting and data export capabilities

Your approach:
1. Understand the user's specific financial goals and situation
2. Delegate tasks to appropriate specialist agents
3. Synthesize information from multiple sources
4. Provide actionable, personalized recommendations
5. Offer both immediate insights and long-term planning

Always ensure accuracy, consider current market conditions, and maintain the highest standards of financial advice.""",
        handoffs=[doc_processor, analyst, advisor],
        tools=[
            read_financial_file,
            calculate_financial_metrics,
            generate_financial_report,
            search_financial_web,
            export_data_analysis
        ]
    )

# ================== Example Usage Functions ==================
async def demo_enhanced_agents():
    """Demonstrate enhanced agent capabilities"""
    print("🚀 Enhanced Financial Agents Demo")
    print("=" * 50)
    
    orchestrator = create_enhanced_orchestrator()
    
    # Example 1: Comprehensive Analysis
    print("\n📊 Running comprehensive financial analysis...")
    result1 = await Runner.run(
        orchestrator,
        "Provide a comprehensive analysis of my financial health including detailed metrics, current market conditions, and personalized recommendations."
    )
    print(result1.final_output)
    
    # Example 2: Document Processing with File Reading
    print("\n📄 Document processing demo...")
    # Note: This would work with actual files in your uploads directory
    result2 = await Runner.run(
        orchestrator,
        "If there are any CSV files in the uploads directory, read and analyze them for financial insights."
    )
    print(result2.final_output)
    
    # Example 3: Market-aware recommendations
    print("\n💡 Getting market-aware recommendations...")
    result3 = await Runner.run(
        orchestrator,
        "Given current interest rates and market conditions, what are the best strategies for optimizing my savings and reducing expenses?"
    )
    print(result3.final_output)

if __name__ == '__main__':
    print("Enhanced Financial Agents with OpenAI SDK")
    print("Run demo_enhanced_agents() to see capabilities")
    
    # Uncomment to run demo
    # asyncio.run(demo_enhanced_agents())