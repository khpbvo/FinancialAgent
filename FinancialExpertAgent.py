# FinancialExpertAgent CLI Application
# Full implementation with OpenAI GPT-5 and Agents SDK

# ================== requirements.txt ==================
"""
openai>=1.40.0
click>=8.1.0
psycopg2-binary>=2.9.0
pgvector>=0.2.0
sqlalchemy>=2.0.0
pypdf>=4.0.0
pandas>=2.0.0
python-dotenv>=1.0.0
rich>=13.0.0
tabulate>=0.9.0
pydantic>=2.0.0
numpy>=1.24.0
"""

# ================== config/settings.py ==================
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

class Settings:
    # OpenAI Configuration - Using latest GPT-5 model
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL = "gpt-5-latest"  # Latest GPT-5 model
    OPENAI_EMBEDDING_MODEL = "text-embedding-3-large"
    
    # Database Configuration
    DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@localhost/financial_agent")
    
    # Application Settings
    MAX_DOCUMENT_SIZE_MB = 100
    ANALYSIS_TIMEOUT_SECONDS = 30
    EMBEDDING_DIMENSION = 3072  # For text-embedding-3-large
    
    # Paths
    BASE_DIR = Path(__file__).parent.parent
    UPLOAD_DIR = BASE_DIR / "uploads"
    EXPORT_DIR = BASE_DIR / "exports"
    
    # Agent Configuration
    AGENT_TEMPERATURE = 1  # Low temperature for financial accuracy
    MAX_COMPLETION_TOKENS = 8196
    
    # Categories for expense classification
    EXPENSE_CATEGORIES = [
        "Housing", "Transportation", "Food & Dining", "Utilities",
        "Healthcare", "Insurance", "Personal Care", "Entertainment",
        "Shopping", "Education", "Savings", "Investments", "Debt Payments",
        "Miscellaneous"
    ]

settings = Settings()

# ================== database/models.py ==================
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from pgvector.sqlalchemy import Vector
from datetime import datetime
import numpy as np

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
    transaction_type = Column(String(50))  # income, expense, transfer
    created_at = Column(DateTime, default=datetime.utcnow)
    embedding = Column(Vector(settings.EMBEDDING_DIMENSION))

class Document(Base):
    __tablename__ = 'documents'
    
    id = Column(Integer, primary_key=True)
    file_path = Column(String(500), nullable=False)
    file_name = Column(String(255), nullable=False)
    upload_date = Column(DateTime, default=datetime.utcnow)
    document_type = Column(String(100))  # bank_statement, credit_card, receipt, etc.
    processed = Column(Integer, default=0)
    metadata = Column(JSON)
    embedding = Column(Vector(settings.EMBEDDING_DIMENSION))

class AnalysisHistory(Base):
    __tablename__ = 'analysis_history'
    
    id = Column(Integer, primary_key=True)
    analysis_date = Column(DateTime, default=datetime.utcnow)
    analysis_type = Column(String(100))
    insights = Column(JSON)
    recommendations = Column(JSON)
    metrics = Column(JSON)
    period_start = Column(DateTime)
    period_end = Column(DateTime)

class UserContext(Base):
    __tablename__ = 'user_context'
    
    id = Column(Integer, primary_key=True)
    context_key = Column(String(100), unique=True)
    context_value = Column(JSON)
    embedding = Column(Vector(settings.EMBEDDING_DIMENSION))
    updated_at = Column(DateTime, default=datetime.utcnow)

# Database initialization
engine = create_engine(settings.DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_database():
    """Initialize database with PGVector extension"""
    with engine.connect() as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        conn.commit()
    Base.metadata.create_all(bind=engine)

# ================== database/vector_store.py ==================
from openai import OpenAI
import numpy as np
from typing import List, Dict, Any

class VectorStore:
    def __init__(self):
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
        self.session = SessionLocal()
    
    def create_embedding(self, text: str) -> np.ndarray:
        """Create embedding using OpenAI's latest embedding model"""
        response = self.client.embeddings.create(
            model=settings.OPENAI_EMBEDDING_MODEL,
            input=text
        )
        return np.array(response.data[0].embedding)
    
    def store_transaction_with_embedding(self, transaction_data: Dict[str, Any]):
        """Store transaction with its embedding"""
        description = f"{transaction_data['description']} {transaction_data['category']} {transaction_data['amount']}"
        embedding = self.create_embedding(description)
        
        transaction = Transaction(
            **transaction_data,
            embedding=embedding
        )
        self.session.add(transaction)
        self.session.commit()
        return transaction.id
    
    def similarity_search(self, query: str, limit: int = 10) -> List[Transaction]:
        """Find similar transactions using vector similarity"""
        query_embedding = self.create_embedding(query)
        
        # Using PGVector's similarity search
        similar_transactions = self.session.query(Transaction).order_by(
            Transaction.embedding.l2_distance(query_embedding)
        ).limit(limit).all()
        
        return similar_transactions
    
    def store_context(self, key: str, value: Any):
        """Store user context with embedding"""
        embedding = self.create_embedding(str(value))
        
        context = self.session.query(UserContext).filter_by(context_key=key).first()
        if context:
            context.context_value = value
            context.embedding = embedding
            context.updated_at = datetime.utcnow()
        else:
            context = UserContext(
                context_key=key,
                context_value=value,
                embedding=embedding
            )
            self.session.add(context)
        
        self.session.commit()

# ================== tools/document_parser.py ==================
import pypdf
import pandas as pd
import re
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime

class DocumentParser:
    """Parse various financial document formats"""
    
    def __init__(self):
        self.supported_formats = ['.pdf', '.csv', '.txt', '.xlsx']
    
    def parse_document(self, file_path: str) -> Dict[str, Any]:
        """Main parsing method that routes to appropriate parser"""
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        suffix = path.suffix.lower()
        
        if suffix == '.pdf':
            return self.parse_pdf(file_path)
        elif suffix == '.csv':
            return self.parse_csv(file_path)
        elif suffix in ['.xlsx', '.xls']:
            return self.parse_excel(file_path)
        elif suffix == '.txt':
            return self.parse_text(file_path)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")
    
    def parse_pdf(self, file_path: str) -> Dict[str, Any]:
        """Parse PDF financial documents"""
        transactions = []
        metadata = {}
        
        with open(file_path, 'rb') as file:
            pdf_reader = pypdf.PdfReader(file)
            text = ""
            
            for page in pdf_reader.pages:
                text += page.extract_text()
            
            # Extract transactions using regex patterns
            # Pattern for common transaction formats
            transaction_pattern = r'(\d{1,2}/\d{1,2}/\d{2,4})\s+([^\$]+)\s+\$?([\d,]+\.?\d*)'
            
            matches = re.findall(transaction_pattern, text)
            
            for match in matches:
                date_str, description, amount = match
                transactions.append({
                    'date': self._parse_date(date_str),
                    'description': description.strip(),
                    'amount': float(amount.replace(',', ''))
                })
            
            # Extract account information
            account_pattern = r'Account\s*(?:Number|#)?\s*:?\s*([\d\-X]+)'
            account_match = re.search(account_pattern, text)
            if account_match:
                metadata['account'] = account_match.group(1)
            
            # Detect document type
            if 'credit card' in text.lower():
                metadata['document_type'] = 'credit_card_statement'
            elif 'checking' in text.lower() or 'savings' in text.lower():
                metadata['document_type'] = 'bank_statement'
            else:
                metadata['document_type'] = 'general_financial'
        
        return {
            'transactions': transactions,
            'metadata': metadata,
            'raw_text': text[:5000]  # Store first 5000 chars for context
        }
    
    def parse_csv(self, file_path: str) -> Dict[str, Any]:
        """Parse CSV financial data"""
        df = pd.read_csv(file_path)
        
        # Detect and standardize column names
        column_mapping = {
            'date': ['date', 'transaction date', 'trans date', 'posted date'],
            'description': ['description', 'merchant', 'payee', 'details'],
            'amount': ['amount', 'debit', 'credit', 'transaction amount'],
            'category': ['category', 'type', 'classification']
        }
        
        standardized_df = pd.DataFrame()
        
        for standard_col, possible_names in column_mapping.items():
            for col in df.columns:
                if col.lower() in possible_names:
                    standardized_df[standard_col] = df[col]
                    break
        
        transactions = standardized_df.to_dict('records')
        
        return {
            'transactions': transactions,
            'metadata': {'document_type': 'csv_import', 'columns': list(df.columns)},
            'raw_text': None
        }
    
    def parse_excel(self, file_path: str) -> Dict[str, Any]:
        """Parse Excel financial data"""
        df = pd.read_excel(file_path, sheet_name=None)
        
        all_transactions = []
        for sheet_name, sheet_df in df.items():
            # Process each sheet similar to CSV
            transactions = sheet_df.to_dict('records')
            all_transactions.extend(transactions)
        
        return {
            'transactions': all_transactions,
            'metadata': {'document_type': 'excel_import', 'sheets': list(df.keys())},
            'raw_text': None
        }
    
    def parse_text(self, file_path: str) -> Dict[str, Any]:
        """Parse plain text financial documents"""
        with open(file_path, 'r') as file:
            text = file.read()
        
        # Similar pattern matching as PDF
        transactions = []
        transaction_pattern = r'(\d{1,2}/\d{1,2}/\d{2,4})\s+([^\$]+)\s+\$?([\d,]+\.?\d*)'
        matches = re.findall(transaction_pattern, text)
        
        for match in matches:
            date_str, description, amount = match
            transactions.append({
                'date': self._parse_date(date_str),
                'description': description.strip(),
                'amount': float(amount.replace(',', ''))
            })
        
        return {
            'transactions': transactions,
            'metadata': {'document_type': 'text_document'},
            'raw_text': text[:5000]
        }
    
    def _parse_date(self, date_str: str) -> datetime:
        """Parse various date formats"""
        date_formats = [
            '%m/%d/%Y', '%m/%d/%y', '%Y-%m-%d',
            '%d/%m/%Y', '%d/%m/%y', '%m-%d-%Y'
        ]
        
        for fmt in date_formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        
        return datetime.now()  # Default to current date if parsing fails

# ================== tools/transaction_analyzer.py ==================
from typing import List, Dict, Any
from collections import defaultdict
from datetime import datetime, timedelta

class TransactionAnalyzer:
    """Analyze and categorize financial transactions"""
    
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
        self.category_keywords = {
            'Food & Dining': ['restaurant', 'cafe', 'coffee', 'pizza', 'burger', 'grocery', 'food'],
            'Transportation': ['uber', 'lyft', 'gas', 'parking', 'toll', 'transit', 'metro'],
            'Shopping': ['amazon', 'walmart', 'target', 'store', 'shop', 'mall'],
            'Utilities': ['electric', 'water', 'gas', 'internet', 'phone', 'cable'],
            'Entertainment': ['netflix', 'spotify', 'movie', 'theater', 'concert', 'game'],
            'Healthcare': ['doctor', 'hospital', 'pharmacy', 'medical', 'health', 'dental'],
            'Housing': ['rent', 'mortgage', 'property', 'maintenance', 'repair'],
            'Insurance': ['insurance', 'premium', 'coverage'],
            'Education': ['tuition', 'school', 'course', 'training', 'book'],
            'Savings': ['savings', 'deposit', 'investment', 'retirement', '401k']
        }
    
    def categorize_transaction(self, description: str, amount: float) -> str:
        """Categorize a transaction based on description and amount"""
        description_lower = description.lower()
        
        # First try keyword matching
        for category, keywords in self.category_keywords.items():
            for keyword in keywords:
                if keyword in description_lower:
                    return category
        
        # If no keyword match, use similarity search from past categorized transactions
        similar_transactions = self.vector_store.similarity_search(description, limit=5)
        if similar_transactions and similar_transactions[0].category:
            return similar_transactions[0].category
        
        # Default category based on amount
        if amount > 1000:
            return 'Housing'
        elif amount > 100:
            return 'Shopping'
        else:
            return 'Miscellaneous'
    
    def analyze_spending_patterns(self, transactions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze spending patterns from transactions"""
        category_totals = defaultdict(float)
        monthly_totals = defaultdict(float)
        merchant_frequency = defaultdict(int)
        
        for transaction in transactions:
            category = transaction.get('category', 'Uncategorized')
            amount = transaction['amount']
            date = transaction['date']
            merchant = transaction.get('merchant', transaction.get('description', 'Unknown'))
            
            category_totals[category] += amount
            
            month_key = f"{date.year}-{date.month:02d}"
            monthly_totals[month_key] += amount
            
            merchant_frequency[merchant] += 1
        
        # Calculate statistics
        total_spending = sum(category_totals.values())
        avg_monthly_spending = sum(monthly_totals.values()) / max(len(monthly_totals), 1)
        
        # Find top merchants
        top_merchants = sorted(merchant_frequency.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # Category percentages
        category_percentages = {
            cat: (amount / total_spending * 100) if total_spending > 0 else 0
            for cat, amount in category_totals.items()
        }
        
        return {
            'total_spending': total_spending,
            'average_monthly_spending': avg_monthly_spending,
            'category_totals': dict(category_totals),
            'category_percentages': category_percentages,
            'monthly_totals': dict(monthly_totals),
            'top_merchants': top_merchants,
            'transaction_count': len(transactions)
        }
    
    def detect_anomalies(self, transactions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect unusual transactions"""
        anomalies = []
        
        # Calculate statistics
        amounts = [t['amount'] for t in transactions]
        if not amounts:
            return []
        
        mean_amount = sum(amounts) / len(amounts)
        std_amount = (sum((x - mean_amount) ** 2 for x in amounts) / len(amounts)) ** 0.5
        
        # Detect outliers (transactions > 3 standard deviations from mean)
        for transaction in transactions:
            if abs(transaction['amount'] - mean_amount) > 3 * std_amount:
                anomalies.append({
                    'transaction': transaction,
                    'reason': 'Unusual amount',
                    'deviation': abs(transaction['amount'] - mean_amount) / std_amount
                })
        
        # Detect duplicate transactions
        seen = {}
        for transaction in transactions:
            key = (transaction['amount'], transaction.get('description', ''))
            if key in seen:
                time_diff = abs((transaction['date'] - seen[key]['date']).days)
                if time_diff <= 1:  # Same or consecutive day
                    anomalies.append({
                        'transaction': transaction,
                        'reason': 'Possible duplicate',
                        'similar_to': seen[key]
                    })
            else:
                seen[key] = transaction
        
        return anomalies

# ================== tools/insight_generator.py ==================
class InsightGenerator:
    """Generate financial insights and recommendations using GPT-5"""
    
    def __init__(self, client: OpenAI):
        self.client = client
    
    def generate_insights(self, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate insights using GPT-5"""
        
        # Prepare context for GPT-5
        context = f"""
        Financial Analysis Summary:
        - Total Spending: ${analysis_data['total_spending']:,.2f}
        - Average Monthly Spending: ${analysis_data['average_monthly_spending']:,.2f}
        - Number of Transactions: {analysis_data['transaction_count']}
        
        Category Breakdown:
        {self._format_categories(analysis_data['category_percentages'])}
        
        Top Merchants:
        {self._format_merchants(analysis_data['top_merchants'])}
        """
        
        # Use GPT-5 to generate insights
        response = self.client.chat.completions.create(
            model=settings.OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You are a financial expert providing personalized insights."},
                {"role": "user", "content": f"Analyze this financial data and provide 5 key insights and 5 actionable recommendations:\n{context}"}
            ],
            temperature=settings.AGENT_TEMPERATURE,
            max_tokens=settings.MAX_COMPLETION_TOKENS
        )
        
        insights_text = response.choices[0].message.content
        
        # Parse insights and recommendations
        insights = self._parse_insights(insights_text)
        
        return {
            'insights': insights['insights'],
            'recommendations': insights['recommendations'],
            'risk_assessment': self._assess_financial_risk(analysis_data),
            'savings_opportunities': self._identify_savings(analysis_data)
        }
    
    def _format_categories(self, categories: Dict[str, float]) -> str:
        """Format category data for GPT-5"""
        return '\n'.join([f"- {cat}: {pct:.1f}%" for cat, pct in categories.items()])
    
    def _format_merchants(self, merchants: List[tuple]) -> str:
        """Format merchant data for GPT-5"""
        return '\n'.join([f"- {merchant}: {count} transactions" for merchant, count in merchants[:5]])
    
    def _parse_insights(self, text: str) -> Dict[str, List[str]]:
        """Parse GPT-5 response into structured insights"""
        insights = []
        recommendations = []
        
        lines = text.split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            if 'insight' in line.lower():
                current_section = 'insights'
            elif 'recommendation' in line.lower():
                current_section = 'recommendations'
            elif line and current_section:
                if current_section == 'insights':
                    insights.append(line.lstrip('- •123456789.'))
                else:
                    recommendations.append(line.lstrip('- •123456789.'))
        
        return {
            'insights': insights[:5],
            'recommendations': recommendations[:5]
        }
    
    def _assess_financial_risk(self, analysis_data: Dict[str, Any]) -> str:
        """Assess financial risk level"""
        total_spending = analysis_data['total_spending']
        
        # Simple risk assessment based on spending patterns
        housing_pct = analysis_data['category_percentages'].get('Housing', 0)
        savings_pct = analysis_data['category_percentages'].get('Savings', 0)
        
        if housing_pct > 40:
            return "High - Housing costs exceed recommended 30% threshold"
        elif savings_pct < 10:
            return "Medium - Savings rate below recommended 20%"
        else:
            return "Low - Spending patterns appear balanced"
    
    def _identify_savings(self, analysis_data: Dict[str, Any]) -> List[str]:
        """Identify potential savings opportunities"""
        opportunities = []
        
        categories = analysis_data['category_percentages']
        
        if categories.get('Food & Dining', 0) > 15:
            opportunities.append("Reduce dining out expenses by cooking more at home")
        
        if categories.get('Entertainment', 0) > 10:
            opportunities.append("Review and cancel unused subscriptions")
        
        if categories.get('Shopping', 0) > 20:
            opportunities.append("Implement a 24-hour rule before non-essential purchases")
        
        # Check for frequent small transactions
        if analysis_data['transaction_count'] > 100:
            opportunities.append("Consolidate small purchases to reduce impulse buying")
        
        return opportunities[:3]

# ================== agents/financial_expert.py ==================
from openai import OpenAI
from typing import Dict, Any, List, Optional
import json

class FinancialExpertAgent:
    """Main Financial Expert Agent using OpenAI GPT-5 and Agents SDK"""
    
    def __init__(self):
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
        self.vector_store = VectorStore()
        self.document_parser = DocumentParser()
        self.transaction_analyzer = TransactionAnalyzer(self.vector_store)
        self.insight_generator = InsightGenerator(self.client)
        
        # Initialize OpenAI Agent with GPT-5
        self.agent = self._create_agent()
    
    def _create_agent(self):
        """Create OpenAI Agent with financial expertise"""
        
        # Define agent tools using OpenAI Agents SDK syntax
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "parse_financial_document",
                    "description": "Parse and extract data from financial documents",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_path": {"type": "string", "description": "Path to the financial document"}
                        },
                        "required": ["file_path"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "analyze_transactions",
                    "description": "Analyze financial transactions for patterns and insights",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "period": {"type": "string", "description": "Analysis period (e.g., 'last_month', 'last_quarter')"}
                        },
                        "required": ["period"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "generate_recommendations",
                    "description": "Generate personalized financial recommendations",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "focus_area": {"type": "string", "description": "Area to focus on (e.g., 'savings', 'debt_reduction', 'investment')"}
                        }
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "search_transactions",
                    "description": "Search for similar transactions using semantic search",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Search query"}
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function", 
                "function": {
                    "name": "web_search",
                    "description": "Search the web for financial information and advice",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Search query"}
                        },
                        "required": ["query"]
                    }
                }
            }
        ]
        
        # Create agent configuration
        agent_config = {
            "model": settings.OPENAI_MODEL,
            "tools": tools,
            "temperature": settings.AGENT_TEMPERATURE,
            "instructions": """You are a highly skilled financial expert agent. Your role is to:
                1. Analyze financial documents with precision
                2. Provide actionable insights on spending patterns
                3. Generate personalized recommendations for financial health
                4. Help users understand their financial situation
                5. Maintain strict privacy and security of financial data
                
                Always be accurate with numbers, clear in explanations, and proactive in identifying opportunities for financial improvement."""
        }
        
        return agent_config
    
    def process_document(self, file_path: str) -> Dict[str, Any]:
        """Process a financial document"""
        # Parse document
        parsed_data = self.document_parser.parse_document(file_path)
        
        # Store document in database
        db = SessionLocal()
        document = Document(
            file_path=file_path,
            file_name=Path(file_path).name,
            document_type=parsed_data['metadata'].get('document_type', 'unknown'),
            metadata=parsed_data['metadata'],
            processed=1
        )
        
        if parsed_data.get('raw_text'):
            document.embedding = self.vector_store.create_embedding(parsed_data['raw_text'][:1000])
        
        db.add(document)
        db.commit()
        
        # Process transactions
        transactions_processed = 0
        for transaction_data in parsed_data['transactions']:
            # Categorize transaction
            category = self.transaction_analyzer.categorize_transaction(
                transaction_data.get('description', ''),
                transaction_data.get('amount', 0)
            )
            
            transaction_data['category'] = category
            transaction_data['source_document'] = file_path
            
            # Store with embedding
            self.vector_store.store_transaction_with_embedding(transaction_data)
            transactions_processed += 1
        
        db.close()
        
        return {
            'status': 'success',
            'document_type': parsed_data['metadata'].get('document_type'),
            'transactions_processed': transactions_processed,
            'file_path': file_path
        }
    
    def analyze_financial_health(self, period: Optional[str] = None) -> Dict[str, Any]:
        """Comprehensive financial health analysis"""
        db = SessionLocal()
        
        # Get transactions for period
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
        
        # Convert to dict format
        transaction_dicts = [
            {
                'date': t.date,
                'amount': t.amount,
                'category': t.category,
                'description': t.description,
                'merchant': t.merchant
            }
            for t in transactions
        ]
        
        # Analyze patterns
        analysis = self.transaction_analyzer.analyze_spending_patterns(transaction_dicts)
        
        # Detect anomalies
        anomalies = self.transaction_analyzer.detect_anomalies(transaction_dicts)
        
        # Generate insights
        insights = self.insight_generator.generate_insights(analysis)
        
        # Store analysis history
        history = AnalysisHistory(
            analysis_type='comprehensive',
            insights=insights['insights'],
            recommendations=insights['recommendations'],
            metrics=analysis,
            period_start=start_date if period else None,
            period_end=datetime.now()
        )
        db.add(history)
        db.commit()
        db.close()
        
        return {
            'analysis': analysis,
            'insights': insights,
            'anomalies': anomalies,
            'period': period or 'all_time',
            'generated_at': datetime.now().isoformat()
        }
    
    def chat_with_agent(self, user_input: str) -> str:
        """Interactive chat with the financial expert agent"""
        
        # Retrieve context from database
        context = self.vector_store.similarity_search(user_input, limit=5)
        
        context_str = "\n".join([
            f"- {t.date}: {t.description} (${t.amount}) [{t.category}]"
            for t in context
        ])
        
        # Create conversation with GPT-5
        messages = [
            {
                "role": "system",
                "content": f"""You are a financial expert assistant. Use this context about the user's finances:
                
                Recent Transactions:
                {context_str}
                
                Provide helpful, accurate financial advice based on their data."""
            },
            {
                "role": "user",
                "content": user_input
            }
        ]
        
        # Get response from GPT-5
        response = self.client.chat.completions.create(
            model=settings.OPENAI_MODEL,
            messages=messages,
            tools=self.agent['tools'],
            tool_choice="auto",
            temperature=settings.AGENT_TEMPERATURE
        )
        
        # Handle tool calls if any
        if response.choices[0].message.tool_calls:
            tool_results = self._handle_tool_calls(response.choices[0].message.tool_calls)
            
            # Get final response with tool results
            messages.append(response.choices[0].message)
            messages.append({
                "role": "tool",
                "content": json.dumps(tool_results)
            })
            
            final_response = self.client.chat.completions.create(
                model=settings.OPENAI_MODEL,
                messages=messages,
                temperature=settings.AGENT_TEMPERATURE
            )
            
            return final_response.choices[0].message.content
        
        return response.choices[0].message.content
    
    def _handle_tool_calls(self, tool_calls) -> List[Dict[str, Any]]:
        """Handle agent tool calls"""
        results = []
        
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)
            
            if function_name == "parse_financial_document":
                result = self.process_document(arguments['file_path'])
            elif function_name == "analyze_transactions":
                result = self.analyze_financial_health(arguments.get('period'))
            elif function_name == "search_transactions":
                transactions = self.vector_store.similarity_search(arguments['query'])
                result = [
                    {
                        'date': t.date.isoformat(),
                        'amount': t.amount,
                        'description': t.description,
                        'category': t.category
                    }
                    for t in transactions
                ]
            elif function_name == "web_search":
                # Simulate web search (would integrate with real search API)
                result = {
                    'results': f"Financial advice for: {arguments['query']}"
                }
            else:
                result = {"error": f"Unknown tool: {function_name}"}
            
            results.append({
                "tool_call_id": tool_call.id,
                "result": result
            })
        
        return results

# ================== main.py ==================
import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import track
from pathlib import Path
import json
from datetime import datetime

console = Console()

@click.group()
def cli():
    """FinancialExpertAgent - AI-Powered Financial Analysis CLI"""
    # Initialize database on first run
    init_database()
    settings.UPLOAD_DIR.mkdir(exist_ok=True)
    settings.EXPORT_DIR.mkdir(exist_ok=True)

@cli.command()
@click.argument('file_path', type=click.Path(exists=True))
def analyze(file_path):
    """Process and analyze a financial document"""
    console.print(f"\n[bold cyan]Processing document:[/bold cyan] {file_path}")
    
    agent = FinancialExpertAgent()
    
    with console.status("[bold green]Analyzing document...") as status:
        try:
            result = agent.process_document(file_path)
            
            console.print(Panel(
                f"✅ Document processed successfully!\n"
                f"Type: {result['document_type']}\n"
                f"Transactions: {result['transactions_processed']}",
                title="Analysis Complete",
                border_style="green"
            ))
            
            # Run immediate analysis
            status.update("[bold green]Generating insights...")
            analysis = agent.analyze_financial_health('last_month')
            
            # Display insights
            console.print("\n[bold cyan]Key Insights:[/bold cyan]")
            for i, insight in enumerate(analysis['insights']['insights'], 1):
                console.print(f"  {i}. {insight}")
            
            console.print("\n[bold cyan]Recommendations:[/bold cyan]")
            for i, rec in enumerate(analysis['insights']['recommendations'], 1):
                console.print(f"  {i}. {rec}")
            
        except Exception as e:
            console.print(f"[bold red]Error:[/bold red] {str(e)}")

@cli.command()
@click.option('--period', default='last_month', 
              type=click.Choice(['last_month', 'last_quarter', 'last_year', 'all']),
              help='Analysis period')
def summary(period):
    """Show financial overview and summary"""
    agent = FinancialExpertAgent()
    
    with console.status("[bold green]Generating financial summary...") as status:
        analysis = agent.analyze_financial_health(period if period != 'all' else None)
        
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
@click.option('--category', default=None, help='Specific category to analyze')
def trends(category):
    """Display spending trends and patterns"""
    agent = FinancialExpertAgent()
    db = SessionLocal()
    
    # Get monthly trends
    if category:
        transactions = db.query(Transaction).filter(
            Transaction.category == category
        ).all()
        title = f"Trends for {category}"
    else:
        transactions = db.query(Transaction).all()
        title = "Overall Spending Trends"
    
    # Group by month
    monthly_data = defaultdict(float)
    for t in transactions:
        month_key = t.date.strftime('%Y-%m')
        monthly_data[month_key] += t.amount
    
    # Create trends table
    table = Table(title=title)
    table.add_column("Month", style="cyan")
    table.add_column("Amount", justify="right", style="green")
    table.add_column("Change", justify="right", style="yellow")
    
    sorted_months = sorted(monthly_data.items())
    prev_amount = 0
    
    for month, amount in sorted_months[-6:]:  # Last 6 months
        change = ((amount - prev_amount) / prev_amount * 100) if prev_amount > 0 else 0
        change_str = f"{change:+.1f}%" if prev_amount > 0 else "—"
        
        table.add_row(
            month,
            f"${amount:,.2f}",
            change_str
        )
        prev_amount = amount
    
    console.print(table)
    db.close()

@cli.command()
def recommendations():
    """Get personalized financial recommendations"""
    agent = FinancialExpertAgent()
    
    with console.status("[bold green]Generating personalized recommendations..."):
        analysis = agent.analyze_financial_health('last_quarter')
        
        console.print(Panel(
            "[bold cyan]Personalized Financial Recommendations[/bold cyan]",
            border_style="cyan"
        ))
        
        # Display recommendations
        for i, rec in enumerate(analysis['insights']['recommendations'], 1):
            console.print(f"\n[bold]{i}.[/bold] {rec}")
        
        # Display savings opportunities
        console.print("\n[bold cyan]Savings Opportunities:[/bold cyan]")
        for opp in analysis['insights']['savings_opportunities']:
            console.print(f"  💰 {opp}")

@cli.command()
@click.option('--format', type=click.Choice(['json', 'csv', 'pdf']), default='json',
              help='Export format')
@click.option('--period', default='last_month',
              type=click.Choice(['last_month', 'last_quarter', 'last_year', 'all']),
              help='Export period')
def export(format, period):
    """Export analysis results"""
    agent = FinancialExpertAgent()
    
    with console.status(f"[bold green]Exporting data as {format}..."):
        analysis = agent.analyze_financial_health(period if period != 'all' else None)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"financial_analysis_{period}_{timestamp}.{format}"
        filepath = settings.EXPORT_DIR / filename
        
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
    db = SessionLocal()
    
    # Get statistics
    total_transactions = db.query(Transaction).count()
    total_documents = db.query(Document).count()
    total_analyses = db.query(AnalysisHistory).count()
    
    # Get recent activity
    recent_docs = db.query(Document).order_by(Document.upload_date.desc()).limit(5).all()
    
    # Display status
    console.print(Panel(
        f"[bold cyan]System Status[/bold cyan]\n\n"
        f"Database: [green]Connected[/green]\n"
        f"Model: [green]{settings.OPENAI_MODEL}[/green]\n"
        f"Embedding Model: [green]{settings.OPENAI_EMBEDDING_MODEL}[/green]\n\n"
        f"[bold]Statistics:[/bold]\n"
        f"Total Transactions: {total_transactions}\n"
        f"Total Documents: {total_documents}\n"
        f"Total Analyses: {total_analyses}",
        title="📊 FinancialExpertAgent Status",
        border_style="blue"
    ))
    
    if recent_docs:
        console.print("\n[bold cyan]Recent Documents:[/bold cyan]")
        for doc in recent_docs:
            console.print(f"  • {doc.file_name} ({doc.document_type}) - {doc.upload_date.strftime('%Y-%m-%d')}")
    
    db.close()

if __name__ == '__main__':
    cli()