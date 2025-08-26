"""Specialized document processing agents used as tools."""
from __future__ import annotations
from typing import List, Dict, Any, Optional
from pathlib import Path
import csv
import json

from agents import Agent, ModelSettings, Runner, RunContextWrapper, RunResult, function_tool
from pydantic import BaseModel, Field
import PyPDF2

from .context import RunDeps
from .models import Transaction, TransactionList, CategorySpending


class CSVAnalysisResult(BaseModel):
    """Structured result from CSV analysis."""
    file_name: str = Field(description="Name of the analyzed file")
    row_count: int = Field(description="Number of rows processed")
    columns: List[str] = Field(description="Column names found")
    date_range: Optional[str] = Field(description="Date range of transactions")
    total_amount: Optional[float] = Field(description="Total transaction amount")
    categories_found: List[str] = Field(description="Unique categories identified")
    parsing_errors: List[str] = Field(description="Any errors encountered")


class PDFAnalysisResult(BaseModel):
    """Structured result from PDF analysis."""
    file_name: str = Field(description="Name of the analyzed file")
    page_count: int = Field(description="Number of pages processed")
    transactions_found: int = Field(description="Number of transactions extracted")
    total_amount: Optional[float] = Field(description="Total amount identified")
    account_info: Dict[str, str] = Field(description="Account information extracted")
    extraction_confidence: float = Field(description="Confidence score of extraction (0-1)")


def build_csv_analyzer_agent() -> Agent[RunDeps]:
    """Build a specialized CSV analysis agent."""
    
    instructions = """You are a CSV analysis specialist. Your job is to:
    1. Parse CSV files and identify their structure
    2. Detect column mappings (date, amount, description, etc.)
    3. Identify the CSV format (ING, generic bank, custom)
    4. Extract and validate transaction data
    5. Categorize transactions based on descriptions
    6. Provide detailed analysis of the data quality
    
    Always return structured results with clear metrics."""
    
    return Agent[RunDeps](
        name="CSVAnalyzer",
        instructions=instructions,
        model="gpt-4o",
        model_settings=ModelSettings(),
        output_type=CSVAnalysisResult,
        tools=[],  # Pure analysis agent, no tools needed
    )


def build_pdf_analyzer_agent() -> Agent[RunDeps]:
    """Build a specialized PDF analysis agent."""
    
    instructions = """You are a PDF bank statement analysis expert. Your job is to:
    1. Extract text from PDF bank statements
    2. Identify the bank and statement format
    3. Parse transaction tables with high accuracy
    4. Extract account information (number, holder, period)
    5. Handle multi-column layouts and page breaks
    6. Validate extracted amounts and dates
    
    Focus on accuracy over speed. Return structured data with confidence scores."""
    
    return Agent[RunDeps](
        name="PDFAnalyzer",
        instructions=instructions,
        model="gpt-4o",
        model_settings=ModelSettings(
            temperature=0.1,  # Lower temperature for more consistent extraction
        ),
        output_type=PDFAnalysisResult,
        tools=[],
    )


@function_tool
async def analyze_csv_with_agent(
    ctx: RunContextWrapper[RunDeps],
    file_path: str,
    preview_rows: int = 5
) -> CSVAnalysisResult:
    """Use the CSV analyzer agent to deeply analyze a CSV file.
    
    Args:
        file_path: Path to the CSV file to analyze
        preview_rows: Number of rows to preview for analysis
    """
    csv_agent = build_csv_analyzer_agent()
    
    # Read CSV preview
    path = Path(file_path)
    if not path.exists():
        return CSVAnalysisResult(
            file_name=file_path,
            row_count=0,
            columns=[],
            categories_found=[],
            parsing_errors=[f"File not found: {file_path}"]
        )
    
    try:
        with open(path, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            rows = []
            for i, row in enumerate(reader):
                if i < preview_rows:
                    rows.append(row)
                else:
                    break
            
            # Create analysis prompt
            prompt = f"""Analyze this CSV file structure and content:
            
File: {path.name}
Columns: {', '.join(reader.fieldnames or [])}

Sample rows (first {len(rows)}):
{json.dumps(rows, indent=2)}

Identify:
1. The CSV format/bank type
2. Column mappings for financial data
3. Date format used
4. Currency and amount format
5. Any data quality issues"""
            
            # Run the agent
            result = await Runner.run(
                csv_agent,
                input=prompt,
                context=ctx.context,
                max_turns=3
            )
            
            # Extract structured result
            if isinstance(result.final_output, CSVAnalysisResult):
                return result.final_output
            else:
                # Fallback to basic analysis
                return CSVAnalysisResult(
                    file_name=path.name,
                    row_count=len(rows),
                    columns=list(reader.fieldnames or []),
                    categories_found=[],
                    parsing_errors=[]
                )
                
    except Exception as e:
        return CSVAnalysisResult(
            file_name=file_path,
            row_count=0,
            columns=[],
            categories_found=[],
            parsing_errors=[str(e)]
        )


@function_tool
async def analyze_pdf_with_agent(
    ctx: RunContextWrapper[RunDeps],
    file_path: str,
    max_pages: int = 10
) -> PDFAnalysisResult:
    """Use the PDF analyzer agent to extract data from PDF statements.
    
    Args:
        file_path: Path to the PDF file to analyze
        max_pages: Maximum number of pages to process
    """
    pdf_agent = build_pdf_analyzer_agent()
    
    path = Path(file_path)
    if not path.exists():
        return PDFAnalysisResult(
            file_name=file_path,
            page_count=0,
            transactions_found=0,
            account_info={},
            extraction_confidence=0.0
        )
    
    try:
        # Extract text from PDF
        text_content = []
        with open(path, 'rb') as f:
            pdf_reader = PyPDF2.PdfReader(f)
            page_count = min(len(pdf_reader.pages), max_pages)
            
            for i in range(page_count):
                page = pdf_reader.pages[i]
                text_content.append(f"--- Page {i+1} ---\n{page.extract_text()}")
        
        # Create extraction prompt
        prompt = f"""Extract financial data from this PDF bank statement:
        
File: {path.name}
Pages: {page_count}

Content:
{chr(10).join(text_content[:3])}  # First 3 pages for context

Extract:
1. Account holder information
2. Statement period
3. All transactions with dates and amounts
4. Account balance information
5. Any fees or charges

Return structured data with confidence scores."""
        
        # Run the agent
        result = await Runner.run(
            pdf_agent,
            input=prompt,
            context=ctx.context,
            max_turns=3
        )
        
        # Extract structured result
        if isinstance(result.final_output, PDFAnalysisResult):
            return result.final_output
        else:
            # Fallback result
            return PDFAnalysisResult(
                file_name=path.name,
                page_count=page_count,
                transactions_found=0,
                account_info={},
                extraction_confidence=0.5
            )
            
    except Exception as e:
        return PDFAnalysisResult(
            file_name=file_path,
            page_count=0,
            transactions_found=0,
            account_info={},
            extraction_confidence=0.0
        )


def build_document_orchestrator() -> Agent[RunDeps]:
    """Build an orchestrator that uses document analysis agents as tools."""
    
    csv_agent = build_csv_analyzer_agent()
    pdf_agent = build_pdf_analyzer_agent()
    
    instructions = """You are a document processing orchestrator. 
    You coordinate specialized agents to analyze financial documents:
    - Use the CSV Analyzer for CSV files
    - Use the PDF Analyzer for PDF statements
    - Combine results from multiple documents for comprehensive analysis
    
    Always validate and cross-reference extracted data."""
    
    return Agent[RunDeps](
        name="DocumentOrchestrator",
        instructions=instructions,
        model="gpt-4o",
        model_settings=ModelSettings(),
        tools=[
            # Agents as tools
            csv_agent.as_tool(
                tool_name="analyze_csv",
                tool_description="Deeply analyze CSV files to extract financial data",
                custom_output_extractor=extract_csv_analysis
            ),
            pdf_agent.as_tool(
                tool_name="analyze_pdf",
                tool_description="Extract and parse data from PDF bank statements",
                custom_output_extractor=extract_pdf_analysis
            ),
            # Function tools for direct analysis
            analyze_csv_with_agent,
            analyze_pdf_with_agent,
        ],
    )


async def extract_csv_analysis(result: RunResult) -> str:
    """Extract CSV analysis results in a formatted way."""
    for item in reversed(result.new_items):
        if hasattr(item, 'output') and isinstance(item.output, CSVAnalysisResult):
            res = item.output
            return f"""CSV Analysis Complete:
- File: {res.file_name}
- Rows: {res.row_count}
- Columns: {', '.join(res.columns)}
- Categories: {len(res.categories_found)} unique
- Date Range: {res.date_range or 'Unknown'}
- Total: €{res.total_amount or 0:,.2f}
- Errors: {len(res.parsing_errors)}"""
    return str(result.final_output)


async def extract_pdf_analysis(result: RunResult) -> str:
    """Extract PDF analysis results in a formatted way."""
    for item in reversed(result.new_items):
        if hasattr(item, 'output') and isinstance(item.output, PDFAnalysisResult):
            res = item.output
            return f"""PDF Analysis Complete:
- File: {res.file_name}
- Pages: {res.page_count}
- Transactions: {res.transactions_found}
- Total: €{res.total_amount or 0:,.2f}
- Confidence: {res.extraction_confidence:.1%}
- Account: {res.account_info.get('number', 'Unknown')}"""
    return str(result.final_output)


class BatchDocumentProcessor:
    """Process multiple documents in parallel using agents."""
    
    def __init__(self, context: RunDeps):
        self.context = context
        self.csv_agent = build_csv_analyzer_agent()
        self.pdf_agent = build_pdf_analyzer_agent()
    
    async def process_batch(
        self,
        csv_files: List[Path],
        pdf_files: List[Path]
    ) -> Dict[str, Any]:
        """Process multiple documents in parallel.
        
        Args:
            csv_files: List of CSV file paths
            pdf_files: List of PDF file paths
            
        Returns:
            Dictionary with processing results
        """
        import asyncio
        
        # Create tasks for parallel processing
        csv_tasks = [
            self._process_csv(file) for file in csv_files
        ]
        pdf_tasks = [
            self._process_pdf(file) for file in pdf_files
        ]
        
        # Run all tasks in parallel
        all_tasks = csv_tasks + pdf_tasks
        results = await asyncio.gather(*all_tasks, return_exceptions=True)
        
        # Separate results
        csv_results = results[:len(csv_files)]
        pdf_results = results[len(csv_files):]
        
        # Compile summary
        successful_csv = sum(1 for r in csv_results if not isinstance(r, Exception))
        successful_pdf = sum(1 for r in pdf_results if not isinstance(r, Exception))
        
        return {
            "csv_processed": successful_csv,
            "pdf_processed": successful_pdf,
            "total_processed": successful_csv + successful_pdf,
            "csv_results": csv_results,
            "pdf_results": pdf_results,
            "errors": [str(r) for r in results if isinstance(r, Exception)]
        }
    
    async def _process_csv(self, file_path: Path) -> CSVAnalysisResult:
        """Process a single CSV file."""
        ctx = RunContextWrapper(self.context)
        return await analyze_csv_with_agent(ctx, str(file_path))
    
    async def _process_pdf(self, file_path: Path) -> PDFAnalysisResult:
        """Process a single PDF file."""
        ctx = RunContextWrapper(self.context)
        return await analyze_pdf_with_agent(ctx, str(file_path))