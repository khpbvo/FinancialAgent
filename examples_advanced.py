"""Examples of using the advanced SDK features."""

import asyncio
from pathlib import Path
from datetime import datetime
import os

from agents import Runner, RunContextWrapper
from financial_agent.agent import build_deps
from financial_agent.advanced_agent import build_advanced_agent
from financial_agent.document_agents import (
    build_document_orchestrator,
    BatchDocumentProcessor,
    analyze_csv_with_agent,
    analyze_pdf_with_agent,
)
from financial_agent.validated_tools import (
    TransactionSearchParams,
    TransactionAddParams,
    BudgetCreationParams,
    CategoryBudget,
    AnalysisParams,
    search_transactions_validated,
    add_validated_transaction,
    create_budget_validated,
    analyze_spending_validated,
)
from financial_agent.session_protocol import (
    FinancialSessionManager,
    StorageBackend,
)


async def example_advanced_agent():
    """Example: Using the advanced agent with all optimizations."""
    print("=" * 60)
    print("Example: Advanced Agent with SDK Optimizations")
    print("=" * 60)

    # Build agent with all features enabled
    agent = build_advanced_agent(
        enable_web_search=True,
        enable_guardrails=True,
        enable_hooks=True,
        tool_choice_mode="critical",  # Forces data ingestion when DB is empty
    )

    with build_deps() as deps:
        # Run with PII protection (guardrails will catch this)
        try:
            result = await Runner.run(
                agent,
                input="My SSN is 123-45-6789, can you analyze my spending?",
                context=deps,
            )
            print(f"Result: {result.final_output}")
        except Exception as e:
            print(f"Guardrail caught PII: {e}")

        # Run with web search for real-time data
        result = await Runner.run(
            agent,
            input="What's the current EUR to USD exchange rate and how does it affect my budget?",
            context=deps,
        )
        print(f"\nWeb Search Result: {result.final_output}")


async def example_document_agents():
    """Example: Using document analysis agents as tools."""
    print("\n" + "=" * 60)
    print("Example: Document Analysis Agents")
    print("=" * 60)

    with build_deps() as deps:
        ctx = RunContextWrapper(deps)

        # Analyze a CSV file with the specialized agent
        csv_path = deps.config.documents_dir / "sample_transactions.csv"
        if csv_path.exists():
            csv_result = await analyze_csv_with_agent(ctx, str(csv_path))
            print("\nCSV Analysis:")
            print(f"  File: {csv_result.file_name}")
            print(f"  Rows: {csv_result.row_count}")
            print(f"  Categories: {', '.join(csv_result.categories_found)}")
            print(f"  Errors: {len(csv_result.parsing_errors)}")

        # Analyze a PDF with the specialized agent
        pdf_path = deps.config.documents_dir / "bank_statement.pdf"
        if pdf_path.exists():
            pdf_result = await analyze_pdf_with_agent(ctx, str(pdf_path))
            print("\nPDF Analysis:")
            print(f"  File: {pdf_result.file_name}")
            print(f"  Pages: {pdf_result.page_count}")
            print(f"  Transactions: {pdf_result.transactions_found}")
            print(f"  Confidence: {pdf_result.extraction_confidence:.1%}")

        # Use the orchestrator for complex document analysis
        orchestrator = build_document_orchestrator()
        result = await Runner.run(
            orchestrator,
            input="Analyze all CSV and PDF files in the documents folder and give me a summary",
            context=deps,
        )
        print(f"\nOrchestrator Result: {result.final_output}")


async def example_batch_processing():
    """Example: Batch processing multiple documents in parallel."""
    print("\n" + "=" * 60)
    print("Example: Batch Document Processing")
    print("=" * 60)

    with build_deps() as deps:
        processor = BatchDocumentProcessor(deps)

        # Find all CSV and PDF files
        csv_files = list(deps.config.documents_dir.glob("*.csv"))
        pdf_files = list(deps.config.documents_dir.glob("*.pdf"))

        if csv_files or pdf_files:
            print(f"Processing {len(csv_files)} CSV and {len(pdf_files)} PDF files...")

            results = await processor.process_batch(csv_files, pdf_files)

            print("\nBatch Processing Results:")
            print(f"  CSV processed: {results['csv_processed']}")
            print(f"  PDF processed: {results['pdf_processed']}")
            print(f"  Total: {results['total_processed']}")

            if results["errors"]:
                print(f"  Errors: {len(results['errors'])}")
                for error in results["errors"][:3]:  # Show first 3 errors
                    print(f"    - {error}")


async def example_validated_tools():
    """Example: Using tools with strict parameter validation."""
    print("\n" + "=" * 60)
    print("Example: Validated Tools")
    print("=" * 60)

    with build_deps() as deps:
        ctx = RunContextWrapper(deps)

        # Search with validated parameters
        search_params: TransactionSearchParams = {
            "keyword": "grocery",
            "min_amount": 10.0,
            "max_amount": 100.0,
            "start_date": "2024-01-01",
            "end_date": "2024-12-31",
            "limit": 10,
        }

        results = await search_transactions_validated(ctx, search_params)
        print(
            f"\nSearch Results: {results.count} transactions, total: €{results.total_amount:.2f}"
        )

        # Add transaction with validation
        new_transaction = TransactionAddParams(
            date="2024-12-25",
            description="Christmas shopping",
            amount=-250.00,
            currency="EUR",
            category="Shopping",
            tags=["holiday", "gifts", "christmas"],
        )

        result = await add_validated_transaction(ctx, new_transaction)
        print(f"\nAdd Transaction: {result}")

        # Create budget with validation
        budget = BudgetCreationParams(
            monthly_income=5000.0,
            fixed_expenses=[
                CategoryBudget(category="Rent", amount=1500.0, is_essential=True),
                CategoryBudget(category="Utilities", amount=200.0, is_essential=True),
                CategoryBudget(
                    category="Food", amount=600.0, is_essential=True, can_reduce_by=10.0
                ),
                CategoryBudget(
                    category="Transport",
                    amount=300.0,
                    is_essential=False,
                    can_reduce_by=25.0,
                ),
            ],
            savings_goal_percentage=20.0,
            budget_period="monthly",
            strict_mode=True,
        )

        result = await create_budget_validated(ctx, budget)
        print(f"\nBudget Creation:\n{result}")

        # Analyze spending with parameters
        analysis_params: AnalysisParams = {
            "period": "monthly",
            "group_by": "category",
            "output_format": "summary",
            "min_transaction_amount": 5.0,
        }

        result = await analyze_spending_validated(ctx, analysis_params)
        print(f"\nSpending Analysis:\n{result}")


async def example_session_management():
    """Example: Using session management with different backends."""
    print("\n" + "=" * 60)
    print("Example: Session Management")
    print("=" * 60)

    # SQLite backend (default)
    sqlite_manager = FinancialSessionManager(
        backend=StorageBackend.SQLITE, db_path="sessions.db"
    )

    # Save an interaction
    await sqlite_manager.save_interaction(
        session_id="user_123",
        user_input="Show me my spending for last month",
        agent_response="Your total spending last month was €2,345.67...",
        metadata={"timestamp": datetime.now().isoformat()},
    )

    # Get context from previous messages
    context = await sqlite_manager.get_context("user_123", max_messages=5)
    print(f"\nSession Context:\n{context}")

    # JSON backend
    json_manager = FinancialSessionManager(
        backend=StorageBackend.JSON, storage_dir="session_data"
    )

    await json_manager.save_interaction(
        session_id="user_456",
        user_input="Create a budget for me",
        agent_response="Based on your income and expenses, here's your budget...",
        metadata={"category": "budget_creation"},
    )

    # Export session in different formats
    json_export = await sqlite_manager.export_session("user_123", format="json")
    print(f"\nJSON Export (first 200 chars):\n{json_export[:200]}...")

    text_export = await json_manager.export_session("user_456", format="text")
    print(f"\nText Export:\n{text_export}")

    # In-memory backend (for testing)
    memory_manager = FinancialSessionManager(backend=StorageBackend.MEMORY)

    await memory_manager.save_interaction(
        session_id="test_session",
        user_input="Test input",
        agent_response="Test response",
        metadata={"test": True},
    )

    # Custom backend registration example
    print("\nCustom backends can be registered using SessionFactory.register_backend()")


async def example_combined_workflow():
    """Example: Complete workflow using all advanced features."""
    print("\n" + "=" * 60)
    print("Example: Combined Advanced Workflow")
    print("=" * 60)

    # Initialize components
    with build_deps() as deps:
        session_manager = FinancialSessionManager(backend=StorageBackend.SQLITE)
        agent = build_advanced_agent(
            enable_web_search=True, enable_guardrails=True, enable_hooks=True
        )

        session_id = f"workflow_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Step 1: Analyze documents
        print("\n1. Analyzing documents...")
        processor = BatchDocumentProcessor(deps)
        csv_files = list(deps.config.documents_dir.glob("*.csv"))[
            :2
        ]  # Limit to 2 files

        if csv_files:
            doc_results = await processor.process_batch(csv_files, [])
            print(f"   Processed {doc_results['total_processed']} documents")

        # Step 2: Create validated budget
        print("\n2. Creating budget...")
        ctx = RunContextWrapper(deps)
        budget = BudgetCreationParams(
            monthly_income=4000.0,
            fixed_expenses=[
                CategoryBudget(category="Housing", amount=1200.0),
                CategoryBudget(category="Food", amount=500.0),
            ],
            savings_goal_percentage=15.0,
        )
        budget_result = await create_budget_validated(ctx, budget)

        # Step 3: Run agent with context
        print("\n3. Getting financial advice...")
        context = await session_manager.get_context(session_id, max_messages=5)

        result = await Runner.run(
            agent,
            input="Based on my budget and recent transactions, what should I focus on?",
            context=deps,
        )

        # Save interaction
        await session_manager.save_interaction(
            session_id=session_id,
            user_input="Based on my budget and recent transactions, what should I focus on?",
            agent_response=str(result.final_output),
            metadata={"workflow": "combined", "step": 3},
        )

        print(f"\nAdvice: {result.final_output}")

        # Step 4: Export session
        print("\n4. Exporting session...")
        export = await session_manager.export_session(session_id, format="json")
        export_path = Path(f"session_export_{session_id}.json")
        export_path.write_text(export)
        print(f"   Session exported to {export_path}")


async def main():
    """Run all examples."""
    print("🚀 Advanced Financial Agent SDK Examples")
    print("=" * 60)

    # Run examples based on what's available
    try:
        await example_advanced_agent()
    except Exception as e:
        print(f"Advanced agent example error: {e}")

    try:
        await example_document_agents()
    except Exception as e:
        print(f"Document agents example error: {e}")

    try:
        await example_batch_processing()
    except Exception as e:
        print(f"Batch processing example error: {e}")

    try:
        await example_validated_tools()
    except Exception as e:
        print(f"Validated tools example error: {e}")

    try:
        await example_session_management()
    except Exception as e:
        print(f"Session management example error: {e}")

    try:
        await example_combined_workflow()
    except Exception as e:
        print(f"Combined workflow example error: {e}")

    print("\n" + "=" * 60)
    print("✅ All examples completed!")


if __name__ == "__main__":
    # Ensure API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("Please set OPENAI_API_KEY environment variable")
        exit(1)

    asyncio.run(main())
