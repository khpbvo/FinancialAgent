#!/usr/bin/env python3
"""
Test Suite for Refactored Financial Agent
Tests the OpenAI Agents SDK implementation
"""

import asyncio
import os
import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

# Set up test environment
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY', 'test-key-placeholder')

try:
    from main_refactored import (
        create_main_orchestrator,
        create_document_processor_agent,
        create_financial_analyst_agent,
        create_advisor_agent,
        parse_financial_document,
        analyze_transactions,
        search_transactions,
        get_spending_summary,
        read_file_content
    )
    from agents import Runner
    print("✅ Successfully imported refactored modules")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure openai-agents is installed: pip install openai-agents")

def create_test_csv():
    """Create a test CSV file for document processing"""
    test_data = """date,description,amount,category
2024-01-01,Grocery Store,-125.50,Food & Dining
2024-01-02,Gas Station,-45.00,Transportation
2024-01-03,Netflix,-15.99,Entertainment
2024-01-05,Salary,3000.00,Income
2024-01-10,Rent,-1200.00,Housing"""
    
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
    temp_file.write(test_data)
    temp_file.close()
    
    return temp_file.name

async def test_function_tools():
    """Test individual function tools"""
    print("\n🔧 Testing Function Tools")
    print("-" * 30)
    
    # Test file reading
    test_file = create_test_csv()
    try:
        content = await read_file_content(test_file)
        print(f"✅ read_file_content: {len(content)} characters read")
    except Exception as e:
        print(f"❌ read_file_content error: {e}")
    finally:
        os.unlink(test_file)  # Clean up
    
    # Test transaction search (will fail without database, but we can test the function signature)
    try:
        result = await search_transactions("test query")
        print("✅ search_transactions: Function callable")
    except Exception as e:
        print(f"⚠️ search_transactions (expected DB error): {type(e).__name__}")
    
    # Test spending summary
    try:
        result = await get_spending_summary()
        print("✅ get_spending_summary: Function callable")
    except Exception as e:
        print(f"⚠️ get_spending_summary (expected DB error): {type(e).__name__}")

async def test_agent_creation():
    """Test agent creation"""
    print("\n🤖 Testing Agent Creation")
    print("-" * 30)
    
    try:
        doc_processor = create_document_processor_agent()
        print(f"✅ Document Processor: {doc_processor.name}")
        
        analyst = create_financial_analyst_agent()
        print(f"✅ Financial Analyst: {analyst.name}")
        
        advisor = create_advisor_agent()
        print(f"✅ Financial Advisor: {advisor.name}")
        
        orchestrator = create_main_orchestrator()
        print(f"✅ Main Orchestrator: {orchestrator.name}")
        
    except Exception as e:
        print(f"❌ Agent creation error: {e}")

async def test_basic_agent_interaction():
    """Test basic agent interaction (requires valid OpenAI API key)"""
    print("\n💬 Testing Basic Agent Interaction")
    print("-" * 30)
    
    if not os.getenv('OPENAI_API_KEY') or os.getenv('OPENAI_API_KEY') == 'test-key-placeholder':
        print("⚠️ Skipping agent interaction test - no valid OpenAI API key")
        return
    
    try:
        # Create a simple advisor agent for testing
        advisor = create_advisor_agent()
        
        # Test a basic financial question
        result = await Runner.run(
            advisor, 
            "What is a good savings rate percentage for someone starting their career?",
            max_turns=1
        )
        
        print(f"✅ Agent Response: {result.final_output[:100]}...")
        
    except Exception as e:
        print(f"❌ Agent interaction error: {e}")

async def test_orchestrator_handoffs():
    """Test orchestrator with handoffs (requires valid API key)"""
    print("\n🎯 Testing Orchestrator Handoffs")
    print("-" * 30)
    
    if not os.getenv('OPENAI_API_KEY') or os.getenv('OPENAI_API_KEY') == 'test-key-placeholder':
        print("⚠️ Skipping orchestrator test - no valid OpenAI API key")
        return
    
    try:
        orchestrator = create_main_orchestrator()
        
        # Test a complex request that should trigger handoffs
        result = await Runner.run(
            orchestrator,
            "I need help analyzing my spending patterns and getting recommendations for better budgeting.",
            max_turns=2
        )
        
        print(f"✅ Orchestrator Response: {result.final_output[:100]}...")
        
    except Exception as e:
        print(f"❌ Orchestrator error: {e}")

def test_configuration():
    """Test configuration and imports"""
    print("\n⚙️ Testing Configuration")
    print("-" * 30)
    
    # Test environment variables
    api_key = os.getenv('OPENAI_API_KEY')
    print(f"✅ OpenAI API Key: {'Set' if api_key else 'Not set'}")
    
    # Test imports
    try:
        from agents import Agent, Runner
        print("✅ Core agents imports successful")
    except ImportError:
        print("❌ Core agents imports failed")
    
    try:
        from agents.tool import WebSearchTool, FileSearchTool
        print("✅ Agent tools imports successful")
    except ImportError:
        print("❌ Agent tools imports failed")
    
    try:
        from agents.models import OpenAIResponsesModel
        print("✅ OpenAI models imports successful")
    except ImportError:
        print("❌ OpenAI models imports failed")

async def run_all_tests():
    """Run all test suites"""
    print("🧪 Financial Agent Refactoring Test Suite")
    print("=" * 50)
    
    # Configuration tests (sync)
    test_configuration()
    
    # Async tests
    await test_function_tools()
    await test_agent_creation()
    await test_basic_agent_interaction()
    await test_orchestrator_handoffs()
    
    print("\n" + "=" * 50)
    print("🏁 Test Suite Complete")
    
    # Summary
    print("\n📋 Summary:")
    print("- Function tools: Tested basic functionality")
    print("- Agent creation: Verified agent instantiation")
    print("- API interaction: Depends on valid OpenAI key")
    print("- Database operations: Require PostgreSQL setup")
    
    print("\n💡 Next Steps:")
    print("1. Set up PostgreSQL with pgvector extension")
    print("2. Configure DATABASE_URL environment variable")
    print("3. Run: pip install -r requirements_agents_sdk.txt")
    print("4. Test with: python main_refactored.py status")

if __name__ == '__main__':
    asyncio.run(run_all_tests())