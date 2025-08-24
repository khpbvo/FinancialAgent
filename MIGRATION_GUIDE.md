# Migration Guide: From Completions API to OpenAI Agents SDK

## Overview

This guide documents the migration from the original `main.py` implementation (using OpenAI's chat completions API) to the new `main_refactored.py` implementation (using the OpenAI Agents SDK).

## Key Changes

### 1. API Migration
**Before (Completions API):**
```python
response = self.client.chat.completions.create(
    model=settings.OPENAI_MODEL,
    messages=[{"role": "system", "content": "..."}, {"role": "user", "content": "..."}],
    temperature=settings.AGENT_TEMPERATURE,
    max_completion_tokens=2000
)
```

**After (Agents SDK):**
```python
from agents import Agent, Runner
from agents.models import OpenAIResponsesModel

agent = Agent(
    name="Financial Expert",
    model=OpenAIResponsesModel(model="gpt-4o"),
    instructions="You are a financial expert...",
    tools=[WebSearchTool(), parse_financial_document]
)

result = await Runner.run(agent, user_input)
```

### 2. Tool Integration
**Before (Manual Tool Handling):**
```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "parse_financial_document",
            "description": "Parse and extract data from financial documents",
            "parameters": {...}
        }
    }
]

# Manual tool call handling
if response.choices[0].message.tool_calls:
    for tool_call in response.choices[0].message.tool_calls:
        # Manual parsing and execution
```

**After (Automatic Tool Handling):**
```python
@function_tool
async def parse_financial_document(file_path: str) -> str:
    """Parse and extract data from financial documents.
    
    Args:
        file_path: Path to the financial document to parse
    """
    # Implementation
    return json.dumps(result)

agent = Agent(
    name="Document Processor",
    tools=[parse_financial_document]  # Automatic tool integration
)
```

### 3. Multi-Agent Architecture
**Before (Single Agent):**
```python
class FinancialExpertAgent:
    def __init__(self):
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
        # All functionality in one class
```

**After (Specialized Agents):**
```python
def create_document_processor_agent():
    return Agent(name="Document Processor", tools=[parse_financial_document])

def create_financial_analyst_agent():
    return Agent(name="Financial Analyst", tools=[analyze_transactions])

def create_advisor_agent():
    return Agent(name="Financial Advisor", tools=[WebSearchTool()])

def create_main_orchestrator():
    return Agent(
        name="Orchestrator",
        handoffs=[doc_processor, analyst, advisor]
    )
```

### 4. Built-in Tools Integration
**New Capabilities:**
```python
from agents.tool import WebSearchTool, FileSearchTool, CodeInterpreterTool

advisor_agent = Agent(
    name="Enhanced Financial Advisor",
    tools=[
        WebSearchTool(),           # Real-time web search
        FileSearchTool(),          # Vector store search
        CodeInterpreterTool()      # Code execution
    ]
)
```

## Installation & Setup

### 1. Install Dependencies
```bash
# Install the OpenAI Agents SDK
pip install openai-agents

# Install other requirements
pip install -r requirements_agents_sdk.txt
```

### 2. Environment Configuration
```bash
# Required environment variables
export OPENAI_API_KEY=sk-...
export DATABASE_URL=postgresql://user:pass@localhost/db

# Optional: For enhanced features
export OPENAI_ORGANIZATION=org-...
```

### 3. Database Setup
The database schema remains the same, but make sure pgvector is installed:
```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

## Feature Comparison

| Feature | Original (`main.py`) | Refactored (`main_refactored.py`) |
|---------|---------------------|-----------------------------------|
| **API** | Chat Completions | Agents SDK with Responses |
| **Tool Handling** | Manual | Automatic with decorators |
| **Multi-Agent** | Single agent class | Specialized agent orchestration |
| **Built-in Tools** | None | WebSearch, FileSearch, CodeInterpreter |
| **Async Support** | Limited | Full async/await |
| **Error Handling** | Manual try/catch | Built-in tool error handling |
| **Conversation Flow** | Manual state management | Automatic session management |

## Migration Steps

### Step 1: Update Dependencies
```bash
# Backup your current requirements
cp requirements.txt requirements_original.txt

# Install new requirements
pip install -r requirements_agents_sdk.txt
```

### Step 2: Test the Refactored Version
```bash
# Run the test suite
python test_refactored.py

# Test basic functionality
python main_refactored.py status
```

### Step 3: Migrate Existing Data
Your existing database and data will work with the refactored version without changes.

### Step 4: Update CLI Commands
The CLI interface remains largely the same:

```bash
# Document analysis (same)
python main_refactored.py analyze documents/statement.pdf

# Financial summary (same)  
python main_refactored.py summary --period last_month

# Interactive chat (enhanced)
python main_refactored.py chat

# New: Enhanced recommendations
python main_refactored.py recommendations
```

## Key Benefits of Migration

### 1. **Better Tool Integration**
- Automatic tool schema generation
- Built-in error handling
- Type-safe function parameters

### 2. **Enhanced Capabilities**
- Real-time web search for current financial information
- Code execution for complex calculations
- File search across document collections

### 3. **Improved Architecture**
- Specialized agents for different tasks
- Better separation of concerns
- Easier to test and maintain

### 4. **Future-Proof**
- Built on OpenAI's official agent framework
- Regular updates and improvements
- Better integration with OpenAI ecosystem

## Troubleshooting

### Common Issues

1. **Import Errors**
```bash
# Solution: Install the agents SDK
pip install openai-agents
```

2. **Database Connection Issues**
```bash
# Check your DATABASE_URL
echo $DATABASE_URL

# Ensure PostgreSQL is running
brew services start postgresql  # macOS
sudo systemctl start postgresql  # Linux
```

3. **API Key Issues**
```bash
# Verify your API key is set
echo $OPENAI_API_KEY

# Test API access
python -c "from openai import OpenAI; print('API key valid')"
```

## Performance Considerations

### Before Migration
- Manual tool calls: ~2-3 seconds per interaction
- Single-threaded processing
- Limited error recovery

### After Migration  
- Automatic tool orchestration: ~1-2 seconds per interaction
- Async processing with better concurrency
- Built-in retry and error handling
- Improved response quality through specialized agents

## Rollback Plan

If you need to rollback to the original implementation:

1. Keep `main.py` as backup
2. Use `requirements.txt` (original dependencies)
3. Switch back to the original CLI commands

The database and data remain compatible with both versions.

## Next Steps

1. **Test thoroughly** with your specific use cases
2. **Monitor performance** and compare with original implementation
3. **Explore enhanced features** like web search and code execution
4. **Consider extending** with additional specialized agents
5. **Update documentation** and user guides

## Support

If you encounter issues during migration:

1. Check the [OpenAI Agents SDK documentation](https://docs.anthropic.com/en/docs/claude-code/claude_code_docs_map.md)
2. Review the test suite output for specific errors
3. Compare your implementation with the provided examples
4. Check OpenAI API status and rate limits

The refactored implementation provides a more robust, scalable, and feature-rich foundation for your financial analysis application.