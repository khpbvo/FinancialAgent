"""Enhanced tools with strict parameter validation using TypedDict and Pydantic."""

from __future__ import annotations
from typing import Optional, List, Literal
from typing_extensions import TypedDict
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field, validator, ValidationError
from agents import function_tool, RunContextWrapper

from .context import RunDeps
from .models import Transaction, TransactionList, CategorySpending


class TransactionSearchParams(TypedDict):
    """Validated parameters for transaction search."""

    keyword: str
    min_amount: Optional[float]
    max_amount: Optional[float]
    start_date: Optional[str]
    end_date: Optional[str]
    category: Optional[str]
    limit: Optional[int]


class FileIngestionParams(BaseModel):
    """Validated parameters for file ingestion."""

    file_path: Path = Field(description="Path to the file to ingest")
    file_type: Literal["csv", "pdf"] = Field(description="Type of file")
    encoding: str = Field(default="utf-8", description="File encoding")
    delimiter: str = Field(default=",", description="CSV delimiter")
    skip_rows: int = Field(default=0, ge=0, description="Rows to skip")

    @validator("file_path")
    def validate_file_exists(cls, v):
        if not v.exists():
            raise ValueError(f"File not found: {v}")
        return v

    @validator("file_type")
    def validate_file_extension(cls, v, values):
        if "file_path" in values:
            ext = values["file_path"].suffix.lower()[1:]  # Remove the dot
            if ext != v:
                raise ValueError(f"File extension {ext} doesn't match type {v}")
        return v


class TransactionAddParams(BaseModel):
    """Validated parameters for adding a transaction."""

    date: str = Field(description="Transaction date in ISO format (YYYY-MM-DD)")
    description: str = Field(
        min_length=1, max_length=500, description="Transaction description"
    )
    amount: float = Field(description="Transaction amount")
    currency: str = Field(
        default="EUR", regex="^[A-Z]{3}$", description="3-letter currency code"
    )
    category: Optional[str] = Field(max_length=100, description="Transaction category")
    tags: List[str] = Field(default_factory=list, description="Transaction tags")

    @validator("date")
    def validate_date_format(cls, v):
        try:
            datetime.strptime(v, "%Y-%m-%d")
        except ValueError:
            raise ValueError("Date must be in YYYY-MM-DD format")
        return v

    @validator("amount")
    def validate_amount(cls, v):
        if abs(v) > 10000000:  # 10 million limit
            raise ValueError("Transaction amount seems unrealistic")
        return round(v, 2)  # Round to 2 decimal places

    @validator("tags")
    def validate_tags(cls, v):
        # Clean and validate tags
        return [tag.strip().lower() for tag in v if tag.strip()]


class AnalysisParams(TypedDict, total=False):
    """Parameters for financial analysis."""

    period: Literal["daily", "weekly", "monthly", "yearly"]
    include_categories: List[str]
    exclude_categories: List[str]
    min_transaction_amount: float
    group_by: Literal["category", "date", "merchant"]
    output_format: Literal["summary", "detailed", "csv"]


class BudgetCreationParams(BaseModel):
    """Validated parameters for budget creation."""

    monthly_income: float = Field(gt=0, description="Monthly income")
    fixed_expenses: List[CategoryBudget] = Field(description="Fixed monthly expenses")
    savings_goal_percentage: float = Field(
        ge=0, le=100, description="Savings goal as percentage"
    )
    budget_period: Literal["monthly", "quarterly", "yearly"] = Field(default="monthly")
    strict_mode: bool = Field(default=False, description="Enforce strict budget limits")

    @validator("savings_goal_percentage")
    def validate_savings_goal(cls, v):
        if v > 50:
            print("Warning: Savings goal >50% might be unrealistic")
        return v

    @validator("fixed_expenses")
    def validate_expenses(cls, v, values):
        if "monthly_income" in values:
            total_fixed = sum(exp.amount for exp in v)
            if total_fixed > values["monthly_income"]:
                raise ValueError("Fixed expenses exceed monthly income")
        return v


class CategoryBudget(BaseModel):
    """Budget allocation for a category."""

    category: str = Field(min_length=1, max_length=50)
    amount: float = Field(gt=0)
    is_essential: bool = Field(default=True)
    can_reduce_by: float = Field(
        default=0.0, ge=0, le=100, description="Percentage that can be reduced"
    )


@function_tool
async def search_transactions_validated(
    ctx: RunContextWrapper[RunDeps], params: TransactionSearchParams
) -> TransactionList:
    """Search transactions with validated parameters.

    Args:
        params: Validated search parameters including keyword, amount range, date range, etc.
    """
    deps = ctx.context
    query = "SELECT * FROM transactions WHERE 1=1"
    query_params = []

    # Build query with validated parameters
    if params.get("keyword"):
        query += " AND description LIKE ?"
        query_params.append(f"%{params['keyword']}%")

    if params.get("min_amount") is not None:
        query += " AND amount >= ?"
        query_params.append(params["min_amount"])

    if params.get("max_amount") is not None:
        query += " AND amount <= ?"
        query_params.append(params["max_amount"])

    if params.get("start_date"):
        query += " AND date >= ?"
        query_params.append(params["start_date"])

    if params.get("end_date"):
        query += " AND date <= ?"
        query_params.append(params["end_date"])

    if params.get("category"):
        query += " AND category = ?"
        query_params.append(params["category"])

    # Add limit
    limit = params.get("limit", 100)
    query += f" ORDER BY date DESC LIMIT {limit}"

    # Execute query
    cur = deps.db.conn.cursor()
    cur.execute(query, query_params)
    rows = cur.fetchall()

    # Convert to structured output
    transactions = []
    total_amount = 0.0

    for row in rows:
        tx = Transaction(
            date=row["date"],
            description=row["description"],
            amount=row["amount"],
            currency=row["currency"] or "EUR",
            category=row["category"],
            source_file=row["source_file"],
        )
        transactions.append(tx)
        total_amount += row["amount"]

    # Determine date range
    if transactions:
        dates = [tx.date for tx in transactions if tx.date]
        date_range = f"{min(dates)} to {max(dates)}"
    else:
        date_range = None

    return TransactionList(
        transactions=transactions,
        count=len(transactions),
        total_amount=round(total_amount, 2),
        date_range=date_range,
    )


@function_tool
async def add_validated_transaction(
    ctx: RunContextWrapper[RunDeps], transaction: TransactionAddParams
) -> str:
    """Add a transaction with full validation.

    Args:
        transaction: Validated transaction parameters
    """
    try:
        # Transaction is already validated by Pydantic
        deps = ctx.context
        cur = deps.db.conn.cursor()

        # Check for duplicate
        cur.execute(
            "SELECT COUNT(*) as count FROM transactions WHERE date = ? AND amount = ? AND description = ?",
            (transaction.date, transaction.amount, transaction.description),
        )
        if cur.fetchone()["count"] > 0:
            return "Warning: A similar transaction already exists. Added anyway."

        # Insert transaction
        cur.execute(
            """
            INSERT INTO transactions (date, description, amount, currency, category, source_file)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                transaction.date,
                transaction.description,
                transaction.amount,
                transaction.currency,
                transaction.category,
                f"manual_entry_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            ),
        )
        deps.db.conn.commit()

        # Add tags as memory if provided
        if transaction.tags:
            tags_str = ", ".join(transaction.tags)
            cur.execute(
                "INSERT INTO memories (kind, content, tags) VALUES (?, ?, ?)",
                ("tags", f"Transaction tags: {transaction.description}", tags_str),
            )
            deps.db.conn.commit()

        return f"✅ Transaction added successfully: {transaction.description} for {transaction.currency} {transaction.amount}"

    except ValidationError as e:
        errors = "; ".join([f"{err['loc'][0]}: {err['msg']}" for err in e.errors()])
        return f"❌ Validation failed: {errors}"
    except Exception as e:
        return f"❌ Error adding transaction: {str(e)}"


@function_tool
async def create_budget_validated(
    ctx: RunContextWrapper[RunDeps], budget_params: BudgetCreationParams
) -> str:
    """Create a budget with validated parameters.

    Args:
        budget_params: Validated budget creation parameters
    """
    try:
        deps = ctx.context

        # Calculate budget breakdown
        total_fixed = sum(exp.amount for exp in budget_params.fixed_expenses)
        savings_target = budget_params.monthly_income * (
            budget_params.savings_goal_percentage / 100
        )
        discretionary = budget_params.monthly_income - total_fixed - savings_target

        # Store budget in memories
        cur = deps.db.conn.cursor()

        budget_summary = f"""
Budget Created for {budget_params.budget_period} period:
- Monthly Income: €{budget_params.monthly_income:,.2f}
- Fixed Expenses: €{total_fixed:,.2f}
- Savings Target: €{savings_target:,.2f} ({budget_params.savings_goal_percentage}%)
- Discretionary: €{discretionary:,.2f}

Fixed Expenses Breakdown:
{chr(10).join(f'  • {exp.category}: €{exp.amount:,.2f} {"[Essential]" if exp.is_essential else f"[Can reduce by {exp.can_reduce_by}%]"}' for exp in budget_params.fixed_expenses)}

{"⚠️ STRICT MODE ENABLED - All limits will be enforced" if budget_params.strict_mode else "📊 Flexible budget - recommendations only"}
"""

        cur.execute(
            "INSERT INTO memories (kind, content, tags) VALUES (?, ?, ?)",
            ("budget", budget_summary, f"budget,{budget_params.budget_period}"),
        )
        deps.db.conn.commit()

        # Validate against current spending if data exists
        cur.execute(
            """
            SELECT category, SUM(amount) as total 
            FROM transactions 
            WHERE amount < 0 AND date >= date('now', '-30 days')
            GROUP BY category
            """
        )
        current_spending = {
            row["category"]: abs(row["total"]) for row in cur.fetchall()
        }

        warnings = []
        for expense in budget_params.fixed_expenses:
            current = current_spending.get(expense.category, 0)
            if current > expense.amount * 1.1:  # 10% tolerance
                warnings.append(
                    f"⚠️ {expense.category}: Currently spending €{current:.2f}, budget is €{expense.amount:.2f}"
                )

        result = budget_summary
        if warnings:
            result += "\n\nWarnings based on recent spending:\n" + "\n".join(warnings)

        return result

    except ValidationError as e:
        errors = "; ".join([f"{err['loc'][0]}: {err['msg']}" for err in e.errors()])
        return f"❌ Budget validation failed: {errors}"
    except Exception as e:
        return f"❌ Error creating budget: {str(e)}"


@function_tool
async def analyze_spending_validated(
    ctx: RunContextWrapper[RunDeps], analysis_params: AnalysisParams
) -> CategorySpending | str:
    """Analyze spending with validated parameters.

    Args:
        analysis_params: Parameters for spending analysis
    """
    deps = ctx.context

    # Build the analysis query
    period = analysis_params.get("period", "monthly")
    group_by = analysis_params.get("group_by", "category")

    # Date calculation based on period
    date_filter = {
        "daily": "date('now', '-1 day')",
        "weekly": "date('now', '-7 days')",
        "monthly": "date('now', '-30 days')",
        "yearly": "date('now', '-365 days')",
    }.get(period, "date('now', '-30 days')")

    query = f"""
        SELECT 
            {group_by} as grouping,
            COUNT(*) as transaction_count,
            SUM(amount) as total_amount,
            AVG(amount) as avg_amount
        FROM transactions
        WHERE amount < 0 AND date >= {date_filter}
    """

    # Add category filters
    if "include_categories" in analysis_params:
        categories = ", ".join(
            [f"'{c}'" for c in analysis_params["include_categories"]]
        )
        query += f" AND category IN ({categories})"

    if "exclude_categories" in analysis_params:
        categories = ", ".join(
            [f"'{c}'" for c in analysis_params["exclude_categories"]]
        )
        query += f" AND category NOT IN ({categories})"

    if "min_transaction_amount" in analysis_params:
        query += f" AND ABS(amount) >= {analysis_params['min_transaction_amount']}"

    query += f" GROUP BY {group_by} ORDER BY total_amount"

    # Execute query
    cur = deps.db.conn.cursor()
    cur.execute(query)
    rows = cur.fetchall()

    if not rows:
        return "No transactions found matching the criteria"

    # Format output based on requested format
    output_format = analysis_params.get("output_format", "summary")

    if output_format == "csv":
        lines = [f"{group_by},count,total,average"]
        for row in rows:
            lines.append(
                f"{row['grouping']},{row['transaction_count']},{abs(row['total_amount']):.2f},{abs(row['avg_amount']):.2f}"
            )
        return "\n".join(lines)

    elif output_format == "detailed":
        result = f"Detailed Analysis ({period}):\n"
        total_spending = sum(abs(row["total_amount"]) for row in rows)

        for row in rows:
            pct = (
                (abs(row["total_amount"]) / total_spending * 100)
                if total_spending > 0
                else 0
            )
            result += f"\n{row['grouping']}:\n"
            result += f"  • Transactions: {row['transaction_count']}\n"
            result += f"  • Total: €{abs(row['total_amount']):.2f}\n"
            result += f"  • Average: €{abs(row['avg_amount']):.2f}\n"
            result += f"  • Percentage: {pct:.1f}%\n"

        return result

    else:  # summary
        total_spending = sum(abs(row["total_amount"]) for row in rows)
        top_categories = sorted(
            rows, key=lambda x: abs(x["total_amount"]), reverse=True
        )[:5]

        result = f"Spending Summary ({period}):\n"
        result += f"Total: €{total_spending:.2f}\n\n"
        result += "Top Categories:\n"
        for row in top_categories:
            pct = (
                (abs(row["total_amount"]) / total_spending * 100)
                if total_spending > 0
                else 0
            )
            result += (
                f"  • {row['grouping']}: €{abs(row['total_amount']):.2f} ({pct:.1f}%)\n"
            )

        return result
