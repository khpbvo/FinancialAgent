from __future__ import annotations
from typing import List, Dict
from datetime import datetime, timedelta
import re
from collections import defaultdict
from agents import RunContextWrapper, function_tool
from ..context import RunDeps


def calculate_frequency(dates: List[str]) -> tuple[str, float]:
    """Infer recurring frequency from transaction dates.

    Strategy (monthly-first):
    - Collapse multiple occurrences per calendar month to a single representative date.
    - Detect monthly/quarterly/yearly using month gaps and day-of-month consistency.
    - Only if month-based detection fails, fall back to bi-weekly/weekly/daily checks.

    Returns: (frequency_type, confidence_score)
    """
    if len(dates) < 2:
        return "unknown", 0.0

    # Parse and sort
    dt_dates = sorted(datetime.strptime(d, "%Y-%m-%d") for d in dates)

    # Group by unique months and compute a representative day (median day-of-month)
    by_month: dict[str, list[datetime]] = defaultdict(list)
    for dt in dt_dates:
        by_month[f"{dt.year:04d}-{dt.month:02d}"].append(dt)

    unique_months = sorted(by_month.keys())

    # Build representative monthly timeline
    monthly_points: list[datetime] = []
    monthly_days: list[int] = []
    for m in unique_months:
        bucket = by_month[m]
        days = sorted(x.day for x in bucket)
        # Representative = median day-of-month to reduce noise
        median_day = days[len(days) // 2]
        y, mo = map(int, m.split("-"))
        monthly_points.append(
            datetime(y, mo, min(median_day, 28))
        )  # clamp to 28 for safety
        monthly_days.append(median_day)

    # Early exit if we don't have enough distinct months
    if len(monthly_points) >= 3:
        # Compute month gaps (in months) between consecutive points
        def months_between(a: datetime, b: datetime) -> int:
            return (b.year - a.year) * 12 + (b.month - a.month)

        gaps = [
            months_between(monthly_points[i], monthly_points[i + 1])
            for i in range(len(monthly_points) - 1)
        ]

        # Helper to build confidence: share of target gaps + day-of-month stability
        def monthly_conf(target_gap: int) -> float:
            if not gaps:
                return 0.0
            target_share = sum(1 for g in gaps if g == target_gap) / len(gaps)
            # Day-of-month stability: penalize if median absolute deviation is large
            if len(monthly_days) > 1:
                med = sorted(monthly_days)[len(monthly_days) // 2]
                mad = sum(abs(d - med) for d in monthly_days) / len(monthly_days)
                day_stability = max(0.0, 1.0 - (mad / 10.0))  # ~10-day tolerance
            else:
                day_stability = 0.8
            base = 0.6 if target_gap == 1 else 0.5  # monthly more likely than others
            return min(
                0.95, max(base, base * 0.5 + target_share * 0.4 + day_stability * 0.3)
            )

        # Prefer exact monthly (gap=1), then quarterly (3), then yearly (12)
        if all(g in (1, 2) for g in gaps) and gaps.count(1) >= max(
            2, int(0.6 * len(gaps))
        ):
            return "monthly", monthly_conf(1)
        if all(g in (3, 6) for g in gaps) and gaps.count(3) >= max(
            1, int(0.5 * len(gaps))
        ):
            return "quarterly", monthly_conf(3)
        if all(g >= 11 for g in gaps):
            return "yearly", monthly_conf(12)

    # Fall back to daily/weekly/bi-weekly checks on actual dates
    if len(dt_dates) >= 4:
        intervals = [
            (dt_dates[i + 1] - dt_dates[i]).days for i in range(len(dt_dates) - 1)
        ]
        if intervals:
            avg_day_interval = sum(intervals) / len(intervals)

            # Bi-weekly pattern (~14 days)
            if 12 <= avg_day_interval <= 16:
                variance = sum((i - 14) ** 2 for i in intervals) / len(intervals)
                confidence = max(0.2, 1 - (variance / 196))
                return "bi-weekly", min(confidence, 0.9)

            # Weekly (~7 days)
            if 5 <= avg_day_interval <= 10:
                variance = sum((i - avg_day_interval) ** 2 for i in intervals) / len(
                    intervals
                )
                confidence = max(0.1, 1 - (variance / 49))
                return "weekly", min(confidence, 0.9)

            # Daily (1–3 days)
            if len(dt_dates) >= 10 and 1 <= avg_day_interval <= 3:
                variance = sum((i - avg_day_interval) ** 2 for i in intervals) / len(
                    intervals
                )
                confidence = max(0.1, 1 - (variance / (avg_day_interval**2 or 1)))
                return "daily", min(confidence, 0.9)

    # Consider irregular recurring if seen across >= 3 months
    if len(set(d[:7] for d in dates)) >= 3:
        return "irregular", 0.4

    return "unknown", 0.0


def normalize_description(desc: str) -> str:
    """Normalize transaction description for pattern matching.
    Optimized for Dutch bank transaction descriptions.
    """
    if not desc:
        return "unknown"

    # Convert to uppercase for consistency (Dutch bank format)
    normalized = desc.upper().strip()

    # Remove common Dutch bank formatting patterns
    # Remove dates in various formats
    normalized = re.sub(r"\b\d{2}-\d{2}-\d{4}\b", "", normalized)  # DD-MM-YYYY
    normalized = re.sub(r"\b\d{4}-\d{2}-\d{2}\b", "", normalized)  # YYYY-MM-DD
    normalized = re.sub(r"\b\d{2}/\d{2}/\d{4}\b", "", normalized)  # DD/MM/YYYY

    # Remove policy numbers, account numbers, and reference numbers
    normalized = re.sub(r"\bPOLISNR[.:]*\s*\d+", "", normalized)
    normalized = re.sub(r"\bIBAN[.:]*\s*[A-Z]{2}\d{2}[A-Z0-9]+", "", normalized)
    normalized = re.sub(r"\bKENMERK[.:]*\s*\d+", "", normalized)
    normalized = re.sub(r"\bMACHTIGING ID[.:]*\s*\d+", "", normalized)
    normalized = re.sub(r"\bINCASS\w+ ID[.:]*\s*[A-Z0-9]+", "", normalized)

    # Remove time stamps and transaction IDs
    normalized = re.sub(r"\b\d{2}:\d{2}(:\d{2})?\b", "", normalized)
    normalized = re.sub(r"\b\d{6,}\b", "", normalized)  # Long numbers (transaction IDs)

    # Remove common Dutch banking terms that don't help identify merchants
    banking_terms = [
        "NAAM:",
        "OMSCHRIJVING:",
        "PERIODE:",
        "IBAN:",
        "KENMERK:",
        "VALUTADATUM:",
        "DOORLOPENDE INCASSO",
        "INCASSO",
        "DATUM/TIJD:",
        "PASVOLGNR:",
        "TRANSACTIE:",
        "TERM:",
        "APPLE PAY",
        "NLD",
        "BV",
        "NV",
        "VOF",
        "CV",
    ]

    for term in banking_terms:
        normalized = normalized.replace(term, " ")

    # Clean up specific patterns from the sample data
    normalized = re.sub(r"\s+BETR\s+", " ", normalized)  # "betr" = regarding
    normalized = re.sub(r"\s+VIA\s+", " ", normalized)  # "via" connector

    # Remove standalone single letters and numbers
    normalized = re.sub(r"\b[A-Z0-9]\b", " ", normalized)

    # Keep only alphanumeric characters and spaces, remove special chars
    normalized = re.sub(r"[^A-Z0-9\s]", " ", normalized)

    # Collapse multiple spaces and trim
    normalized = " ".join(normalized.split())

    # Take first significant part (up to 40 chars) to focus on merchant name
    if len(normalized) > 40:
        # Try to break at word boundary
        words = normalized.split()
        result = []
        char_count = 0
        for word in words:
            if char_count + len(word) > 35:  # Leave room for spaces
                break
            result.append(word)
            char_count += len(word) + 1  # +1 for space
        normalized = " ".join(result) if result else normalized[:40]

    return normalized.lower() if normalized else "unknown"


@function_tool
def detect_recurring(
    ctx: RunContextWrapper[RunDeps], min_occurrences: int = 3, lookback_months: int = 6
) -> str:
    """Detect recurring transactions like subscriptions and regular payments.

    Args:
        min_occurrences: Minimum times a transaction must occur to be considered recurring
        lookback_months: How many months back to analyze
    """
    deps = ctx.context
    cur = deps.db.conn.cursor()

    # Get transactions from the lookback period
    start_date = (datetime.now() - timedelta(days=lookback_months * 30)).strftime(
        "%Y-%m-%d"
    )

    cur.execute(
        """SELECT date, description, amount, category 
           FROM transactions 
           WHERE date >= ? 
           AND amount < 0
           ORDER BY description, date""",
        (start_date,),
    )

    transactions = cur.fetchall()

    if not transactions:
        return "No transactions found in the specified period."

    # Group transactions by normalized description with amount tolerance
    patterns = defaultdict(list)

    for tx in transactions:
        normalized = normalize_description(tx["description"])
        if not normalized or normalized == "unknown":
            continue

        # Use normalized description as primary key, handle amounts separately
        patterns[normalized].append(tx)

    # Find recurring patterns with amount clustering
    recurring = []

    for desc_pattern, txs in patterns.items():
        if len(txs) < min_occurrences:
            continue

        # Group by similar amounts (more flexible tolerance for utilities)
        amount_groups = defaultdict(list)
        for tx in txs:
            amount = abs(tx["amount"])
            # Find existing group with more flexible tolerance for bills
            group_key = None
            for existing_amount in amount_groups.keys():
                # Use 20% tolerance or €10 minimum (more lenient for variable bills)
                tolerance = max(10.0, existing_amount * 0.2)
                if abs(amount - existing_amount) <= tolerance:
                    group_key = existing_amount
                    break

            if group_key is None:
                group_key = amount

            amount_groups[group_key].append(tx)

        # Process each amount group separately
        for amount_key, group_txs in amount_groups.items():
            if len(group_txs) >= min_occurrences:
                # Calculate frequency and confidence
                dates = [tx["date"] for tx in group_txs]
                frequency, confidence = calculate_frequency(dates)

                # Even lower confidence threshold for better detection
                if confidence > 0.25:  # Very lenient threshold to catch more patterns
                    avg_amount = sum(abs(tx["amount"]) for tx in group_txs) / len(
                        group_txs
                    )

                    # Boost confidence for clearly recurring patterns
                    if frequency in ["monthly", "quarterly"] and len(group_txs) >= 4:
                        confidence = min(confidence + 0.2, 0.95)
                    elif frequency == "yearly" and len(group_txs) >= 2:
                        confidence = min(confidence + 0.1, 0.9)

                    # Check for amount consistency to boost confidence
                    amounts = [abs(tx["amount"]) for tx in group_txs]
                    if len(amounts) > 1:
                        amount_variance = sum(
                            (a - avg_amount) ** 2 for a in amounts
                        ) / len(amounts)
                        amount_consistency = (
                            1 - (amount_variance / (avg_amount**2))
                            if avg_amount > 0
                            else 0
                        )
                        confidence = min(confidence + (amount_consistency * 0.1), 0.95)

                    recurring.append(
                        {
                            "pattern": desc_pattern,
                            "sample_desc": group_txs[0]["description"],
                            "amount": avg_amount,
                            "frequency": frequency,
                            "confidence": confidence,
                            "occurrences": len(group_txs),
                            "category": group_txs[0]["category"],
                            "last_date": max(dates),
                        }
                    )

    if not recurring:
        return "No recurring transactions detected. Try reducing min_occurrences or increasing lookback period."

    # Sort by confidence and amount
    recurring.sort(key=lambda x: (x["confidence"], x["amount"]), reverse=True)

    # Store in database
    for rec in recurring:
        # Check if already exists
        cur.execute(
            "SELECT id FROM recurring_transactions WHERE description_pattern = ?",
            (rec["pattern"],),
        )

        if not cur.fetchone():
            cur.execute(
                """INSERT INTO recurring_transactions 
                   (description_pattern, amount, frequency, category, last_seen, confidence)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    rec["pattern"],
                    rec["amount"],
                    rec["frequency"],
                    rec["category"],
                    rec["last_date"],
                    rec["confidence"],
                ),
            )

    deps.db.conn.commit()

    # Format results
    results = [f"🔄 Detected {len(recurring)} Recurring Transactions\n" + "=" * 50]

    total_monthly = 0

    for rec in recurring:
        conf_percent = rec["confidence"] * 100

        # Calculate monthly cost with improved accuracy
        if rec["frequency"] == "daily":
            monthly_cost = rec["amount"] * 30.44  # Average days per month
        elif rec["frequency"] == "weekly":
            monthly_cost = rec["amount"] * 4.345  # 52.14 weeks per year / 12 months
        elif rec["frequency"] == "bi-weekly":
            monthly_cost = rec["amount"] * 2.173  # 26.07 payments per year / 12 months
        elif rec["frequency"] == "monthly":
            monthly_cost = rec["amount"]
        elif rec["frequency"] == "quarterly":
            monthly_cost = rec["amount"] / 3
        elif rec["frequency"] == "yearly":
            monthly_cost = rec["amount"] / 12
        elif rec["frequency"] == "irregular":
            # For irregular patterns, estimate based on occurrences over time period
            monthly_cost = rec["amount"]  # Conservative estimate
        else:
            monthly_cost = rec["amount"]

        total_monthly += monthly_cost

        # Determine icon based on confidence
        icon = "🟢" if conf_percent > 80 else "🟡" if conf_percent > 65 else "🔴"

        results.append(f"\n{icon} {rec['sample_desc'][:40]}")
        results.append(f"   Amount: €{rec['amount']:.2f} {rec['frequency']}")
        results.append(f"   Monthly cost: €{monthly_cost:.2f}")
        results.append(f"   Confidence: {conf_percent:.0f}%")
        results.append(f"   Seen {rec['occurrences']} times")

        if rec["category"]:
            results.append(f"   Category: {rec['category']}")

    results.append("\n" + "=" * 50)
    results.append(f"💰 TOTAL RECURRING MONTHLY: €{total_monthly:.2f}")
    results.append(f"💰 TOTAL RECURRING YEARLY: €{total_monthly * 12:.2f}")

    return "\n".join(results)


@function_tool
def list_subscriptions(ctx: RunContextWrapper[RunDeps]) -> str:
    """List all detected recurring transactions/subscriptions."""
    deps = ctx.context
    cur = deps.db.conn.cursor()

    cur.execute(
        """SELECT * FROM recurring_transactions 
           ORDER BY confidence DESC, amount DESC"""
    )

    subscriptions = cur.fetchall()

    if not subscriptions:
        return "No recurring transactions found. Run detect_recurring first to identify subscriptions."

    results = ["📋 Recurring Transactions & Subscriptions\n" + "=" * 50]

    # Group by frequency
    by_frequency = defaultdict(list)
    for sub in subscriptions:
        by_frequency[sub["frequency"]].append(sub)

    total_monthly = 0

    for freq in ["daily", "weekly", "bi-weekly", "monthly", "quarterly", "yearly"]:
        if freq not in by_frequency:
            continue

        results.append(f"\n📅 {freq.upper()}")

        for sub in by_frequency[freq]:
            # Calculate monthly cost with improved accuracy
            if freq == "daily":
                monthly = sub["amount"] * 30.44  # Average days per month
            elif freq == "weekly":
                monthly = sub["amount"] * 4.345  # 52.14 weeks per year / 12 months
            elif freq == "bi-weekly":
                monthly = sub["amount"] * 2.173  # 26.07 payments per year / 12 months
            elif freq == "monthly":
                monthly = sub["amount"]
            elif freq == "quarterly":
                monthly = sub["amount"] / 3
            elif freq == "yearly":
                monthly = sub["amount"] / 12
            else:  # irregular
                monthly = sub["amount"]  # Conservative estimate

            total_monthly += monthly

            confidence_str = f"({sub['confidence']*100:.0f}% sure)"
            results.append(
                f"   • {sub['description_pattern'][:30]}: €{sub['amount']:.2f} "
                f"{confidence_str}"
            )

    results.append("\n" + "=" * 50)
    results.append(f"💳 Total Monthly Recurring: €{total_monthly:.2f}")
    results.append(f"💳 Total Yearly Recurring: €{total_monthly * 12:.2f}")

    return "\n".join(results)


@function_tool
def analyze_subscription_value(ctx: RunContextWrapper[RunDeps]) -> str:
    """Analyze recurring transactions and suggest which ones might be worth cancelling."""
    deps = ctx.context
    cur = deps.db.conn.cursor()

    # Get all recurring transactions
    cur.execute(
        """SELECT rt.*, 
           (SELECT COUNT(*) FROM transactions t 
            WHERE t.description LIKE '%' || SUBSTR(rt.description_pattern, 1, 10) || '%'
            AND t.date >= date('now', '-30 days')) as recent_usage
           FROM recurring_transactions rt
           WHERE confidence > 0.6
           ORDER BY amount DESC"""
    )

    subscriptions = cur.fetchall()

    if not subscriptions:
        return "No recurring transactions to analyze. Run detect_recurring first."

    # Calculate total spending
    cur.execute(
        """SELECT AVG(ABS(amount)) as avg_monthly 
           FROM transactions 
           WHERE amount < 0 
           AND date >= date('now', '-90 days')"""
    )
    avg_monthly_spending = cur.fetchone()["avg_monthly"] or 0

    results = ["💡 Subscription Value Analysis\n" + "=" * 50]

    high_value = []
    medium_value = []
    low_value = []

    total_monthly_recurring = 0

    for sub in subscriptions:
        # Calculate monthly cost with improved accuracy
        if sub["frequency"] == "daily":
            monthly_cost = sub["amount"] * 30.44  # Average days per month
        elif sub["frequency"] == "weekly":
            monthly_cost = sub["amount"] * 4.345  # 52.14 weeks per year / 12 months
        elif sub["frequency"] == "bi-weekly":
            monthly_cost = sub["amount"] * 2.173  # 26.07 payments per year / 12 months
        elif sub["frequency"] == "monthly":
            monthly_cost = sub["amount"]
        elif sub["frequency"] == "quarterly":
            monthly_cost = sub["amount"] / 3
        elif sub["frequency"] == "yearly":
            monthly_cost = sub["amount"] / 12
        elif sub["frequency"] == "irregular":
            monthly_cost = sub["amount"]  # Conservative estimate
        else:
            monthly_cost = sub["amount"]

        total_monthly_recurring += monthly_cost

        # Calculate value score based on usage and cost
        cost_ratio = (
            monthly_cost / avg_monthly_spending if avg_monthly_spending > 0 else 0
        )
        usage_score = 1 if sub["recent_usage"] > 0 else 0

        # Categorize based on cost and usage
        if usage_score == 0 and monthly_cost > 10:
            low_value.append((sub, monthly_cost, "Unused recently"))
        elif cost_ratio > 0.05:  # More than 5% of spending
            if sub["category"] in ["entertainment", "subscriptions"]:
                medium_value.append((sub, monthly_cost, "High cost entertainment"))
            else:
                high_value.append((sub, monthly_cost, "Essential service"))
        else:
            high_value.append((sub, monthly_cost, "Low cost, likely valuable"))

    # Show recommendations
    if low_value:
        results.append("\n🔴 CONSIDER CANCELLING (Low Value)")
        potential_savings = 0
        for sub, monthly, reason in low_value:
            results.append(
                f"   • {sub['description_pattern'][:30]}: €{monthly:.2f}/month"
            )
            results.append(f"     Reason: {reason}")
            potential_savings += monthly
        results.append(f"   💰 Potential savings: €{potential_savings:.2f}/month")

    if medium_value:
        results.append("\n🟡 REVIEW USAGE (Medium Value)")
        for sub, monthly, reason in medium_value:
            results.append(
                f"   • {sub['description_pattern'][:30]}: €{monthly:.2f}/month"
            )
            results.append(f"     Consider: {reason}")

    if high_value:
        results.append("\n🟢 KEEP (High Value)")
        for sub, monthly, reason in high_value[:5]:  # Show top 5
            results.append(
                f"   • {sub['description_pattern'][:30]}: €{monthly:.2f}/month"
            )

    # Summary
    results.append("\n" + "=" * 50)
    results.append("📊 SUMMARY")
    results.append(f"   Total recurring: €{total_monthly_recurring:.2f}/month")

    if low_value:
        savings = sum(m for _, m, _ in low_value)
        results.append(
            f"   Potential savings: €{savings:.2f}/month (€{savings*12:.2f}/year)"
        )

    percentage = (
        (total_monthly_recurring / avg_monthly_spending * 100)
        if avg_monthly_spending > 0
        else 0
    )
    results.append(f"   Recurring as % of spending: {percentage:.1f}%")

    if percentage > 20:
        results.append("   ⚠️ High percentage - consider reviewing subscriptions")

    return "\n".join(results)


@function_tool
def predict_next_recurring(
    ctx: RunContextWrapper[RunDeps], days_ahead: int = 30
) -> str:
    """Predict upcoming recurring transactions.

    Args:
        days_ahead: How many days ahead to predict
    """
    deps = ctx.context
    cur = deps.db.conn.cursor()

    cur.execute(
        """SELECT * FROM recurring_transactions 
           WHERE confidence > 0.7
           ORDER BY last_seen DESC"""
    )

    recurring = cur.fetchall()

    if not recurring:
        return "No recurring transactions found with sufficient confidence."

    predictions = []
    end_date = datetime.now() + timedelta(days=days_ahead)

    for rec in recurring:
        last_date = datetime.strptime(rec["last_seen"], "%Y-%m-%d")

        # Calculate next occurrence based on frequency with better accuracy
        if rec["frequency"] == "daily":
            next_date = last_date + timedelta(days=1)
            interval = timedelta(days=1)
        elif rec["frequency"] == "weekly":
            next_date = last_date + timedelta(days=7)
            interval = timedelta(days=7)
        elif rec["frequency"] == "bi-weekly":
            next_date = last_date + timedelta(days=14)
            interval = timedelta(days=14)
        elif rec["frequency"] == "monthly":
            # Use more accurate monthly prediction (avoid month-end issues)
            next_date = last_date + timedelta(days=30)
            interval = timedelta(days=30)
        elif rec["frequency"] == "quarterly":
            next_date = last_date + timedelta(days=91)  # More accurate quarter
            interval = timedelta(days=91)
        elif rec["frequency"] == "yearly":
            next_date = last_date + timedelta(days=365)
            interval = timedelta(days=365)
        elif rec["frequency"] == "irregular":
            # For irregular patterns, predict conservatively
            next_date = last_date + timedelta(days=30)
            interval = timedelta(days=45)  # Longer interval for irregular
        else:
            continue

        # Find all occurrences in the prediction period
        while next_date <= end_date:
            if next_date > datetime.now():
                predictions.append(
                    {
                        "date": next_date,
                        "description": rec["description_pattern"],
                        "amount": rec["amount"],
                        "frequency": rec["frequency"],
                        "confidence": rec["confidence"],
                    }
                )
            next_date += interval

    if not predictions:
        return f"No recurring transactions expected in the next {days_ahead} days."

    # Sort by date
    predictions.sort(key=lambda x: x["date"])

    results = [
        f"📅 Predicted Recurring Transactions (next {days_ahead} days)\n" + "=" * 50
    ]

    total_expected = 0
    by_week = defaultdict(float)

    for pred in predictions:
        date_str = pred["date"].strftime("%Y-%m-%d (%A)")
        week_num = pred["date"].isocalendar()[1]

        results.append(f"\n📍 {date_str}")
        results.append(f"   {pred['description'][:40]}")
        results.append(f"   Amount: €{pred['amount']:.2f}")
        results.append(f"   Confidence: {pred['confidence']*100:.0f}%")

        total_expected += pred["amount"]
        by_week[week_num] += pred["amount"]

    results.append("\n" + "=" * 50)
    results.append("💰 SUMMARY")
    results.append(f"   Total expected: €{total_expected:.2f}")
    results.append(f"   Transactions: {len(predictions)}")

    if by_week:
        results.append("\n   By week:")
        for week, amount in sorted(by_week.items())[:4]:
            results.append(f"   Week {week}: €{amount:.2f}")

    return "\n".join(results)


@function_tool
def export_clean_monthly_recurring(
    ctx: RunContextWrapper[RunDeps],
    format: str = "csv",
    months_required: int = 3,
    include_all_bills: bool = False,
) -> str:
    """Export ONLY clean monthly recurring payments (subscriptions, utilities, etc.) for the last 3+ months.
    Filters out irregular transactions, gas stations, and one-time purchases.

    Args:
        format: Export format - "csv", "pdf", "excel", or "json"
        months_required: Minimum months a payment must appear to be considered truly recurring
        include_all_bills: If True, includes more variable bills like utilities and insurance
    """
    deps = ctx.context
    cur = deps.db.conn.cursor()

    # Get the last N months of data
    end_date = datetime.now()
    start_date = end_date - timedelta(
        days=months_required * 31
    )  # Slightly more than months_required months

    # Get transactions from this period, focusing on outgoing payments
    cur.execute(
        """SELECT date, description, amount, category 
           FROM transactions 
           WHERE date >= ? 
           AND amount < 0
           AND ABS(amount) >= 5.00
           ORDER BY description, date""",
        (start_date.strftime("%Y-%m-%d"),),
    )

    transactions = cur.fetchall()

    if not transactions:
        return "No transactions found in the specified period."

    # Group by normalized description and analyze patterns
    patterns = defaultdict(list)

    for tx in transactions:
        # More aggressive normalization for monthly payments
        normalized = _normalize_for_monthly(tx["description"])
        if normalized and normalized != "unknown":
            patterns[normalized].append(tx)

    # Find truly monthly recurring payments
    clean_monthly = []

    for desc_pattern, txs in patterns.items():
        # More lenient requirement: appear in at least (N-1) months
        min_required_months = max(2, months_required - 1)
        if len(txs) < min_required_months:
            continue

        # Group by month to check monthly consistency
        monthly_groups = defaultdict(list)
        for tx in txs:
            tx_date = datetime.strptime(tx["date"], "%Y-%m-%d")
            month_key = f"{tx_date.year}-{tx_date.month:02d}"
            monthly_groups[month_key].append(tx)

        # Must appear in at least (N-1) different months to allow for occasional gaps
        if len(monthly_groups) < min_required_months:
            continue

        # More lenient pattern check - allow more transactions per month for utilities/bills
        monthly_ok = True
        amounts = []
        sample_tx = None
        transaction_counts = []

        for month, month_txs in monthly_groups.items():
            transaction_counts.append(len(month_txs))
            # Allow up to 4 transactions per month (covers utilities, insurance installments)
            if len(month_txs) > 4:
                monthly_ok = False
                break

            # Collect amounts for consistency check
            for tx in month_txs:
                amounts.append(abs(tx["amount"]))
                if sample_tx is None:
                    sample_tx = tx

        if not monthly_ok or not amounts or not sample_tx:
            continue

        # More flexible amount consistency check
        avg_amount = sum(amounts) / len(amounts)
        amount_variance = sum((amount - avg_amount) ** 2 for amount in amounts) / len(
            amounts
        )
        amount_coefficient_variation = (
            (amount_variance**0.5) / avg_amount if avg_amount > 0 else 1
        )

        # Relaxed consistency threshold - allow for seasonal variations in utilities
        consistency_threshold = (
            0.5  # Allow 50% variation for utilities like energy bills
        )

        # Be more lenient for known recurring categories
        desc_lower = desc_pattern.lower()
        if any(
            utility_word in desc_lower
            for utility_word in [
                "energie",
                "energy",
                "gas",
                "water",
                "electric",
                "heating",
                "cooling",
                "verzekering",
                "insurance",
                "health",
                "zorg",
                "medical",
                "telefoon",
                "mobile",
                "internet",
                "telecom",
                "phone",
                "hypotheek",
                "mortgage",
                "rent",
                "huur",
            ]
        ):
            consistency_threshold = 0.7  # Even more lenient for utilities/insurance

        # Only exclude if amounts are extremely inconsistent
        if amount_coefficient_variation > consistency_threshold:
            continue

        # Filter out obvious non-subscription patterns (much more selective now)
        desc_lower = desc_pattern.lower()
        if any(
            skip_word in desc_lower
            for skip_word in [
                "tanken",
                "benzine",
                "shell",
                "bp",
                "esso",  # Gas stations only
                "parking meter",
                "parkeerautomaat",  # Parking meters only
                "restaurant",
                "cafe",
                "bar",
                "mcdonald",  # Dining out only
                "cash",
                "geld opname",
                "geldautomaat",  # Cash withdrawals only
                "amazon.com",
                "bol.com",  # One-off shopping (but allow subscriptions)
            ]
        ):
            continue

        # Calculate monthly cost
        monthly_cost = avg_amount

        # Determine confidence based on consistency and pattern
        confidence = 1.0 - amount_coefficient_variation

        # Boost confidence for known subscription patterns
        if any(
            sub_word in desc_lower
            for sub_word in [
                "netflix",
                "spotify",
                "amazon prime",
                "subscription",
                "abonnement",
                "insurance",
                "verzekering",
                "energie",
                "energy",
                "gas",
                "water",
                "electric",
                "internet",
                "telefoon",
                "mobile",
                "phone",
                "telecom",
                "kpn",
                "ziggo",
                "t-mobile",
                "bank",
                "zorgverzekeraar",
                "health",
                "zorg",
                "cz",
                "vgz",
                "zilveren kruis",
                "hypotheek",
                "mortgage",
                "rent",
                "huur",
                "lease",
                "auto",
                "gym",
                "fitness",
                "sport",
                "subscription",
            ]
        ):
            confidence = min(confidence + 0.3, 0.98)  # Higher confidence boost

        clean_monthly.append(
            {
                "description_pattern": desc_pattern,
                "sample_description": sample_tx["description"],
                "monthly_amount": monthly_cost,
                "months_active": len(monthly_groups),
                "total_transactions": len(txs),
                "confidence": confidence,
                "category": sample_tx["category"],
                "avg_day_of_month": _calculate_avg_day(txs),
            }
        )

    if not clean_monthly:
        return f"No clean monthly recurring payments found. Need payments that appear in at least {months_required} different months."

    # Sort by monthly amount (highest first)
    clean_monthly.sort(key=lambda x: x["monthly_amount"], reverse=True)

    # Generate filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"recurring_payments_{start_date.strftime('%Y-%m-%d')}_to_{end_date.strftime('%Y-%m-%d')}_{timestamp}.{format}"
    filepath = deps.config.documents_dir / filename

    # Export based on format
    if format == "csv":
        _export_clean_csv(clean_monthly, filepath)
    elif format == "json":
        _export_clean_json(clean_monthly, filepath, start_date, end_date)
    elif format == "pdf":
        from ..tools.export import PDF_AVAILABLE

        if not PDF_AVAILABLE:
            return (
                "PDF export requires 'reportlab'. Install with: pip install reportlab"
            )
        _export_clean_pdf(clean_monthly, filepath, start_date, end_date)
    elif format == "excel":
        from ..tools.export import EXCEL_AVAILABLE

        if not EXCEL_AVAILABLE:
            return (
                "Excel export requires 'openpyxl'. Install with: pip install openpyxl"
            )
        _export_clean_excel(clean_monthly, filepath, start_date, end_date)
    else:
        return f"Unsupported format: {format}. Use csv, json, pdf, or excel."

    # Calculate summary
    total_monthly = sum(item["monthly_amount"] for item in clean_monthly)
    total_yearly = total_monthly * 12

    # Create summary output for the user
    summary_lines = [
        "Done—here are your clean monthly recurring expenses for the last 3 months (strict monthly-only, excluding irregulars and one-offs):",
        "",
        "Clean monthly recurring payments",
    ]

    for payment in clean_monthly[:5]:  # Show top 5
        monthly_amount = payment["monthly_amount"]
        summary_lines.append(
            f"- {payment['sample_description']} — €{monthly_amount:.2f} per month"
        )
        summary_lines.append(
            f"  - Months active: {payment['months_active']}; Avg billing day: {payment['avg_day_of_month']}; Confidence: {payment['confidence']*100:.0f}%"
        )

    if len(clean_monthly) > 5:
        summary_lines.append(
            f"- ... and {len(clean_monthly) - 5} more recurring payments"
        )

    summary_lines.extend(
        [
            "",
            "Totals",
            f"- Monthly total: €{total_monthly:.2f}",
            f"- Yearly equivalent: ~€{total_yearly:.2f}",
            "",
            "File saved",
            f"- {format.upper()}: {filename}",
            "- It includes a clean list of only those items that recur every month with stable behavior.",
            "",
            "Notes",
            "- By design, this \"clean monthly\" view filters out irregular transactions, gas/fuel, and variable/erratic bills. That's why some likely monthly bills (e.g., health insurance, energy, telecom, auto/lease) may not appear if they showed multiple charges per month or fluctuating amounts that didn't meet the strict filter.",
            '- Including the ING credit card repayment can double-count if your card purchases are also tracked as separate transactions. For strict "bills and subscriptions only," you might want to exclude credit card repayments and bank fees.',
            "",
            "Would you like me to:",
            "1) Export this as a PDF or Excel as well?",
            '2) Produce a "bills and subscriptions only" version (exclude credit card and bank charges) and include near-monthly items like health insurance, energy, and telecom?',
            "3) Route to the Budget Specialist to review and optimize monthly bills (utilities/telecom) and suggest savings?",
        ]
    )

    return "\n".join(summary_lines)


def _normalize_for_monthly(desc: str) -> str:
    """Normalize description specifically for identifying monthly recurring payments."""
    if not desc:
        return "unknown"

    # Convert to uppercase and clean
    normalized = desc.upper().strip()

    # Remove dates and transaction IDs (more aggressive for subscriptions)
    normalized = re.sub(r"\b\d{2}-\d{2}-\d{4}\b", "", normalized)
    normalized = re.sub(r"\b\d{4}-\d{2}-\d{2}\b", "", normalized)
    normalized = re.sub(r"\b\d{2}/\d{2}/\d{4}\b", "", normalized)
    normalized = re.sub(r"\b\d{2}:\d{2}(:\d{2})?\b", "", normalized)
    normalized = re.sub(r"\b\d{6,}\b", "", normalized)

    # Remove banking specific info that varies
    normalized = re.sub(r"\bPOLISNR[.:]*\s*\w+", "", normalized)
    normalized = re.sub(r"\bIBAN[.:]*\s*[A-Z]{2}\d{2}[A-Z0-9]+", "", normalized)
    normalized = re.sub(r"\bKENMERK[.:]*\s*\w+", "", normalized)
    normalized = re.sub(r"\bMACHTIGING ID[.:]*\s*\w+", "", normalized)
    normalized = re.sub(r"\bINCASS\w+ ID[.:]*\s*[A-Z0-9]+", "", normalized)
    normalized = re.sub(r"\bEREF[.:]*\s*\w+", "", normalized)
    normalized = re.sub(r"\bRTRP[.:]*\s*\w+", "", normalized)

    # Remove period-specific info
    normalized = re.sub(r"\bPERIODE[.:]*\s*[^A-Z]*", "", normalized)
    normalized = re.sub(
        r"\b(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)\w*\s*\d{4}",
        "",
        normalized,
    )
    normalized = re.sub(r"\b\d{4}\b", "", normalized)  # Remove years

    # Remove common banking terms
    banking_terms = [
        "NAAM:",
        "OMSCHRIJVING:",
        "IBAN:",
        "KENMERK:",
        "VALUTADATUM:",
        "BIC:",
        "DOORLOPENDE INCASSO",
        "INCASSO",
        "DATUM/TIJD:",
        "PASVOLGNR:",
        "TRANSACTIE:",
        "TERM:",
        "APPLE PAY",
        "NLD",
        "BETR",
        "VIA",
    ]

    for term in banking_terms:
        normalized = normalized.replace(term, " ")

    # Clean up spaces and punctuation
    normalized = re.sub(r"[^A-Z0-9\s]", " ", normalized)
    normalized = " ".join(normalized.split())

    # Keep only the most relevant part (first 3-4 key words)
    words = normalized.split()
    if len(words) > 4:
        # Prioritize company names, keep first few significant words
        key_words = []
        for word in words:
            if len(word) >= 3 and word not in ["BV", "NV", "LTD", "GMBH"]:
                key_words.append(word)
                if len(key_words) >= 3:
                    break
        normalized = " ".join(key_words) if key_words else normalized[:40]

    return normalized.lower() if normalized else "unknown"


def _calculate_avg_day(transactions: List) -> int:
    """Calculate the average day of month for transactions."""
    days = []
    for tx in transactions:
        tx_date = datetime.strptime(tx["date"], "%Y-%m-%d")
        days.append(tx_date.day)

    return round(sum(days) / len(days)) if days else 15


def _export_clean_csv(monthly_payments: List[Dict], filepath) -> str:
    """Export clean monthly payments to CSV."""
    import csv

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "payment_name",
            "monthly_amount",
            "yearly_amount",
            "months_active",
            "confidence",
            "avg_payment_day",
            "category",
            "sample_description",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for payment in monthly_payments:
            writer.writerow(
                {
                    "payment_name": payment["description_pattern"],
                    "monthly_amount": f"{payment['monthly_amount']:.2f}",
                    "yearly_amount": f"{payment['monthly_amount'] * 12:.2f}",
                    "months_active": payment["months_active"],
                    "confidence": f"{payment['confidence']*100:.0f}%",
                    "avg_payment_day": payment["avg_day_of_month"],
                    "category": payment["category"] or "",
                    "sample_description": payment["sample_description"],
                }
            )

    total_monthly = sum(p["monthly_amount"] for p in monthly_payments)
    return f"Clean CSV with {len(monthly_payments)} monthly payments | Total: €{total_monthly:.2f}/month"


def _export_clean_json(
    monthly_payments: List[Dict], filepath, start_date, end_date
) -> str:
    """Export clean monthly payments to JSON."""
    import json

    export_data = {
        "export_date": datetime.now().isoformat(),
        "analysis_period": {
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d"),
            "months_analyzed": len(
                set(
                    (
                        datetime.strptime(p["sample_description"][:10], "%Y-%m-%d")
                        if len(p["sample_description"]) >= 10
                        else datetime.now()
                    ).strftime("%Y-%m")
                    for p in monthly_payments
                )
            ),
        },
        "summary": {
            "total_monthly_payments": len(monthly_payments),
            "total_monthly_cost": sum(p["monthly_amount"] for p in monthly_payments),
            "total_yearly_cost": sum(
                p["monthly_amount"] * 12 for p in monthly_payments
            ),
            "average_monthly_payment": (
                sum(p["monthly_amount"] for p in monthly_payments)
                / len(monthly_payments)
                if monthly_payments
                else 0
            ),
        },
        "recurring_payments": monthly_payments,
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(export_data, f, indent=2, ensure_ascii=False, default=str)

    return f"JSON with summary and {len(monthly_payments)} clean recurring payments"


def _export_clean_pdf(
    monthly_payments: List[Dict], filepath, start_date, end_date
) -> str:
    """Export clean monthly payments to PDF."""
    from ..tools.export import (
        SimpleDocTemplate,
        Table,
        TableStyle,
        Paragraph,
        Spacer,
        getSampleStyleSheet,
        ParagraphStyle,
        colors,
        letter,
        inch,
    )

    doc = SimpleDocTemplate(str(filepath), pagesize=letter)
    elements = []
    styles = getSampleStyleSheet()

    # Title
    title_style = ParagraphStyle(
        "CleanTitle",
        parent=styles["Title"],
        fontSize=22,
        textColor=colors.HexColor("#1f4788"),
        alignment=1,
    )
    elements.append(Paragraph("Clean Monthly Recurring Payments", title_style))
    elements.append(Spacer(1, 20))

    # Summary
    total_monthly = sum(p["monthly_amount"] for p in monthly_payments)
    total_yearly = total_monthly * 12

    summary_data = [
        ["Summary", "Amount"],
        ["Monthly Recurring Payments Found", str(len(monthly_payments))],
        ["Total Monthly Cost", f"€{total_monthly:.2f}"],
        ["Estimated Yearly Cost", f"€{total_yearly:.2f}"],
        [
            "Analysis Period",
            f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}",
        ],
    ]

    summary_table = Table(summary_data, colWidths=[3 * inch, 2 * inch])
    summary_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.darkblue),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 1, colors.black),
            ]
        )
    )
    elements.append(summary_table)
    elements.append(Spacer(1, 25))

    # Payments table
    elements.append(Paragraph("Monthly Recurring Payments", styles["Heading1"]))
    elements.append(Spacer(1, 12))

    payments_data = [["Payment Name", "Monthly", "Yearly", "Confidence"]]

    for payment in monthly_payments:
        payments_data.append(
            [
                payment["description_pattern"][:35],
                f"€{payment['monthly_amount']:.2f}",
                f"€{payment['monthly_amount'] * 12:.2f}",
                f"{payment['confidence']*100:.0f}%",
            ]
        )

    payments_table = Table(
        payments_data, colWidths=[3 * inch, 1.2 * inch, 1.2 * inch, 1 * inch]
    )
    payments_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
                ("ALIGN", (0, 0), (0, -1), "LEFT"),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
            ]
        )
    )
    elements.append(payments_table)

    doc.build(elements)
    return (
        f"Professional PDF report with {len(monthly_payments)} clean recurring payments"
    )


def _export_clean_excel(
    monthly_payments: List[Dict], filepath, start_date, end_date
) -> str:
    """Export clean monthly payments to Excel."""
    from ..tools.export import openpyxl, Font, PatternFill, get_column_letter

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Monthly Recurring Payments"

    # Title
    ws["A1"] = "Clean Monthly Recurring Payments"
    ws["A1"].font = Font(size=16, bold=True)

    # Summary
    total_monthly = sum(p["monthly_amount"] for p in monthly_payments)
    ws["A3"] = (
        f'Analysis Period: {start_date.strftime("%Y-%m-%d")} to {end_date.strftime("%Y-%m-%d")}'
    )
    ws["A4"] = f"Total Monthly Cost: €{total_monthly:.2f}"
    ws["A5"] = f"Estimated Yearly Cost: €{total_monthly * 12:.2f}"

    # Headers
    headers = [
        "Payment Name",
        "Monthly Amount",
        "Yearly Amount",
        "Months Active",
        "Confidence",
        "Category",
    ]
    header_row = 7

    header_fill = PatternFill(
        start_color="366092", end_color="366092", fill_type="solid"
    )
    header_font = Font(color="FFFFFF", bold=True)

    for col, header in enumerate(headers, 1):
        cell = ws.cell(row=header_row, column=col, value=header)
        cell.fill = header_fill
        cell.font = header_font

    # Data
    for row, payment in enumerate(monthly_payments, header_row + 1):
        ws.cell(row=row, column=1, value=payment["description_pattern"])
        ws.cell(row=row, column=2, value=payment["monthly_amount"])
        ws.cell(row=row, column=3, value=payment["monthly_amount"] * 12)
        ws.cell(row=row, column=4, value=payment["months_active"])
        ws.cell(row=row, column=5, value=f"{payment['confidence']*100:.0f}%")
        ws.cell(row=row, column=6, value=payment["category"] or "")

    # Auto-adjust column widths
    for column_cells in ws.columns:
        length = max(len(str(cell.value or "")) for cell in column_cells)
        ws.column_dimensions[get_column_letter(column_cells[0].column)].width = min(
            length + 2, 50
        )

    wb.save(filepath)
    return f"Excel workbook with {len(monthly_payments)} clean recurring payments"
