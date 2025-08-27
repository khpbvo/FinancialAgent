from __future__ import annotations
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import re
from collections import defaultdict
from agents import RunContextWrapper, function_tool
from ..context import RunDeps


def calculate_frequency(dates: List[str]) -> tuple[str, float]:
    """Calculate the frequency of transactions based on dates.
    Uses a month-based approach that's more accurate for recurring transactions.
    
    Returns: (frequency_type, confidence_score)
    """
    if len(dates) < 2:
        return "unknown", 0.0
    
    # Convert to datetime objects and sort
    dt_dates = sorted([datetime.strptime(d, '%Y-%m-%d') for d in dates])
    
    # Group dates by month to detect monthly patterns
    monthly_groups = defaultdict(list)
    for dt in dt_dates:
        month_key = f"{dt.year}-{dt.month:02d}"
        monthly_groups[month_key].append(dt)
    
    # Get unique months with transactions
    months = sorted(monthly_groups.keys())
    
    if len(months) < 2:
        return "unknown", 0.0
    
    # Calculate intervals between months with transactions
    month_dates = [datetime.strptime(month + "-01", "%Y-%m-%d") for month in months]
    month_intervals = [(month_dates[i+1] - month_dates[i]).days for i in range(len(month_dates)-1)]
    
    if not month_intervals:
        return "unknown", 0.0
    
    # Calculate average monthly interval
    avg_month_interval = sum(month_intervals) / len(month_intervals)
    
    # For daily/weekly analysis, use actual transaction dates
    if len(dt_dates) >= 7:  # Need enough data points for daily/weekly analysis
        intervals = [(dt_dates[i+1] - dt_dates[i]).days for i in range(len(dt_dates)-1)]
        avg_day_interval = sum(intervals) / len(intervals)
        
        # Check for daily pattern (transactions every 1-3 days)
        if 1 <= avg_day_interval <= 3 and len(dt_dates) >= 10:
            variance = sum((i - avg_day_interval) ** 2 for i in intervals) / len(intervals)
            confidence = max(0.1, 1 - (variance / (avg_day_interval ** 2)))
            return "daily", min(confidence, 0.9)
        
        # Check for weekly pattern (5-10 day average)
        if 5 <= avg_day_interval <= 10:
            variance = sum((i - avg_day_interval) ** 2 for i in intervals) / len(intervals)
            confidence = max(0.1, 1 - (variance / 49))  # normalized to weekly variance
            return "weekly", min(confidence, 0.9)
    
    # Monthly pattern detection (most common for subscriptions)
    if 25 <= avg_month_interval <= 35:  # ~1 month intervals
        # Calculate confidence based on consistency
        target = 30  # days in a month
        variance = sum((i - target) ** 2 for i in month_intervals) / len(month_intervals)
        confidence = max(0.3, 1 - (variance / (target ** 2)))
        return "monthly", min(confidence, 0.95)
    
    # Bi-weekly pattern (every 2 weeks, so ~14 days)
    if len(dt_dates) >= 4:
        intervals = [(dt_dates[i+1] - dt_dates[i]).days for i in range(len(dt_dates)-1)]
        avg_interval = sum(intervals) / len(intervals)
        if 12 <= avg_interval <= 16:
            variance = sum((i - 14) ** 2 for i in intervals) / len(intervals)
            confidence = max(0.2, 1 - (variance / 196))  # normalized to bi-weekly variance
            return "bi-weekly", min(confidence, 0.9)
    
    # Quarterly pattern (every 3 months)
    if 80 <= avg_month_interval <= 100:  # ~3 month intervals
        target = 90  # days in a quarter
        variance = sum((i - target) ** 2 for i in month_intervals) / len(month_intervals)
        confidence = max(0.2, 1 - (variance / (target ** 2)))
        return "quarterly", min(confidence, 0.9)
    
    # Yearly pattern (every 12 months)
    if 330 <= avg_month_interval <= 400:  # ~1 year intervals
        target = 365  # days in a year
        variance = sum((i - target) ** 2 for i in month_intervals) / len(month_intervals)
        confidence = max(0.2, 1 - (variance / (target ** 2)))
        return "yearly", min(confidence, 0.9)
    
    # If we have multiple months but irregular timing, it's still recurring
    if len(months) >= 3:
        # Look for seasonal or irregular patterns
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
    normalized = re.sub(r'\b\d{2}-\d{2}-\d{4}\b', '', normalized)  # DD-MM-YYYY
    normalized = re.sub(r'\b\d{4}-\d{2}-\d{2}\b', '', normalized)  # YYYY-MM-DD
    normalized = re.sub(r'\b\d{2}/\d{2}/\d{4}\b', '', normalized)  # DD/MM/YYYY
    
    # Remove policy numbers, account numbers, and reference numbers
    normalized = re.sub(r'\bPOLISNR[.:]*\s*\d+', '', normalized)
    normalized = re.sub(r'\bIBAN[.:]*\s*[A-Z]{2}\d{2}[A-Z0-9]+', '', normalized)
    normalized = re.sub(r'\bKENMERK[.:]*\s*\d+', '', normalized)
    normalized = re.sub(r'\bMACHTIGING ID[.:]*\s*\d+', '', normalized)
    normalized = re.sub(r'\bINCASS\w+ ID[.:]*\s*[A-Z0-9]+', '', normalized)
    
    # Remove time stamps and transaction IDs
    normalized = re.sub(r'\b\d{2}:\d{2}(:\d{2})?\b', '', normalized)
    normalized = re.sub(r'\b\d{6,}\b', '', normalized)  # Long numbers (transaction IDs)
    
    # Remove common Dutch banking terms that don't help identify merchants
    banking_terms = [
        'NAAM:', 'OMSCHRIJVING:', 'PERIODE:', 'IBAN:', 'KENMERK:', 'VALUTADATUM:',
        'DOORLOPENDE INCASSO', 'INCASSO', 'DATUM/TIJD:', 'PASVOLGNR:',
        'TRANSACTIE:', 'TERM:', 'APPLE PAY', 'NLD', 'BV', 'NV', 'VOF', 'CV'
    ]
    
    for term in banking_terms:
        normalized = normalized.replace(term, ' ')
    
    # Clean up specific patterns from the sample data
    normalized = re.sub(r'\s+BETR\s+', ' ', normalized)  # "betr" = regarding
    normalized = re.sub(r'\s+VIA\s+', ' ', normalized)   # "via" connector
    
    # Remove standalone single letters and numbers
    normalized = re.sub(r'\b[A-Z0-9]\b', ' ', normalized)
    
    # Keep only alphanumeric characters and spaces, remove special chars
    normalized = re.sub(r'[^A-Z0-9\s]', ' ', normalized)
    
    # Collapse multiple spaces and trim
    normalized = ' '.join(normalized.split())
    
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
        normalized = ' '.join(result) if result else normalized[:40]
    
    return normalized.lower() if normalized else "unknown"


@function_tool
def detect_recurring(
    ctx: RunContextWrapper[RunDeps],
    min_occurrences: int = 3,
    lookback_months: int = 6
) -> str:
    """Detect recurring transactions like subscriptions and regular payments.
    
    Args:
        min_occurrences: Minimum times a transaction must occur to be considered recurring
        lookback_months: How many months back to analyze
    """
    deps = ctx.context
    cur = deps.db.conn.cursor()
    
    # Get transactions from the lookback period
    start_date = (datetime.now() - timedelta(days=lookback_months * 30)).strftime('%Y-%m-%d')
    
    cur.execute(
        """SELECT date, description, amount, category 
           FROM transactions 
           WHERE date >= ? 
           AND amount < 0
           ORDER BY description, date""",
        (start_date,)
    )
    
    transactions = cur.fetchall()
    
    if not transactions:
        return "No transactions found in the specified period."
    
    # Group transactions by normalized description with amount tolerance
    patterns = defaultdict(list)
    
    for tx in transactions:
        normalized = normalize_description(tx['description'])
        if not normalized or normalized == "unknown":
            continue
        
        # Use normalized description as primary key, handle amounts separately
        patterns[normalized].append(tx)
    
    # Find recurring patterns with amount clustering
    recurring = []
    
    for desc_pattern, txs in patterns.items():
        if len(txs) < min_occurrences:
            continue
        
        # Group by similar amounts (within 10% tolerance)
        amount_groups = defaultdict(list)
        for tx in txs:
            amount = abs(tx['amount'])
            # Find existing group within 10% tolerance
            group_key = None
            for existing_amount in amount_groups.keys():
                if abs(amount - existing_amount) <= max(5.0, existing_amount * 0.1):  # 10% or €5 minimum
                    group_key = existing_amount
                    break
            
            if group_key is None:
                group_key = amount
            
            amount_groups[group_key].append(tx)
        
        # Process each amount group separately
        for amount_key, group_txs in amount_groups.items():
            if len(group_txs) >= min_occurrences:
                # Calculate frequency and confidence
                dates = [tx['date'] for tx in group_txs]
                frequency, confidence = calculate_frequency(dates)
                
                # Lower confidence threshold for better detection
                if confidence > 0.3:  # More lenient threshold
                    avg_amount = sum(abs(tx['amount']) for tx in group_txs) / len(group_txs)
                    
                    # Boost confidence for clearly recurring patterns
                    if frequency in ['monthly', 'quarterly'] and len(group_txs) >= 4:
                        confidence = min(confidence + 0.2, 0.95)
                    elif frequency == 'yearly' and len(group_txs) >= 2:
                        confidence = min(confidence + 0.1, 0.9)
                    
                    # Check for amount consistency to boost confidence
                    amounts = [abs(tx['amount']) for tx in group_txs]
                    if len(amounts) > 1:
                        amount_variance = sum((a - avg_amount) ** 2 for a in amounts) / len(amounts)
                        amount_consistency = 1 - (amount_variance / (avg_amount ** 2)) if avg_amount > 0 else 0
                        confidence = min(confidence + (amount_consistency * 0.1), 0.95)
                    
                    recurring.append({
                        'pattern': desc_pattern,
                        'sample_desc': group_txs[0]['description'],
                        'amount': avg_amount,
                        'frequency': frequency,
                        'confidence': confidence,
                        'occurrences': len(group_txs),
                        'category': group_txs[0]['category'],
                        'last_date': max(dates)
                    })
    
    if not recurring:
        return "No recurring transactions detected. Try reducing min_occurrences or increasing lookback period."
    
    # Sort by confidence and amount
    recurring.sort(key=lambda x: (x['confidence'], x['amount']), reverse=True)
    
    # Store in database
    for rec in recurring:
        # Check if already exists
        cur.execute(
            "SELECT id FROM recurring_transactions WHERE description_pattern = ?",
            (rec['pattern'],)
        )
        
        if not cur.fetchone():
            cur.execute(
                """INSERT INTO recurring_transactions 
                   (description_pattern, amount, frequency, category, last_seen, confidence)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (rec['pattern'], rec['amount'], rec['frequency'], 
                 rec['category'], rec['last_date'], rec['confidence'])
            )
    
    deps.db.conn.commit()
    
    # Format results
    results = [f"🔄 Detected {len(recurring)} Recurring Transactions\n" + "=" * 50]
    
    total_monthly = 0
    
    for rec in recurring:
        conf_percent = rec['confidence'] * 100
        
        # Calculate monthly cost with improved accuracy
        if rec['frequency'] == 'daily':
            monthly_cost = rec['amount'] * 30.44  # Average days per month
        elif rec['frequency'] == 'weekly':
            monthly_cost = rec['amount'] * 4.345  # 52.14 weeks per year / 12 months
        elif rec['frequency'] == 'bi-weekly':
            monthly_cost = rec['amount'] * 2.173  # 26.07 payments per year / 12 months
        elif rec['frequency'] == 'monthly':
            monthly_cost = rec['amount']
        elif rec['frequency'] == 'quarterly':
            monthly_cost = rec['amount'] / 3
        elif rec['frequency'] == 'yearly':
            monthly_cost = rec['amount'] / 12
        elif rec['frequency'] == 'irregular':
            # For irregular patterns, estimate based on occurrences over time period
            monthly_cost = rec['amount']  # Conservative estimate
        else:
            monthly_cost = rec['amount']
        
        total_monthly += monthly_cost
        
        # Determine icon based on confidence
        icon = "🟢" if conf_percent > 80 else "🟡" if conf_percent > 65 else "🔴"
        
        results.append(f"\n{icon} {rec['sample_desc'][:40]}")
        results.append(f"   Amount: €{rec['amount']:.2f} {rec['frequency']}")
        results.append(f"   Monthly cost: €{monthly_cost:.2f}")
        results.append(f"   Confidence: {conf_percent:.0f}%")
        results.append(f"   Seen {rec['occurrences']} times")
        
        if rec['category']:
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
        by_frequency[sub['frequency']].append(sub)
    
    total_monthly = 0
    
    for freq in ['daily', 'weekly', 'bi-weekly', 'monthly', 'quarterly', 'yearly']:
        if freq not in by_frequency:
            continue
        
        results.append(f"\n📅 {freq.upper()}")
        
        for sub in by_frequency[freq]:
            # Calculate monthly cost with improved accuracy
            if freq == 'daily':
                monthly = sub['amount'] * 30.44  # Average days per month
            elif freq == 'weekly':
                monthly = sub['amount'] * 4.345  # 52.14 weeks per year / 12 months
            elif freq == 'bi-weekly':
                monthly = sub['amount'] * 2.173  # 26.07 payments per year / 12 months
            elif freq == 'monthly':
                monthly = sub['amount']
            elif freq == 'quarterly':
                monthly = sub['amount'] / 3
            elif freq == 'yearly':
                monthly = sub['amount'] / 12
            else:  # irregular
                monthly = sub['amount']  # Conservative estimate
            
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
    avg_monthly_spending = cur.fetchone()['avg_monthly'] or 0
    
    results = ["💡 Subscription Value Analysis\n" + "=" * 50]
    
    high_value = []
    medium_value = []
    low_value = []
    
    total_monthly_recurring = 0
    
    for sub in subscriptions:
        # Calculate monthly cost with improved accuracy
        if sub['frequency'] == 'daily':
            monthly_cost = sub['amount'] * 30.44  # Average days per month
        elif sub['frequency'] == 'weekly':
            monthly_cost = sub['amount'] * 4.345  # 52.14 weeks per year / 12 months
        elif sub['frequency'] == 'bi-weekly':
            monthly_cost = sub['amount'] * 2.173  # 26.07 payments per year / 12 months
        elif sub['frequency'] == 'monthly':
            monthly_cost = sub['amount']
        elif sub['frequency'] == 'quarterly':
            monthly_cost = sub['amount'] / 3
        elif sub['frequency'] == 'yearly':
            monthly_cost = sub['amount'] / 12
        elif sub['frequency'] == 'irregular':
            monthly_cost = sub['amount']  # Conservative estimate
        else:
            monthly_cost = sub['amount']
        
        total_monthly_recurring += monthly_cost
        
        # Calculate value score based on usage and cost
        cost_ratio = monthly_cost / avg_monthly_spending if avg_monthly_spending > 0 else 0
        usage_score = 1 if sub['recent_usage'] > 0 else 0
        
        # Categorize based on cost and usage
        if usage_score == 0 and monthly_cost > 10:
            low_value.append((sub, monthly_cost, "Unused recently"))
        elif cost_ratio > 0.05:  # More than 5% of spending
            if sub['category'] in ['entertainment', 'subscriptions']:
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
            results.append(f"   • {sub['description_pattern'][:30]}: €{monthly:.2f}/month")
            results.append(f"     Reason: {reason}")
            potential_savings += monthly
        results.append(f"   💰 Potential savings: €{potential_savings:.2f}/month")
    
    if medium_value:
        results.append("\n🟡 REVIEW USAGE (Medium Value)")
        for sub, monthly, reason in medium_value:
            results.append(f"   • {sub['description_pattern'][:30]}: €{monthly:.2f}/month")
            results.append(f"     Consider: {reason}")
    
    if high_value:
        results.append("\n🟢 KEEP (High Value)")
        for sub, monthly, reason in high_value[:5]:  # Show top 5
            results.append(f"   • {sub['description_pattern'][:30]}: €{monthly:.2f}/month")
    
    # Summary
    results.append("\n" + "=" * 50)
    results.append("📊 SUMMARY")
    results.append(f"   Total recurring: €{total_monthly_recurring:.2f}/month")
    
    if low_value:
        savings = sum(m for _, m, _ in low_value)
        results.append(f"   Potential savings: €{savings:.2f}/month (€{savings*12:.2f}/year)")
    
    percentage = (total_monthly_recurring / avg_monthly_spending * 100) if avg_monthly_spending > 0 else 0
    results.append(f"   Recurring as % of spending: {percentage:.1f}%")
    
    if percentage > 20:
        results.append("   ⚠️ High percentage - consider reviewing subscriptions")
    
    return "\n".join(results)


@function_tool
def predict_next_recurring(
    ctx: RunContextWrapper[RunDeps],
    days_ahead: int = 30
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
        last_date = datetime.strptime(rec['last_seen'], '%Y-%m-%d')
        
        # Calculate next occurrence based on frequency with better accuracy
        if rec['frequency'] == 'daily':
            next_date = last_date + timedelta(days=1)
            interval = timedelta(days=1)
        elif rec['frequency'] == 'weekly':
            next_date = last_date + timedelta(days=7)
            interval = timedelta(days=7)
        elif rec['frequency'] == 'bi-weekly':
            next_date = last_date + timedelta(days=14)
            interval = timedelta(days=14)
        elif rec['frequency'] == 'monthly':
            # Use more accurate monthly prediction (avoid month-end issues)
            next_date = last_date + timedelta(days=30)
            interval = timedelta(days=30)
        elif rec['frequency'] == 'quarterly':
            next_date = last_date + timedelta(days=91)  # More accurate quarter
            interval = timedelta(days=91)
        elif rec['frequency'] == 'yearly':
            next_date = last_date + timedelta(days=365)
            interval = timedelta(days=365)
        elif rec['frequency'] == 'irregular':
            # For irregular patterns, predict conservatively
            next_date = last_date + timedelta(days=30)
            interval = timedelta(days=45)  # Longer interval for irregular
        else:
            continue
        
        # Find all occurrences in the prediction period
        while next_date <= end_date:
            if next_date > datetime.now():
                predictions.append({
                    'date': next_date,
                    'description': rec['description_pattern'],
                    'amount': rec['amount'],
                    'frequency': rec['frequency'],
                    'confidence': rec['confidence']
                })
            next_date += interval
    
    if not predictions:
        return f"No recurring transactions expected in the next {days_ahead} days."
    
    # Sort by date
    predictions.sort(key=lambda x: x['date'])
    
    results = [f"📅 Predicted Recurring Transactions (next {days_ahead} days)\n" + "=" * 50]
    
    total_expected = 0
    by_week = defaultdict(float)
    
    for pred in predictions:
        date_str = pred['date'].strftime('%Y-%m-%d (%A)')
        week_num = pred['date'].isocalendar()[1]
        
        results.append(f"\n📍 {date_str}")
        results.append(f"   {pred['description'][:40]}")
        results.append(f"   Amount: €{pred['amount']:.2f}")
        results.append(f"   Confidence: {pred['confidence']*100:.0f}%")
        
        total_expected += pred['amount']
        by_week[week_num] += pred['amount']
    
    results.append("\n" + "=" * 50)
    results.append("💰 SUMMARY")
    results.append(f"   Total expected: €{total_expected:.2f}")
    results.append(f"   Transactions: {len(predictions)}")
    
    if by_week:
        results.append("\n   By week:")
        for week, amount in sorted(by_week.items())[:4]:
            results.append(f"   Week {week}: €{amount:.2f}")
    
    return "\n".join(results)