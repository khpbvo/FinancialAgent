from __future__ import annotations
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import re
from collections import defaultdict
from agents import RunContextWrapper, function_tool
from ..context import RunDeps


def calculate_frequency(dates: List[str]) -> tuple[str, float]:
    """Calculate the frequency of transactions based on dates.
    
    Returns: (frequency_type, confidence_score)
    """
    if len(dates) < 2:
        return "unknown", 0.0
    
    # Convert to datetime objects and sort
    dt_dates = sorted([datetime.strptime(d, '%Y-%m-%d') for d in dates])
    
    # Group dates that are within 3 days of each other (same occurrence period)
    # This handles cases where multiple transactions happen on consecutive days
    occurrence_groups = []
    current_group = [dt_dates[0]]
    
    for i in range(1, len(dt_dates)):
        # If this date is within 3 days of the last date in current group, add to group
        if (dt_dates[i] - current_group[-1]).days <= 3:
            current_group.append(dt_dates[i])
        else:
            # Start a new group
            occurrence_groups.append(current_group)
            current_group = [dt_dates[i]]
    occurrence_groups.append(current_group)
    
    # Use the first date of each group as the occurrence date
    occurrence_dates = [group[0] for group in occurrence_groups]
    
    if len(occurrence_dates) < 2:
        return "unknown", 0.0
    
    # Calculate intervals between occurrences
    intervals = [(occurrence_dates[i+1] - occurrence_dates[i]).days for i in range(len(occurrence_dates)-1)]
    
    if not intervals:
        return "unknown", 0.0
    
    # Filter out outlier intervals (more than 3x the median)
    intervals_sorted = sorted(intervals)
    median_interval = intervals_sorted[len(intervals_sorted) // 2]
    filtered_intervals = [i for i in intervals if i <= median_interval * 3]
    
    if not filtered_intervals:
        filtered_intervals = intervals
    
    avg_interval = sum(filtered_intervals) / len(filtered_intervals)
    
    # Tighter frequency definitions with better tolerances
    frequencies = [
        (1, "daily", 0),         # Exactly daily (no tolerance)
        (7, "weekly", 2),        # 7 days ± 2 (5-9 days)
        (14, "bi-weekly", 3),    # 14 days ± 3 (11-17 days)
        (30, "monthly", 7),      # 30 days ± 7 (23-37 days)
        (90, "quarterly", 15),   # 90 days ± 15 (75-105 days)
        (365, "yearly", 30),     # 365 days ± 30 (335-395 days)
    ]
    
    best_match = None
    best_confidence = 0.0
    
    for expected, freq_name, tolerance in frequencies:
        if abs(avg_interval - expected) <= tolerance:
            # Calculate confidence based on consistency
            variance = sum((i - avg_interval) ** 2 for i in filtered_intervals) / len(filtered_intervals)
            # Normalize variance by expected interval squared
            normalized_variance = variance / (expected ** 2) if expected > 0 else 1
            # Confidence decreases with variance and increases with more occurrences
            occurrence_bonus = min(0.2, len(occurrence_dates) * 0.05)
            confidence = max(0.1, (1 - normalized_variance) * 0.8 + occurrence_bonus)
            
            if confidence > best_confidence:
                best_match = (freq_name, confidence)
                best_confidence = confidence
    
    if best_match:
        return best_match
    
    # If no clear pattern, check for irregular but recurring
    if len(occurrence_dates) >= 3:
        return "irregular", 0.4
    
    return "unknown", 0.0


def normalize_description(desc: str) -> str:
    """Normalize transaction description for pattern matching."""
    # Remove dates, transaction IDs, and variable numbers
    normalized = re.sub(r'\d{4}-\d{2}-\d{2}', '', desc)
    normalized = re.sub(r'\b\d{4,}\b', '', normalized)
    normalized = re.sub(r'#\d+', '', normalized)
    # Keep only alphanumeric and spaces
    normalized = re.sub(r'[^a-zA-Z0-9\s]', ' ', normalized)
    # Collapse multiple spaces
    normalized = ' '.join(normalized.split()).lower()
    return normalized


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
    
    # Group transactions by normalized description and similar amounts
    patterns = defaultdict(list)
    
    for tx in transactions:
        normalized = normalize_description(tx['description'])
        # Round amount to nearest euro for grouping
        amount_key = round(abs(tx['amount']))
        key = (normalized[:30], amount_key)  # Use first 30 chars of normalized desc
        patterns[key].append(tx)
    
    # Find recurring patterns
    recurring = []
    
    for (desc_pattern, amount), txs in patterns.items():
        if len(txs) >= min_occurrences:
            # Calculate frequency and confidence
            dates = [tx['date'] for tx in txs]
            frequency, confidence = calculate_frequency(dates)
            
            if confidence > 0.6:  # Only include if confidence is reasonable
                avg_amount = sum(abs(tx['amount']) for tx in txs) / len(txs)
                recurring.append({
                    'pattern': desc_pattern,
                    'sample_desc': txs[0]['description'],
                    'amount': avg_amount,
                    'frequency': frequency,
                    'confidence': confidence,
                    'occurrences': len(txs),
                    'category': txs[0]['category'],
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
        
        # Calculate monthly cost
        if rec['frequency'] == 'daily':
            monthly_cost = rec['amount'] * 30
        elif rec['frequency'] == 'weekly':
            monthly_cost = rec['amount'] * 4.33
        elif rec['frequency'] == 'bi-weekly':
            monthly_cost = rec['amount'] * 2.17  # 26 payments per year / 12 months
        elif rec['frequency'] == 'monthly':
            monthly_cost = rec['amount']
        elif rec['frequency'] == 'quarterly':
            monthly_cost = rec['amount'] / 3
        elif rec['frequency'] == 'yearly':
            monthly_cost = rec['amount'] / 12
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
            # Calculate monthly cost
            if freq == 'daily':
                monthly = sub['amount'] * 30
            elif freq == 'weekly':
                monthly = sub['amount'] * 4.33
            elif freq == 'bi-weekly':
                monthly = sub['amount'] * 2.17
            elif freq == 'monthly':
                monthly = sub['amount']
            elif freq == 'quarterly':
                monthly = sub['amount'] / 3
            else:  # yearly
                monthly = sub['amount'] / 12
            
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
        # Calculate monthly cost
        if sub['frequency'] == 'daily':
            monthly_cost = sub['amount'] * 30
        elif sub['frequency'] == 'weekly':
            monthly_cost = sub['amount'] * 4.33
        elif sub['frequency'] == 'bi-weekly':
            monthly_cost = sub['amount'] * 2.17
        elif sub['frequency'] == 'monthly':
            monthly_cost = sub['amount']
        elif sub['frequency'] == 'quarterly':
            monthly_cost = sub['amount'] / 3
        elif sub['frequency'] == 'yearly':
            monthly_cost = sub['amount'] / 12
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
        
        # Calculate next occurrence based on frequency
        if rec['frequency'] == 'daily':
            next_date = last_date + timedelta(days=1)
            interval = timedelta(days=1)
        elif rec['frequency'] == 'weekly':
            next_date = last_date + timedelta(weeks=1)
            interval = timedelta(weeks=1)
        elif rec['frequency'] == 'bi-weekly':
            next_date = last_date + timedelta(weeks=2)
            interval = timedelta(weeks=2)
        elif rec['frequency'] == 'monthly':
            next_date = last_date + timedelta(days=30)
            interval = timedelta(days=30)
        elif rec['frequency'] == 'quarterly':
            next_date = last_date + timedelta(days=90)
            interval = timedelta(days=90)
        elif rec['frequency'] == 'yearly':
            next_date = last_date + timedelta(days=365)
            interval = timedelta(days=365)
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