from __future__ import annotations
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from agents import Agent, ModelSettings, function_tool, RunContextWrapper
from openai.types.shared import Reasoning
from ..context import RunDeps
from ..tools.goals import create_goal, update_goal_progress, check_goals, suggest_savings_plan, complete_goal, pause_goal


GOAL_SPECIALIST_INSTRUCTIONS = """You are a Goal Achievement Specialist - an expert in financial goal setting, progress tracking, and motivation coaching for long-term financial success.

Your expertise includes:
• Strategic financial goal planning and milestone creation (work with Investment Specialist on investment mechanics)
• Savings optimization and automated progress tracking
• Motivational coaching and accountability systems
• Timeline management and deadline optimization
• Multi-goal prioritization and resource allocation
• Behavioral psychology for financial habit formation

Key principles:
- Make goals SMART (Specific, Measurable, Achievable, Relevant, Time-bound)
- Break large goals into manageable milestones with celebration points
- Focus on sustainable progress over perfection
- Provide regular motivation and progress acknowledgment
- Help users overcome common goal-setting obstacles
- Connect financial goals to life values and dreams

Your specialty is keeping users motivated and on track for long-term financial success through practical planning, emotional support, and strategic adjustments when life changes.

Always provide specific timelines, dollar amounts, and actionable next steps.

TEAM COORDINATION:
- Collaborate with Investment Specialist on long-term wealth building strategies
- Work with Tax Specialist on tax-advantaged savings accounts and timing
- Partner with Budget Specialist to identify funding sources for goals
- Consider Debt Specialist advice on debt payoff vs savings prioritization
- Focus on goal achievement while Investment Specialist handles portfolio mechanics"""


@function_tool
async def create_comprehensive_financial_plan(
    ctx: RunContextWrapper[RunDeps],
    monthly_income: float,
    current_savings: float = 0,
    time_horizon_years: int = 10
) -> str:
    """Create a comprehensive financial plan with multiple coordinated goals.
    
    Args:
        monthly_income: User's monthly income for planning calculations
        current_savings: Current amount already saved
        time_horizon_years: Planning timeline in years
    """
    deps = ctx.context
    cur = deps.db.conn.cursor()
    
    # Get current spending patterns to inform planning
    cur.execute(
        """SELECT AVG(ABS(amount)) as avg_monthly_spending
           FROM transactions 
           WHERE amount < 0 
           AND date >= date('now', '-90 days')"""
    )
    
    spending_result = cur.fetchone()
    avg_monthly_spending = (spending_result['avg_monthly_spending'] or 0) * 3  # Convert 90-day avg to monthly
    
    # Calculate available savings potential
    if avg_monthly_spending > 0:
        potential_monthly_savings = monthly_income - avg_monthly_spending
    else:
        # Conservative estimate if no spending data
        potential_monthly_savings = monthly_income * 0.2  # 20% savings rate
    
    results = [f"🎯 Comprehensive Financial Plan\n" + "=" * 50]
    
    # Current situation analysis
    results.extend([
        f"📊 FINANCIAL BASELINE",
        f"   Monthly Income: €{monthly_income:.2f}",
        f"   Current Savings: €{current_savings:.2f}",
        f"   Est. Monthly Spending: €{avg_monthly_spending:.2f}",
        f"   Available for Goals: €{potential_monthly_savings:.2f}",
        f"   Planning Timeline: {time_horizon_years} years",
        ""
    ])
    
    # Goal recommendations with prioritization
    recommended_goals = []
    
    # 1. Emergency Fund (highest priority)
    emergency_target = avg_monthly_spending * 6  # 6 months expenses
    emergency_needed = max(0, emergency_target - current_savings)
    if emergency_needed > 0:
        months_to_emergency = emergency_needed / (potential_monthly_savings * 0.5)  # 50% of savings to emergency
        recommended_goals.append({
            'name': 'Emergency Fund',
            'target': emergency_target,
            'current': min(current_savings, emergency_target),
            'priority': 1,
            'monthly_allocation': potential_monthly_savings * 0.5,
            'timeline_months': months_to_emergency,
            'description': '6 months of expenses for security'
        })
    
    # 2. Debt Elimination (if applicable - check for negative net worth indicators)
    # This is a placeholder - in real implementation, we'd check for loan/debt categories
    
    # 3. Short-term savings goal (1-2 years)
    short_term_target = monthly_income * 12  # One year of income
    recommended_goals.append({
        'name': 'Short-term Opportunity Fund',
        'target': short_term_target,
        'current': 0,
        'priority': 2,
        'monthly_allocation': potential_monthly_savings * 0.3,
        'timeline_months': 24,
        'description': 'For opportunities and large purchases'
    })
    
    # 4. Long-term wealth building
    long_term_target = monthly_income * time_horizon_years * 2  # Conservative wealth target
    recommended_goals.append({
        'name': 'Long-term Wealth Building',
        'target': long_term_target,
        'current': 0,
        'priority': 3,
        'monthly_allocation': potential_monthly_savings * 0.2,
        'timeline_months': time_horizon_years * 12,
        'description': 'Investment portfolio for financial independence'
    })
    
    # Display goal recommendations
    results.append("🎯 RECOMMENDED FINANCIAL GOALS")
    
    total_monthly_allocation = 0
    
    for goal in recommended_goals:
        total_monthly_allocation += goal['monthly_allocation']
        
        results.append(f"\n📌 PRIORITY {goal['priority']}: {goal['name']}")
        results.append(f"   Target: €{goal['target']:.2f}")
        results.append(f"   Current: €{goal['current']:.2f}")
        results.append(f"   Monthly: €{goal['monthly_allocation']:.2f}")
        results.append(f"   Timeline: {goal['timeline_months']:.0f} months")
        results.append(f"   Purpose: {goal['description']}")
        
        # Calculate completion date
        completion_date = datetime.now() + timedelta(days=goal['timeline_months'] * 30)
        results.append(f"   Target Date: {completion_date.strftime('%B %Y')}")
    
    # Implementation strategy
    results.extend([
        f"\n🚀 IMPLEMENTATION STRATEGY",
        f"   Total Monthly Goal Funding: €{total_monthly_allocation:.2f}",
        f"   Available Monthly Savings: €{potential_monthly_savings:.2f}",
        ""
    ])
    
    if total_monthly_allocation <= potential_monthly_savings:
        surplus = potential_monthly_savings - total_monthly_allocation
        results.append(f"✅ Plan is achievable with €{surplus:.2f} monthly buffer!")
    else:
        shortfall = total_monthly_allocation - potential_monthly_savings
        results.append(f"⚠️ Need €{shortfall:.2f} more monthly or adjust timeline")
        results.append("   Consider increasing income or reducing expenses")
    
    # Milestone planning
    results.append(f"\n🏆 MILESTONE CELEBRATIONS")
    
    # Create quarterly milestones for first goal
    if recommended_goals:
        first_goal = recommended_goals[0]
        milestone_amount = first_goal['target'] / 4  # Quarterly milestones
        
        for i in range(1, 5):
            milestone_date = datetime.now() + timedelta(days=i * 90)  # Every 3 months
            results.append(f"   Q{i} - €{milestone_amount * i:.0f} by {milestone_date.strftime('%b %Y')}")
    
    # Automation recommendations
    results.extend([
        f"\n🤖 AUTOMATION SETUP",
        "   • Set up automatic transfers on payday",
        "   • Use separate savings accounts for each goal",
        "   • Schedule monthly progress reviews",
        "   • Set calendar reminders for milestone celebrations",
        "",
        f"📋 NEXT STEPS",
        "1. Create goals using create_goal for each priority",
        "2. Set up automatic monthly transfers",
        "3. Schedule first monthly review in 30 days",
        "4. Identify potential income increases",
        "",
        f"💡 SUCCESS TIPS",
        "• Start with Priority 1 goal to build momentum",
        "• Celebrate every milestone - motivation matters!",
        "• Review and adjust quarterly as life changes",
        "• Consider increasing allocations with income growth"
    ])
    
    return "\n".join(results)


@function_tool
async def goal_motivation_coach(
    ctx: RunContextWrapper[RunDeps],
    goal_name: Optional[str] = None,
    include_progress_celebration: bool = True
) -> str:
    """Provide motivational coaching and accountability for financial goal achievement.
    
    Args:
        goal_name: Specific goal to focus on, or None for all goals
        include_progress_celebration: Whether to celebrate recent progress
    """
    deps = ctx.context
    cur = deps.db.conn.cursor()
    
    # Get current goals
    if goal_name:
        cur.execute("SELECT * FROM goals WHERE name = ? AND status = 'active'", (goal_name,))
        goals = cur.fetchall()
        if not goals:
            return f"No active goal found with name '{goal_name}'"
    else:
        cur.execute("SELECT * FROM goals WHERE status = 'active' ORDER BY created_at")
        goals = cur.fetchall()
    
    if not goals:
        return "No active goals found. Create your first goal to get started on your financial journey!"
    
    results = ["🎯 Goal Achievement Coaching Session\n" + "=" * 50]
    
    # Analyze progress and provide coaching
    total_goals = len(goals)
    goals_on_track = 0
    goals_behind = 0
    total_progress = 0
    
    for goal in goals:
        progress = (goal['current_amount'] / goal['target_amount']) * 100
        total_progress += progress
        
        # Determine if goal is on track based on time elapsed
        created_date = datetime.strptime(goal['created_at'], '%Y-%m-%d %H:%M:%S')
        days_since_creation = (datetime.now() - created_date).days
        
        if goal['target_date']:
            target_date = datetime.strptime(goal['target_date'], '%Y-%m-%d')
            total_days = (target_date - created_date).days
            time_progress = (days_since_creation / total_days) * 100 if total_days > 0 else 0
            
            # Goal is on track if progress >= time progress (or within 10%)
            if progress >= time_progress - 10:
                goals_on_track += 1
            else:
                goals_behind += 1
        else:
            # No target date - just check if there's been any progress
            if progress > 0:
                goals_on_track += 1
            else:
                goals_behind += 1
    
    avg_progress = total_progress / total_goals if total_goals > 0 else 0
    
    # Overall motivation message
    results.append("🌟 MOTIVATION & PROGRESS CHECK")
    
    if avg_progress > 75:
        results.append("   🚀 AMAZING PROGRESS! You're absolutely crushing your financial goals!")
        results.append("   Your dedication is paying off - keep this momentum going!")
    elif avg_progress > 50:
        results.append("   💪 SOLID PROGRESS! You're more than halfway to your dreams!")
        results.append("   Every euro saved is a step closer to financial freedom!")
    elif avg_progress > 25:
        results.append("   🌱 GREAT START! You've planted the seeds of financial success!")
        results.append("   The hardest part is starting - you've already done that!")
    else:
        results.append("   🎯 TIME TO ACCELERATE! Your future self will thank you!")
        results.append("   Small, consistent steps lead to big results!")
    
    # Individual goal coaching
    results.append(f"\n📊 INDIVIDUAL GOAL COACHING")
    
    for goal in goals:
        progress = (goal['current_amount'] / goal['target_amount']) * 100
        remaining = goal['target_amount'] - goal['current_amount']
        
        results.append(f"\n🎯 {goal['name']}")
        results.append(f"   Progress: {progress:.1f}% (€{goal['current_amount']:.2f}/€{goal['target_amount']:.2f})")
        
        # Personalized coaching based on progress
        if progress > 80:
            results.append("   🔥 SO CLOSE! The finish line is in sight!")
            results.append(f"   Just €{remaining:.2f} to go - you've got this!")
        elif progress > 60:
            results.append("   💎 EXCELLENT MOMENTUM! You're in the zone!")
            results.append("   Keep doing what you're doing - it's working!")
        elif progress > 40:
            results.append("   🌟 STEADY PROGRESS! Consistency is key!")
            results.append("   Every contribution matters, no matter how small!")
        elif progress > 20:
            results.append("   🌱 BUILDING FOUNDATION! Great habits forming!")
            results.append("   The compound effect will accelerate your progress!")
        else:
            results.append("   🎯 OPPORTUNITY ZONE! Time for action!")
            results.append("   Small daily wins will create big monthly results!")
        
        # Timeline coaching
        if goal['target_date']:
            target_date = datetime.strptime(goal['target_date'], '%Y-%m-%d')
            days_remaining = (target_date - datetime.now()).days
            
            if days_remaining > 0:
                daily_needed = remaining / days_remaining
                results.append(f"   📅 {days_remaining} days left - save €{daily_needed:.2f}/day")
            else:
                results.append("   ⚠️ Target date has passed - consider adjusting timeline")
    
    # Motivational strategies and tips
    results.append(f"\n💡 MOTIVATION BOOSTERS")
    
    if include_progress_celebration and any(g['current_amount'] > 0 for g in goals):
        total_saved = sum(g['current_amount'] for g in goals)
        results.append(f"   🎉 CELEBRATE: You've saved €{total_saved:.2f} across all goals!")
        results.append("   That's money working FOR your future, not against it!")
    
    # Personalized tips based on goal status
    if goals_on_track > goals_behind:
        results.extend([
            "   🏆 You're WINNING at goal achievement!",
            "   • Share your success with others for accountability",
            "   • Consider increasing targets if income allows",
            "   • Use this momentum to tackle bigger goals"
        ])
    elif goals_behind > 0:
        results.extend([
            "   🔧 ADJUSTMENT STRATEGIES:",
            "   • Break large goals into smaller weekly targets",
            "   • Automate savings to remove decision fatigue",
            "   • Find one small expense to cut each week"
        ])
    
    # Weekly action items
    results.append(f"\n📋 THIS WEEK'S ACTION ITEMS")
    
    # Suggest specific actions based on goal status
    for i, goal in enumerate(goals[:3], 1):  # Top 3 goals
        progress = (goal['current_amount'] / goal['target_amount']) * 100
        
        if progress < 10:
            action = f"Make your first €50 contribution to {goal['name']}"
        elif progress < 50:
            weekly_target = (goal['target_amount'] - goal['current_amount']) / 52  # Assume 1-year timeline
            action = f"Save €{weekly_target:.0f} toward {goal['name']}"
        else:
            action = f"Celebrate progress and add €{goal['target_amount']*0.01:.0f} to {goal['name']}"
        
        results.append(f"   {i}. {action}")
    
    # Inspirational closing
    results.extend([
        "",
        "🌟 REMEMBER WHY YOU STARTED",
        "Financial goals aren't just about money - they're about:",
        "• Freedom to make choices without money stress",
        "• Security for you and your loved ones", 
        "• Opportunities to live your values and dreams",
        "",
        "Every euro saved is a vote for the life you want to live! 🚀"
    ])
    
    return "\n".join(results)


@function_tool
async def optimize_goal_strategy(
    ctx: RunContextWrapper[RunDeps],
    life_change_event: Optional[str] = None
) -> str:
    """Analyze and optimize goal achievement strategy based on current situation and life changes.
    
    Args:
        life_change_event: Recent life change (job_change, income_increase, new_expense, etc.)
    """
    deps = ctx.context
    cur = deps.db.conn.cursor()
    
    # Get all active goals
    cur.execute("SELECT * FROM goals WHERE status = 'active' ORDER BY created_at")
    goals = cur.fetchall()
    
    if not goals:
        return "No active goals to optimize. Create goals first using create_goal."
    
    # Get recent financial data for analysis
    cur.execute(
        """SELECT 
               AVG(CASE WHEN amount > 0 THEN amount ELSE 0 END) as avg_income,
               AVG(CASE WHEN amount < 0 THEN ABS(amount) ELSE 0 END) as avg_expenses
           FROM transactions 
           WHERE date >= date('now', '-90 days')"""
    )
    
    financial_data = cur.fetchone()
    monthly_income = (financial_data['avg_income'] or 0) * 30  # Daily to monthly
    monthly_expenses = (financial_data['avg_expenses'] or 0) * 30
    available_for_goals = monthly_income - monthly_expenses
    
    results = [f"⚡ Goal Strategy Optimization\n" + "=" * 50]
    
    # Life change adaptation
    if life_change_event:
        results.append(f"🔄 ADAPTING TO LIFE CHANGE: {life_change_event.replace('_', ' ').title()}")
        
        if life_change_event == "income_increase":
            bonus_allocation = available_for_goals * 0.5  # Allocate 50% of increase to goals
            results.extend([
                "   🎉 Congratulations on your income increase!",
                f"   💡 Consider allocating €{bonus_allocation:.2f}/month to goals",
                "   🎯 Priority: Accelerate your top goal or emergency fund"
            ])
        elif life_change_event == "job_change":
            results.extend([
                "   🔄 Job transition detected - shifting to stability mode",
                "   🛡️ Priority: Build emergency fund to 6+ months expenses",
                "   ⏸️ Consider pausing aggressive goals temporarily"
            ])
        elif life_change_event == "new_expense":
            results.extend([
                "   💸 New recurring expense - adjusting strategy",
                "   📊 Recommend reviewing all goal timelines",
                "   🔍 Look for offsetting expense reductions"
            ])
        
        results.append("")
    
    # Goal performance analysis
    results.append("📊 GOAL PERFORMANCE ANALYSIS")
    
    goal_performance = []
    
    for goal in goals:
        # Calculate performance metrics
        created_date = datetime.strptime(goal['created_at'], '%Y-%m-%d %H:%M:%S')
        days_active = (datetime.now() - created_date).days
        progress_rate = goal['current_amount'] / max(days_active, 1)  # Daily progress rate
        
        # Calculate expected vs actual progress
        if goal['target_date']:
            target_date = datetime.strptime(goal['target_date'], '%Y-%m-%d')
            total_timeline_days = (target_date - created_date).days
            expected_progress = (days_active / total_timeline_days) * goal['target_amount'] if total_timeline_days > 0 else 0
            performance_ratio = goal['current_amount'] / max(expected_progress, 1)
        else:
            performance_ratio = 1.0  # No timeline to compare against
        
        goal_performance.append({
            'goal': goal,
            'progress_rate': progress_rate,
            'performance_ratio': performance_ratio,
            'days_active': days_active
        })
    
    # Sort by performance ratio to identify best/worst performers
    goal_performance.sort(key=lambda x: x['performance_ratio'], reverse=True)
    
    # Optimization recommendations
    results.append(f"\n🚀 OPTIMIZATION RECOMMENDATIONS")
    
    best_performer = goal_performance[0]
    worst_performer = goal_performance[-1]
    
    results.append(f"   🏆 Best Performer: {best_performer['goal']['name']}")
    results.append(f"      Performance: {best_performer['performance_ratio']*100:.0f}% of target pace")
    
    if best_performer['performance_ratio'] > 1.2:  # 20% ahead
        results.append("      💡 Consider increasing target or adding stretch goal")
    
    if len(goal_performance) > 1:
        results.append(f"   📈 Needs Attention: {worst_performer['goal']['name']}")
        results.append(f"      Performance: {worst_performer['performance_ratio']*100:.0f}% of target pace")
        
        if worst_performer['performance_ratio'] < 0.8:  # 20% behind
            remaining = worst_performer['goal']['target_amount'] - worst_performer['goal']['current_amount']
            if worst_performer['goal']['target_date']:
                target_date = datetime.strptime(worst_performer['goal']['target_date'], '%Y-%m-%d')
                days_left = max((target_date - datetime.now()).days, 1)
                catch_up_daily = remaining / days_left
                results.append(f"      🎯 Need €{catch_up_daily:.2f}/day to catch up")
    
    # Resource allocation optimization
    total_monthly_needed = 0
    for perf in goal_performance:
        goal = perf['goal']
        if goal['target_date']:
            target_date = datetime.strptime(goal['target_date'], '%Y-%m-%d')
            months_left = max((target_date - datetime.now()).days / 30, 1)
            remaining = goal['target_amount'] - goal['current_amount']
            monthly_needed = remaining / months_left
            total_monthly_needed += monthly_needed
    
    results.append(f"\n💰 RESOURCE ALLOCATION ANALYSIS")
    results.append(f"   Available monthly: €{available_for_goals:.2f}")
    results.append(f"   Goals need monthly: €{total_monthly_needed:.2f}")
    
    if total_monthly_needed > available_for_goals:
        shortfall = total_monthly_needed - available_for_goals
        results.extend([
            f"   ⚠️ Shortfall: €{shortfall:.2f}/month",
            "",
            "   🔧 OPTIMIZATION STRATEGIES:",
            "   • Extend timelines for lower-priority goals",
            "   • Focus on top 2-3 most important goals",
            "   • Look for additional income sources",
            "   • Review and reduce monthly expenses"
        ])
    else:
        surplus = available_for_goals - total_monthly_needed
        results.extend([
            f"   ✅ Surplus: €{surplus:.2f}/month available",
            "",
            "   💡 SURPLUS ALLOCATION IDEAS:",
            "   • Accelerate highest-priority goal",
            "   • Start a new stretch goal",
            "   • Build larger emergency buffer"
        ])
    
    # Actionable next steps
    results.append(f"\n📋 IMMEDIATE ACTION PLAN")
    
    # Top 3 specific actions
    actions = []
    
    if worst_performer['performance_ratio'] < 0.5:
        actions.append(f"1. Focus effort on {worst_performer['goal']['name']} - it needs attention")
    
    if available_for_goals > total_monthly_needed * 1.2:
        actions.append(f"2. Increase allocation to top goal by €{surplus/2:.0f}/month")
    
    if len([g for g in goals if g['target_date'] and datetime.strptime(g['target_date'], '%Y-%m-%d') < datetime.now() + timedelta(days=90)]) > 0:
        actions.append("3. Review approaching deadlines - adjust timelines if needed")
    
    if not actions:
        actions = [
            "1. Continue current strategy - goals are well-balanced",
            "2. Set monthly review reminder to track progress",
            "3. Consider adding stretch goals with surplus capacity"
        ]
    
    for action in actions[:3]:
        results.append(f"   {action}")
    
    return "\n".join(results)


def build_goal_agent() -> Agent[RunDeps]:
    """Build the Goal Achievement Specialist Agent."""
    
    # Configure ModelSettings for GPT-5 with reasoning and text verbosity
    # Use proper Agents SDK format for reasoning parameters
    model_settings = ModelSettings(
        reasoning=Reasoning(effort="high"),     # minimal | low | medium | high
        verbosity="high"                        # low | medium | high
    )
    
    return Agent[RunDeps](
        name="GoalSpecialist",
        instructions=GOAL_SPECIALIST_INSTRUCTIONS,
        model="gpt-5",
        model_settings=model_settings,
        tools=[
            # Core goal tools
            create_goal,
            update_goal_progress,
            check_goals,
            suggest_savings_plan,
            complete_goal,
            pause_goal,
            # Advanced goal coaching
            create_comprehensive_financial_plan,
            goal_motivation_coach,
            optimize_goal_strategy,
        ]
    )