import math
from datetime import datetime
from typing import Dict, List, Optional, Any


def should_use_fallback(usage_history_length: int) -> bool:
    """
    Determine if fallback rule-based alerting should be used instead of ML.
    
    Threshold: 5 days of usage history.
    
    Rationale: Linear regression and moving averages require minimum data
    points to establish trend reliability. With fewer than 5 observations:
    - Noise dominates signals (one day's variance skews slope significantly)
    - Weekday/weekend patterns not yet emerged (cafe has 7-day cycle)
    - Inventory management decisions would be unreliable
    
    At exactly 5 days, the ML predictor can still work but should be paired
    with rule-based alerts for safety. Below 5, rule-based alerts are safer.
    
    Args:
        usage_history_length (int): Number of days of usage data available
    
    Returns:
        bool: True if fallback rules should be used, False if ML is adequate
    
    Example:
        >>> should_use_fallback(3)
        True
        >>> should_use_fallback(7)
        False
    """
    return usage_history_length < 5


def rule_based_alert(
    item: Dict[str, Any],
    avg_daily_usage: float
) -> Dict[str, Any]:
    """
    Generate inventory alerts using simple threshold rules with no ML dependencies.
    
    This function implements heuristic thresholds for reorder, expiry, and
    threshold-breach scenarios. Designed as a fallback when usage history
    is too short for reliable trend prediction.
    
    Alert types:
    - "reorder": Current stock would deplete within 7 days at current usage rate
    - "expiry": Item expires within 5 days
    - "threshold": Current stock has fallen below the reorder threshold
    
    Business guarantees:
    - Days remaining calculation only attempted if consumption > 0
    - Expiry dates parsed strictly (YYYY-MM-DD format expected)
    - Multiple alerts can fire for a single item (e.g., both expiry and reorder)
    
    Args:
        item (Dict[str, Any]): Item metadata with keys:
            - item_id (str): Unique item identifier
            - name (str): Item name for messaging
            - current_stock (float): Quantity in inventory
            - expiry_date (str): YYYY-MM-DD formatted date string
            - reorder_threshold (float, optional): Min stock level before alert
        avg_daily_usage (float): Average daily consumption rate
    
    Returns:
        Dict[str, Any]: Alert result with keys:
            - has_alert (bool): True if any alert was raised
            - alerts (List[Dict]): List of alert dicts, each with:
                - type: "reorder", "expiry", or "threshold"
                - message: Human-readable alert text
            - source (str): Always "rule-based"
            - data_note (str): Explanation of why fallback was triggered
    
    Example:
        >>> item = {
        ...     'item_id': '001',
        ...     'name': 'Coffee Beans',
        ...     'current_stock': 2.0,
        ...     'expiry_date': '2026-03-22',
        ...     'reorder_threshold': 1.0
        ... }
        >>> rule_based_alert(item, 0.5)
        {
            'has_alert': True,
            'alerts': [
                {'type': 'reorder', 'message': 'Stock depleting: 4 days remaining...'},
                {'type': 'expiry', 'message': 'Item expires in 3 days'}
            ],
            'source': 'rule-based',
            'data_note': 'Insufficient usage history for AI prediction — rule-based alerts applied'
        }
    """
    alerts: List[Dict[str, str]] = []
    today = datetime.now().date()
    
    # Parse expiry date
    expiry_date = datetime.strptime(item['expiry_date'], '%Y-%m-%d').date()
    days_until_expiry = (expiry_date - today).days
    
    # Calculate days remaining at current usage rate
    days_remaining = None
    if avg_daily_usage > 0:
        days_remaining = item['current_stock'] / avg_daily_usage
    
    # Alert 1: Reorder (running low on stock)
    if days_remaining is not None and days_remaining < 7:
        alerts.append({
            'type': 'reorder',
            'message': f"Stock {item['name']}: {math.ceil(days_remaining)} days remaining at current usage rate"
        })
    
    # Alert 2: Expiry approaching
    if days_until_expiry < 5:
        alerts.append({
            'type': 'expiry',
            'message': f"Expiry alert {item['name']}: expires in {days_until_expiry} days"
        })
    
    # Alert 3: Below reorder threshold
    reorder_threshold = item.get('reorder_threshold', 0)
    if reorder_threshold > 0 and item['current_stock'] <= reorder_threshold:
        alerts.append({
            'type': 'threshold',
            'message': f"Minimum stock alert {item['name']}: current {item['current_stock']}, threshold {reorder_threshold}"
        })
    
    return {
        'has_alert': len(alerts) > 0,
        'alerts': alerts,
        'source': 'rule-based',
        'data_note': 'Insufficient usage history for AI prediction — rule-based alerts applied'
    }


def generate_rule_based_summary(
    item: Dict[str, Any],
    alert_result: Dict[str, Any],
    top_supplier: Optional[Dict[str, Any]] = None
) -> str:
    """
    Generate a plain-English summary of rule-based alerts for cafe staff.
    
    This is the fallback output shown to users when insufficient data prevents
    ML predictions. It provides clear, actionable guidance in 2-3 sentences
    with optional supplier information.
    
    Summary structure:
    1. Current status (alert type and severity)
    2. Context (lead time, supplier if available)
    3. Action (single clear recommendation)
    
    Args:
        item (Dict[str, Any]): Item metadata (same structure as rule_based_alert)
        alert_result (Dict[str, Any]): Output from rule_based_alert function
        top_supplier (Optional[Dict[str, Any]]): Supplier info with keys:
            - name: Supplier company name
            - lead_time: Delivery window description (e.g., "2-3 business days")
    
    Returns:
        str: Plain-text summary suitable for chat display
    
    Example:
        >>> summary = generate_rule_based_summary(
        ...     item,
        ...     alert_result,
        ...     top_supplier={'name': 'Blue Tokai', 'lead_time': '2-3 business days'}
        ... )
        >>> print(summary)
        'Coffee Beans is running low (4 days stock remaining). Blue Tokai can 
        deliver in 2-3 business days. Order now to avoid stockout.'
    """
    if not alert_result.get('has_alert', False):
        return f"{item['name']} levels are normal. No immediate action needed."
    
    # Summarize the primary alert
    alerts = alert_result.get('alerts', [])
    alert_types = [a['type'] for a in alerts]
    
    # Build status line
    status_parts = []
    if 'expiry' in alert_types:
        status_parts.append(f"{item['name']} expires soon")
    if 'reorder' in alert_types:
        status_parts.append(f"{item['name']} stock is low")
    if 'threshold' in alert_types:
        status_parts.append(f"{item['name']} is below minimum level")
    
    status_line = " and ".join(status_parts) + "." if status_parts else (
        f"{item['name']} needs manual review."
    )
    
    # Build supplier context
    supplier_line = ""
    if top_supplier:
        supplier_name = top_supplier.get('name') or top_supplier.get('supplier_name') or 'The supplier'
        lead_time = top_supplier.get('lead_time')
        if lead_time is None and top_supplier.get('lead_time_days') is not None:
            days = top_supplier['lead_time_days']
            lead_time = f"{days} day" if days == 1 else f"{days} days"

        if lead_time:
            supplier_line = f" {supplier_name} can deliver in {lead_time}."
        else:
            supplier_line = f" {supplier_name} is available for reorder."
    
    # Build recommendation
    recommendation = "Order now to prevent stockout."
    if 'expiry' in alert_types and 'reorder' not in alert_types:
        recommendation = "Use existing stock quickly before expiry."
    
    # Combine into 2-3 sentence summary
    summary = status_line + supplier_line + " " + recommendation
    return summary.strip()
