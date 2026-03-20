"""
Supplier matching module for cafe procurement decisions.

In production, this module would integrate with supplier APIs (Alibaba,
TradeKey, local B2B platforms) to fetch real-time pricing and availability.
Currently, it processes statically scraped supplier pages for demo/testing.

The scoring algorithm balances three factors:
- Sustainability alignment (40%): lower environmental impact
- Preference alignment (40%): supplier tags match cafe values
- Lead time suitability (20%): delivery timing relative to urgency
"""

from typing import List, Dict, Optional, Any
import re


def match_suppliers(
    item_name: str,
    urgency_days: int,
    cafe_preferences: Dict[str, List[str]],
    scraped_products: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Match inventory items to suppliers based on urgency, sustainability, and preferences.
    
    Scoring algorithm (weighted average):
    - 40% Sustainability: Higher score_0-100 = better environmental profile
    - 40% Preference alignment: Tag overlap vs. cafe's preferred_tags list
    - 20% Lead time suitability: Fast delivery scores higher (capped at urgency window)
    
    Lead time scoring:
    - If supplier delivers within urgency window: 100 points
    - If delivery slow: decreases 20 points per day beyond urgency
    - Floor: 0 points (too late to help)
    
    Business guarantees:
    - Returns top 2 suppliers by match_score (or fewer if fewer exist)
    - Each result includes why_recommended summary
    - Works without inventory grouping metadata
    
    Args:
        item_name (str): Item name from inventory (currently informational)
        urgency_days (int): Days remaining until stockout (from predictor)
        cafe_preferences (Dict[str, List[str]]): Cafe preferences with key
            'preferred_tags' containing list of strings like "local", "organic"
        scraped_products (List[Dict[str, Any]]): Supplier products from scraper,
            each with keys: supplier_name, product_name, price,
            min_order, sustainability_score, tags (list), lead_time_days, description
    
    Returns:
        List[Dict[str, Any]]: Top 2 matching suppliers, each with:
            - All original product fields
            - match_score: Weighted score 0-100
            - why_recommended: Single-sentence explanation of match
    
    Example:
        >>> prefs = {'preferred_tags': ['local', 'organic']}
        >>> products = [
        ...     {
        ...         'supplier_name': 'Blue Tokai',
        ...         'product_name': 'Arabica',
        ...         'sustainability_score': 94,
        ...         'tags': ['local', 'direct-trade'],
        ...         'lead_time_days': 2
        ...     }
        ... ]
        >>> matches = match_suppliers('Coffee Beans', 5, prefs, products)
        >>> matches[0]['match_score']
        87.0
    """
    preferred_tags = cafe_preferences.get('preferred_tags', [])

    # Group-agnostic matching: score all available products.
    candidates = list(scraped_products)

    if not candidates:
        return []
    
    # Score each candidate
    scored: List[Dict[str, Any]] = []
    
    for product in candidates:
        # Component 1: Sustainability (40%)
        sustainability_score = float(product.get('sustainability_score', 0))
        sustainability_component = sustainability_score * 0.40
        
        # Component 2: Tag overlap (40%)
        product_tags = product.get('tags', [])
        if preferred_tags:
            matched_tags = len(set(product_tags) & set(preferred_tags))
            tag_alignment = (matched_tags / len(preferred_tags)) * 100
        else:
            tag_alignment = 50.0  # Neutral if no preferences
        tag_component = tag_alignment * 0.40
        
        # Component 3: Lead time suitability (20%)
        lead_time = product.get('lead_time_days', 999)
        if lead_time <= urgency_days:
            lead_time_score = 100.0
        else:
            days_overdue = lead_time - urgency_days
            lead_time_score = max(0, 100 - (days_overdue * 20))
        lead_time_component = lead_time_score * 0.20
        
        # Total match score
        match_score = sustainability_component + tag_component + lead_time_component
        
        # Determine why_recommended
        scores = {
            'sustainability': sustainability_component,
            'tags': tag_component,
            'lead_time': lead_time_component
        }
        top_factor = max(scores, key=scores.get)
        
        if top_factor == 'sustainability':
            why = f"Highest sustainability score ({sustainability_score}/100)."
        elif top_factor == 'tags':
            why = f"Strong alignment with cafe values ({matched_tags}/{len(preferred_tags)} preferred tags)."
        else:
            why = f"Fastest delivery ({lead_time} days, within urgency window)."
        
        scored.append({
            **product,
            'match_score': round(match_score, 1),
            'why_recommended': why
        })
    
    # Sort by score descending, return top 2
    scored.sort(key=lambda x: x['match_score'], reverse=True)
    return scored[:2]


def get_default_preferences() -> Dict[str, List[str]]:
    """
    Return cafe's default supplier preferences when none configured.
    
    Default values reflect typical priorities for small eco-conscious
    Indian cafe: local sourcing, direct relationships, sustainable packaging,
    and environmental responsibility.
    
    Returns:
        Dict[str, List[str]]: Preferences dict with key 'preferred_tags'
    
    Example:
        >>> prefs = get_default_preferences()
        >>> 'local' in prefs['preferred_tags']
        True
    """
    return {
        'preferred_tags': [
            'local',
            'direct-trade',
            'compostable-packaging',
            'low-carbon',
            'organic'
        ]
    }


def format_price_display(price_string: str) -> Optional[Dict[str, Any]]:
    """
    Parse wholesale price strings into structured data for display.
    
    Handles Indian rupee prices with unit denominations:
    - "₹850/kg" → amount=850, currency="₹", per_unit="kg"
    - "₹72/L" → amount=72, currency="₹", per_unit="L"
    - "₹4.50/unit" → amount=4.5, currency="₹", per_unit="unit"
    
    Graceful failure: Returns None for unparseable strings, allowing
    upstream code to fall back to display original price string.
    
    Args:
        price_string (str): Price in format "₹[amount]/[unit]"
    
    Returns:
        Optional[Dict[str, Any]]: Parsed price with keys:
            - amount (float): Numeric price value
            - currency (str): Currency symbol or code
            - per_unit (str): Unit of measurement (kg, L, units, etc.)
            Or None if string cannot be parsed
    
    Example:
        >>> format_price_display("₹850/kg")
        {'amount': 850.0, 'currency': '₹', 'per_unit': 'kg'}
        
        >>> format_price_display("Invalid")
        None
    """
    try:
        # Match pattern: currency symbol, amount (int or float), /, unit
        pattern = r'([₹$€])\s*([0-9.]+)\s*/\s*(\w+)'
        match = re.match(pattern, price_string.strip())
        
        if not match:
            return None
        
        currency, amount_str, per_unit = match.groups()
        amount = float(amount_str)
        
        return {
            'amount': amount,
            'currency': currency,
            'per_unit': per_unit
        }
    except (ValueError, AttributeError):
        # Malformed string, return None
        return None
