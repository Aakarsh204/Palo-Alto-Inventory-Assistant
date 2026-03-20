"""
Inventory analysis agent using retrieval-augmented generation (RAG) pattern.

This module implements a lightweight RAG approach: classical AI/statistical
methods compute factual numbers (predictions, clusters, forecasts), and the
LLM provides narrative reasoning and operational insights based on those
authoritative numbers. The LLM output is guidance only—all decisions should
reference the underlying metrics, not LLM assertions alone.

Temperature choice (0.3): Low temperature favors deterministic, factual
responses over creative speculation. Cafe inventory decisions require clarity;
0.3 prevents hallucination while still allowing coherent summarization.

LLM role: Synthesizes patterns across multiple data streams and suggests
operational improvements. It does NOT generate novel predictions or override
the classical models' scores.
"""

import os
import time
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Optional
import certifi
import httpx
from groq import Groq
from dotenv import load_dotenv

from .fallback import generate_rule_based_summary

# Load environment variables at module level
load_dotenv()

# On Windows/corporate networks, this package merges OS certs into certifi.
try:
    import certifi_win32  # type: ignore  # noqa: F401
except Exception:
    certifi_win32 = None


# In-memory cache keyed by <date>::<item_id_or_name>
_INSIGHTS_CACHE: Dict[str, Dict[str, Any]] = {}


def _get_groq_client() -> Groq:
    """Create a Groq client with explicit key validation."""
    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is missing. Add it to your .env file.")

    skip_ssl_verify = os.environ.get("GROQ_SKIP_SSL_VERIFY", "false").lower() == "true"
    custom_ca_bundle = os.environ.get("GROQ_CA_BUNDLE", "").strip()

    verify: bool | str
    if skip_ssl_verify:
        verify = False
    elif custom_ca_bundle:
        verify = custom_ca_bundle
    else:
        verify = certifi.where()

    http_client = httpx.Client(
        timeout=30.0,
        verify=verify,
        trust_env=True
    )

    return Groq(api_key=api_key, http_client=http_client)


def build_agent_context(
    item: Dict[str, Any],
    usage_reconciliation_df: pd.DataFrame,
    cluster_result: Dict[str, Any],
    scraped_suppliers: List[Dict[str, Any]],
    recipe_forecast: Dict[str, Any],
    stockout_prediction: Dict[str, Any]
) -> str:
    """
    Build a structured multi-section context prompt for the Groq LLM.
    
    Assembles data from usage analysis, clustering, supplier matching, and
    demand forecasting into a readable prompt that guides the model to
    provide coherent inventory insights.
    
    Args:
        item (Dict[str, Any]): Inventory item with keys: name,
                               current_stock, unit, expiry_date, reorder_threshold
        usage_reconciliation_df (pd.DataFrame): Reconciled usage with columns:
                                                date, usage, estimated_usage,
                                                discrepancy, variance_flagged
                                                (uses last 7 rows)
        cluster_result (Dict[str, Any]): Cluster assignment with keys:
                                         cluster_label, stock_days_remaining
        scraped_suppliers (List[Dict[str, Any]]): Top 2 matched suppliers from
                                                  procurement, each with:
                                                  supplier_name, product_name,
                                                  price, lead_time_days,
                                                  sustainability_score, tags
        recipe_forecast (Dict[str, Any]): Demand estimate with keys:
                                         avg_daily_demand, projected_total,
                                         contributing_recipes
        stockout_prediction (Dict[str, Any]): Prediction comparison with keys:
                                             stock_days, order_days, confidence
    
    Returns:
        str: Structured prompt ready for Groq API
    
    Example:
        >>> context = build_agent_context(item, usage_df, cluster, suppliers, ...)
        >>> context.startswith("=== INVENTORY ITEM ===")
        True
    """
    # Section 1: Inventory Item Overview
    stock_days = stockout_prediction.get('stock_days')
    order_days = stockout_prediction.get('order_days')
    confidence = stockout_prediction.get('confidence')
    cluster_label = cluster_result.get('cluster_label', 'Unknown')
    stock_days_remaining = cluster_result.get('stock_days_remaining', 0)
    
    section1 = f"""=== INVENTORY ITEM ===
Name: {item.get('name', 'Unknown')}
Current Stock: {item.get('current_stock', 0)} {item.get('unit', 'units')}
Predicted Stockout (stock method): {stock_days} days (confidence: {confidence})
Predicted Stockout (order method): {order_days} days
Cluster Classification: {cluster_label}
Stock Coverage: {stock_days_remaining} days at current consumption
"""
    
    # Section 2: Usage History (last 7 days)
    section2 = "=== USAGE HISTORY (Last 7 Days) ===\n"
    section2 += "Date | Stock-Derived | Order-Derived | Discrepancy | Flagged\n"
    section2 += "-" * 65 + "\n"
    
    if not usage_reconciliation_df.empty:
        last_7 = usage_reconciliation_df.tail(7)
        for idx, row in last_7.iterrows():
            date_str = str(row['date'])[:10]
            usage = row['usage']
            est_usage = row['estimated_usage']
            discrepancy = row['discrepancy']
            flagged = "✓" if row['variance_flagged'] else ""
            section2 += f"{date_str} | {usage:>13.2f} | {est_usage:>13.2f} | {discrepancy:>11.2f} | {flagged}\n"
    else:
        section2 += "(No usage history available)\n"
    
    # Section 3: Recipe Demand Forecast
    avg_demand = recipe_forecast.get('avg_daily_demand', 0)
    projected = recipe_forecast.get('projected_total', 0)
    contributing = recipe_forecast.get('contributing_recipes', [])
    
    section3 = f"""
=== RECIPE DEMAND FORECAST (7-Day Projection) ===
Average Daily Demand: {avg_demand} (across {len(contributing)} recipes)
Projected Total Consumption: {projected}
Contributing Recipes: {', '.join(contributing) if contributing else 'None'}
"""
    
    # Section 4: Available Suppliers
    section4 = "=== AVAILABLE SUPPLIERS ===\n"
    if scraped_suppliers:
        for i, supplier in enumerate(scraped_suppliers[:2], 1):
            section4 += f"\nOption {i}: {supplier.get('supplier_name', 'Unknown')}\n"
            section4 += f"  Product: {supplier.get('product_name', 'Unknown')}\n"
            section4 += f"  Price: {supplier.get('price', 'N/A')}\n"
            section4 += f"  Lead Time: {supplier.get('lead_time_days', 'N/A')} days\n"
            section4 += f"  Sustainability Score: {supplier.get('sustainability_score', 'N/A')}/100\n"
            tags = supplier.get('tags', [])
            section4 += f"  Tags: {', '.join(tags) if tags else 'None'}\n"
    else:
        section4 += "(No suppliers matched current constraints)\n"
    
    # Section 5: Task for the Model
    section5 = """
=== TASK ===
Based on the data above, provide inventory optimization insights:

1. **Usage Patterns**: Identify 2-3 specific observations about consumption
   (e.g., "Order-derived and stock-derived usage match closely: minimal
   untracked movement" or "Large discrepancies on Feb 24-25: likely spillage
   or recount adjustment").

2. **Flagged Discrepancies**: Explain any flagged variance events (where
   abs(discrepancy) > 20% of actual usage). Possible causes: spillage,
   waste, theft, data entry errors, or delivery count discrepancies.

3. **Supplier Recommendation**: Choose the best option (1 or 2) with
   specific reasoning (lead time + sustainability + cost trade-offs).
   Include the why_recommended score if available.

4. **Operational Insight**: Suggest one actionable change to reduce waste,
   prevent stockouts, or improve forecasting (e.g., "daily rather than
   weekly counts" or "smaller batch sizes to prevent expiry loss").

Keep response under 200 words. Be specific with numbers and actual data.
Never invent data that isn't in the provided context.
"""
    
    return section1 + section2 + section3 + section4 + section5


def get_agent_insights(
    context_prompt: str,
    item_name: str,
    item: Optional[Dict[str, Any]] = None,
    top_supplier: Optional[Dict[str, Any]] = None,
    cache_key: Optional[str] = None,
    cache_date: Optional[str] = None
) -> Dict[str, Any]:
    """
    Call Groq LLM to generate inventory insights from context.
    
    Submits a structured prompt to the Groq Llama 3 model and returns
    insights. On any failure (API error, network issue, etc.), gracefully
    falls back to rule-based alerts.
    
    Args:
        context_prompt (str): Structured prompt from build_agent_context()
        item_name (str): Item name for error logging
        item (Optional[Dict[str, Any]]): Item data for fallback (passed to
                                         generate_rule_based_summary)
        top_supplier (Optional[Dict[str, Any]]): Top supplier for fallback
    
    Returns:
        Dict[str, Any]: Insights result with keys:
            - insights (str): LLM response or fallback summary
            - source (str): "ai-agent" or "rule-based-fallback"
            - model (str or None): Model name if AI, None if fallback
            - fallback_used (bool): True if fallback was triggered
            - error (str, optional): Error message if fallback triggered
    
    Example:
        >>> result = get_agent_insights(context, "Coffee Beans")
        >>> result['source']
        'ai-agent'
    """
    system_prompt = """You are a cafe inventory optimization specialist.
Your role is to synthesize data from multiple sources and provide clear, 
actionable insights. ALWAYS respond in strict Markdown format with the 
exact structure below. Do NOT deviate from this format.

## REQUIRED MARKDOWN FORMAT:

### 1. Usage Patterns
- Observation 1 (2-3 words): Description with numbers
- Observation 2 (2-3 words): Description with numbers

### 2. Flagged Discrepancies  
- Discrepancy type: Explanation and likely cause
- (If none flagged, write: "No significant discrepancies detected")

### 3. Supplier Recommendation
- **Recommended:** Supplier name | Reason (lead time, cost, sustainability)
- **Why:** Specific trade-off analysis

### 4. Operational Insight
- **Action:** One specific, implementable change to reduce waste/prevent stockouts

IMPORTANT: 
- Always use ### for section headers
- Always use - for bullet points
- Always use **bold** for key terms
- Keep total under 200 words
- Never invent data not in context
- Always cite actual numbers and dates from data provided"""
    
    try:
        # Cache key is item/day scoped to avoid repeated calls in the same day.
        resolved_date = cache_date or datetime.now().date().isoformat()
        resolved_item_key = cache_key or item_name
        composed_cache_key = f"{resolved_date}::{resolved_item_key}"

        cached = _INSIGHTS_CACHE.get(composed_cache_key)
        if cached:
            return {
                **cached,
                'cache_hit': True,
                'source': 'ai-agent-cache'
            }

        client = _get_groq_client()

        model_candidates = [
            "llama-3.1-8b-instant",
            "llama3-8b-8192"
        ]
        last_error = None

        for model_name in model_candidates:
            for attempt in range(1, 4):
                try:
                    completion = client.chat.completions.create(
                        model=model_name,
                        messages=[
                            {
                                "role": "system",
                                "content": system_prompt
                            },
                            {
                                "role": "user",
                                "content": context_prompt
                            }
                        ],
                        max_tokens=350,
                        temperature=0.3
                    )

                    response_text = completion.choices[0].message.content if completion.choices else ""

                    result = {
                        'insights': response_text,
                        'source': 'ai-agent',
                        'model': model_name,
                        'fallback_used': False,
                        'cache_hit': False
                    }

                    _INSIGHTS_CACHE[composed_cache_key] = {
                        'insights': response_text,
                        'source': 'ai-agent',
                        'model': model_name,
                        'fallback_used': False
                    }

                    return result
                except Exception as model_error:
                    last_error = model_error
                    error_name = type(model_error).__name__
                    is_connection_error = (
                        "connection" in str(model_error).lower()
                        or "connect" in error_name.lower()
                    )

                    if not is_connection_error:
                        break

                    if attempt < 3:
                        time.sleep(0.5 * attempt)

        if last_error is not None:
            raise last_error

        raise RuntimeError("No Groq model candidates were available.")
    
    except Exception as e:
        cause = getattr(e, '__cause__', None)
        detail = f"{type(e).__name__}: {e}"
        if cause:
            detail += f" | cause={type(cause).__name__}: {cause}"

        # Fallback to rule-based alerts
        fallback_summary = ""
        if item:
            alert_result = {
                'has_alert': True,
                'alerts': [
                    {
                        'type': 'info',
                        'message': 'AI service temporarily unavailable'
                    }
                ]
            }
            fallback_summary = generate_rule_based_summary(
                item,
                alert_result,
                top_supplier
            )
        else:
            fallback_summary = "Inventory analysis unavailable. Please review stock levels manually."
        
        return {
            'insights': fallback_summary,
            'source': 'rule-based-fallback',
            'model': None,
            'fallback_used': True,
            'error': detail
        }
