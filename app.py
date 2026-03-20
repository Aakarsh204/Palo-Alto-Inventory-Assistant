"""
Green-Tech Inventory Assistant - Main Flask Application

A comprehensive inventory management system for small Indian cafes,
combining classical statistical models with LLM-powered insights.
Uses clustering, demand forecasting, and rule-based fallbacks when
data is insufficient for reliable AI predictions.
"""

from flask import Flask, render_template, request, redirect, url_for, jsonify, flash
import pandas as pd
import json
import os
import copy
import math
import tempfile
from datetime import datetime, timedelta, date
from typing import Optional
from dotenv import load_dotenv
import markdown2

# Internal AI/ML module imports
from src.ai.usage_calculator import (
    derive_usage_from_events,
    derive_usage_from_orders,
    reconcile_usage
)
from src.ai.predictor import predict_stockout, compare_predictions
from src.ai.clusterer import build_feature_matrix, cluster_items
from src.ai.fallback import should_use_fallback, rule_based_alert
from src.ai.procurement import match_suppliers, get_default_preferences
from src.ai.recipe_estimator import (
    estimate_ingredient_demand,
    find_bottleneck_ingredient,
    get_recipe_ingredient_summary
)
from src.ai.agent import build_agent_context, get_agent_insights
from src.scraper.scrape import scrape_all_suppliers


# ============================================================================
# STARTUP: Module-level initialization
# ============================================================================

load_dotenv()

# Initialize Flask app with correct template and static folders
app = Flask(
    __name__,
    template_folder='src/templates',
    static_folder='src/static'
)
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-key-change-in-production")

# Register Jinja2 markdown filter for rendering markdown in templates
@app.template_filter('markdown')
def markdown_filter(text: str) -> str:
    """Convert markdown string to HTML."""
    if not text:
        return ""
    # Convert markdown to HTML, allowing HTML passthrough
    html = markdown2.markdown(text, extras=['fenced-code-blocks', 'tables', 'breaks'])
    return html

# Read configuration from environment
DEVELOPER_MODE = os.environ.get("DEVELOPER_MODE", "False").lower() == "true"
DATE_OFFSET_DAYS = int(os.environ.get("DATE_OFFSET_DAYS", "0"))

# Disk-backed storage map (single source of truth)
DATA_FILES = {
    'inventory': 'src/data/inventory.csv',
    'events': 'src/data/stock_events.csv',
    'orders': 'src/data/order_history.csv',
    'recipes': 'src/data/recipes.csv',
    'ingredients': 'src/data/recipe_ingredients.csv'
}


# ============================================================================
# CONTEXT PROCESSOR: Inject developer_mode and simulated_date into all templates
# ============================================================================

@app.context_processor
def inject_template_context():
    """
    Make developer_mode and simulated_date available in all Jinja2 templates.
    
    Enables base.html and child templates to display developer mode banner
    and use simulated date in any context without explicit passing.
    """
    return {
        'developer_mode': DEVELOPER_MODE,
        'simulated_date': get_today()
    }

inventory_df = pd.DataFrame()
events_df = pd.DataFrame()
order_df = pd.DataFrame()
recipe_df = pd.DataFrame()
ingredients_df = pd.DataFrame()

# Runtime state/caches to reduce repeated heavy recomputation per request.
DATA_FILE_MTIMES: dict[str, float | None] = {}
APP_STATE_VERSION = 0
CLUSTER_CACHE: dict[str, object] = {
    'version': -1,
    'date': None,
    'df': pd.DataFrame()
}
ITEM_INSIGHTS_CACHE: dict[tuple[int, str, str], dict] = {}


def normalize_identifier_column(
    df: pd.DataFrame,
    column_name: str,
    width: int | None = None
) -> pd.DataFrame:
    """
    Normalize identifier columns loaded from CSV.

    Pandas can infer numeric-looking IDs like 001 as integers and strip
    leading zeros. This helper restores a consistent string format so route
    params, form values, and CSV-backed IDs compare correctly.
    """
    if df.empty or column_name not in df.columns:
        return df

    normalized = df.copy()
    normalized[column_name] = normalized[column_name].astype(str).str.strip()

    if width is not None:
        normalized[column_name] = normalized[column_name].str.zfill(width)

    return normalized


def read_csv_safe(file_path: str) -> pd.DataFrame:
    """Read a CSV file safely; return empty DataFrame if missing."""
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Warning: {file_path} not found")
        return pd.DataFrame()


def refresh_dataframes() -> None:
    """
    Reload all application DataFrames from disk.

    This ensures every request sees persisted data written by form submissions
    instead of relying on stale module-level snapshots.
    """
    global inventory_df, events_df, order_df, recipe_df, ingredients_df

    inventory_df = read_csv_safe(DATA_FILES['inventory'])
    events_df = read_csv_safe(DATA_FILES['events'])
    order_df = read_csv_safe(DATA_FILES['orders'])
    recipe_df = read_csv_safe(DATA_FILES['recipes'])
    ingredients_df = read_csv_safe(DATA_FILES['ingredients'])

    inventory_df = normalize_identifier_column(inventory_df, 'item_id', width=3)
    events_df = normalize_identifier_column(events_df, 'item_id', width=3)
    ingredients_df = normalize_identifier_column(ingredients_df, 'item_id', width=3)
    recipe_df = normalize_identifier_column(recipe_df, 'recipe_id')
    ingredients_df = normalize_identifier_column(ingredients_df, 'recipe_id')
    order_df = normalize_identifier_column(order_df, 'recipe_id')


def invalidate_runtime_caches() -> None:
    """Clear derived caches after any data change."""
    global CLUSTER_CACHE, ITEM_INSIGHTS_CACHE
    CLUSTER_CACHE = {
        'version': -1,
        'date': None,
        'df': pd.DataFrame()
    }
    ITEM_INSIGHTS_CACHE.clear()


def _get_file_mtime(file_path: str) -> float | None:
    """Return file mtime if present, else None."""
    try:
        return os.path.getmtime(file_path)
    except FileNotFoundError:
        return None


def refresh_dataframes_if_needed(force: bool = False) -> None:
    """Reload CSVs only when source files changed on disk."""
    global APP_STATE_VERSION

    current_mtimes = {
        key: _get_file_mtime(path)
        for key, path in DATA_FILES.items()
    }

    if force or current_mtimes != DATA_FILE_MTIMES:
        refresh_dataframes()
        DATA_FILE_MTIMES.clear()
        DATA_FILE_MTIMES.update(current_mtimes)
        APP_STATE_VERSION += 1
        invalidate_runtime_caches()


def persist_dataframe(df: pd.DataFrame, data_key: str) -> None:
    """Persist a DataFrame to CSV atomically, then invalidate derived caches."""
    global APP_STATE_VERSION

    file_path = DATA_FILES[data_key]
    target_dir = os.path.dirname(file_path) or '.'
    fd, tmp_path = tempfile.mkstemp(
        prefix=f".{data_key}_",
        suffix='.tmp',
        dir=target_dir
    )
    os.close(fd)

    try:
        df.to_csv(tmp_path, index=False)
        # Atomic replace prevents partial files during crashes/interruption.
        os.replace(tmp_path, file_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    # Keep mtime/version in sync so request-time reload checks stay cheap.
    DATA_FILE_MTIMES[data_key] = _get_file_mtime(file_path)
    APP_STATE_VERSION += 1
    invalidate_runtime_caches()


def get_food_inventory_df() -> pd.DataFrame:
    """Return current inventory rows."""
    if inventory_df.empty:
        return pd.DataFrame()
    return inventory_df.copy()


def normalize_item_id_input(item_id: str | None) -> str:
    """Normalize incoming item IDs to match zero-padded CSV identifiers."""
    if item_id is None:
        return ''
    return str(item_id).strip().zfill(3)


def update_inventory_stock(item_id: str, event_type: str, quantity: float) -> None:
    """Apply stock change to inventory and persist the updated inventory CSV."""
    global inventory_df

    if inventory_df.empty:
        return

    match_mask = inventory_df['item_id'] == item_id
    if not match_mask.any():
        return

    updated_inventory_df = inventory_df.copy()

    current_stock = pd.to_numeric(
        updated_inventory_df.loc[match_mask, 'current_stock'],
        errors='coerce'
    ).fillna(0.0).iloc[0]

    if event_type == 'received':
        updated_stock = current_stock + quantity
    else:
        updated_stock = quantity

    updated_inventory_df.loc[match_mask, 'current_stock'] = round(updated_stock, 3)
    persist_dataframe(updated_inventory_df, 'inventory')
    inventory_df = updated_inventory_df


def apply_order_to_inventory(recipe_id: str, quantity_sold: int) -> None:
    """Reduce inventory stock based on recipe ingredients consumed by orders."""
    global inventory_df

    if inventory_df.empty or ingredients_df.empty or quantity_sold <= 0:
        return

    recipe_ingredients = ingredients_df[
        ingredients_df['recipe_id'] == recipe_id
    ].copy()

    if recipe_ingredients.empty:
        return

    updated_inventory_df = inventory_df.copy()

    recipe_ingredients['quantity_per_unit'] = pd.to_numeric(
        recipe_ingredients['quantity_per_unit'],
        errors='coerce'
    ).fillna(0.0)

    for _, ingredient in recipe_ingredients.iterrows():
        item_id = str(ingredient.get('item_id', '')).strip()
        consumption = float(ingredient['quantity_per_unit']) * quantity_sold
        if not item_id or consumption <= 0:
            continue

        match_mask = updated_inventory_df['item_id'] == item_id
        if not match_mask.any():
            continue

        current_stock = pd.to_numeric(
            updated_inventory_df.loc[match_mask, 'current_stock'],
            errors='coerce'
        ).fillna(0.0).iloc[0]
        updated_stock = max(0.0, current_stock - consumption)
        updated_inventory_df.loc[match_mask, 'current_stock'] = round(updated_stock, 3)

    persist_dataframe(updated_inventory_df, 'inventory')
    inventory_df = updated_inventory_df


# Initial load from persistent storage
refresh_dataframes_if_needed(force=True)

# Keep runtime data synchronized with on-disk storage on each request
@app.before_request
def sync_runtime_data() -> None:
    refresh_dataframes_if_needed()

# Scrape supplier catalogs (static mock pages in demo, would be live APIs in production)
scraped_suppliers = scrape_all_suppliers()

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_today() -> date:
    """
    Return today's date, offset by DATE_OFFSET_DAYS for developer testing.
    
    Allows time-travel simulation for testing seasonal patterns and
    expiry scenarios without waiting for actual calendar dates.
    Developer mode: Set DATE_OFFSET_DAYS in .env to simulate future dates.
    """
    return datetime.now().date() + timedelta(days=DATE_OFFSET_DAYS)


def get_clustered_inventory() -> pd.DataFrame:
    """
    Build a consistent clustered inventory view for dashboard and insights.

    Centralizing clustering avoids mismatches where one screen uses live
    cluster labels and another uses placeholders or a separate calculation.
    """
    cache_date = get_today().isoformat()
    if (
        CLUSTER_CACHE['version'] == APP_STATE_VERSION
        and CLUSTER_CACHE['date'] == cache_date
        and isinstance(CLUSTER_CACHE['df'], pd.DataFrame)
        and not CLUSTER_CACHE['df'].empty
    ):
        return CLUSTER_CACHE['df'].copy()

    inventory_view = get_food_inventory_df()
    if inventory_view.empty:
        return pd.DataFrame()

    feature_df = build_feature_matrix(
        inventory_view,
        events_df,
        reference_date=get_today()
    )
    if feature_df.empty:
        return pd.DataFrame()

    clustered_df, _ = cluster_items(feature_df, n_clusters=4)
    CLUSTER_CACHE['version'] = APP_STATE_VERSION
    CLUSTER_CACHE['date'] = cache_date
    CLUSTER_CACHE['df'] = clustered_df.copy()
    return clustered_df


# ============================================================================
# CONTEXT PROCESSOR: Inject developer_mode and simulated_date into all templates
# ============================================================================

@app.context_processor
def inject_template_context():
    """
    Make developer_mode and simulated_date available in all Jinja2 templates.
    
    Enables base.html and child templates to display developer mode banner
    and use simulated date in any context without explicit passing.
    """
    return {
        'developer_mode': DEVELOPER_MODE,
        'simulated_date': get_today()
    }


def get_item_insights(
    item_id: str,
    clustered_df: pd.DataFrame | None = None,
    ai_budget: Optional[dict] = None
) -> dict:
    """
    Run complete analysis pipeline for a single inventory item.
    
    Pipeline:
    1. Derive usage from stock events and recipe orders
    2. Reconcile discrepancies (spillage/waste detection)
    3. Predict stockout using both methods, compare confidence
    4. Call LLM agent for insights with fallback to rules
    5. Match to best suppliers
    
    Returns:
        dict with keys: item_dict, usage_df, stockout_prediction,
                       insights_result, matched_suppliers
    """
    cache_key = (APP_STATE_VERSION, get_today().isoformat(), item_id)
    if cache_key in ITEM_INSIGHTS_CACHE:
        return copy.deepcopy(ITEM_INSIGHTS_CACHE[cache_key])

    if inventory_df.empty:
        return {'error': 'No inventory data'}
    
    item = inventory_df[inventory_df['item_id'] == item_id]
    if item.empty:
        return {'error': 'Item not found'}
    
    item_dict = item.iloc[0].to_dict()

    if clustered_df is None or clustered_df.empty:
        clustered_df = get_clustered_inventory()

    cluster_row = pd.DataFrame()
    if not clustered_df.empty:
        cluster_row = clustered_df[clustered_df['item_id'] == item_id]

    cluster_result = {
        'cluster_label': cluster_row.iloc[0]['cluster_label'] if not cluster_row.empty else 'Unknown',
        'cluster_color': cluster_row.iloc[0]['cluster_color'] if not cluster_row.empty else '#6b7280',
        'stock_days_remaining': cluster_row.iloc[0]['stock_days_remaining'] if not cluster_row.empty else None,
        'days_until_expiry': cluster_row.iloc[0]['days_until_expiry'] if not cluster_row.empty else None,
        'avg_daily_usage': cluster_row.iloc[0]['avg_daily_usage'] if not cluster_row.empty else None,
    }
    
    # Derive usage from both sources
    stock_usage_df = derive_usage_from_events(item_id, events_df)
    order_usage_df = derive_usage_from_orders(item_id, order_df, ingredients_df)
    
    # Reconcile for anomalies (spillage/waste detection)
    reconciliation_df = reconcile_usage(stock_usage_df, order_usage_df)
    
    # Predict stockout using both methods
    stock_prediction = None
    order_prediction = None
    
    if not stock_usage_df.empty:
        usage_list = stock_usage_df['usage'].tolist()
        stock_prediction = predict_stockout(usage_list, item_dict['current_stock'])
    
    if not order_usage_df.empty:
        # Derive usage from orders for prediction
        orders_for_item = order_df[order_df['recipe_id'].isin(
            ingredients_df[ingredients_df['item_id'] == item_id]['recipe_id'].unique()
        )]
        if not orders_for_item.empty:
            order_usage_list = order_usage_df['estimated_usage'].tolist()
            order_prediction = predict_stockout(order_usage_list, item_dict['current_stock'])
    
    # Compare predictions (confidence assessment)
    stockout_prediction = compare_predictions(stock_prediction, order_prediction)
    recipe_forecast = estimate_ingredient_demand(item_id, order_df, ingredients_df)
    
    # Decide: fallback or AI?
    fallback_needed = should_use_fallback(len(stock_usage_df))
    ai_budget_exhausted = ai_budget is not None and ai_budget.get('remaining_calls', 0) <= 0
    
    if fallback_needed or ai_budget_exhausted:
        # Use rule-based alerts
        avg_usage = stock_usage_df['usage'].mean() if not stock_usage_df.empty else 0
        alerts = rule_based_alert(item_dict, avg_usage)
        from src.ai.fallback import generate_rule_based_summary
        summary = generate_rule_based_summary(
            item_dict,
            alerts,
            scraped_suppliers[0] if scraped_suppliers else None
        )
        insights_result = {
            'insights': summary,
            'source': 'rule-based',
            'model': None,
            'fallback_used': True,
            'budget_limited': ai_budget_exhausted
        }
    else:
        # Use LLM agent with full context
        # Match suppliers for this item without relying on item group metadata
        cafe_prefs = get_default_preferences()
        matched = match_suppliers(
            item_dict.get('name', ''),
            stockout_prediction.get('stock_days', 7) or 7,
            cafe_prefs,
            scraped_suppliers
        )
        
        # Build agent context
        context = build_agent_context(
            item_dict,
            reconciliation_df if not reconciliation_df.empty else pd.DataFrame(),
            cluster_result,
            matched,
            recipe_forecast,
            stockout_prediction
        )
        
        # Get LLM insights with fallback
        insights_result = get_agent_insights(
            context,
            item_dict.get('name', 'Unknown'),
            item_dict,
            matched[0] if matched else None,
            cache_key=item_id,
            cache_date=get_today().isoformat()
        )

        # Decrement only when this request performed a live AI call.
        if ai_budget is not None and not insights_result.get('cache_hit', False):
            ai_budget['remaining_calls'] = max(0, ai_budget.get('remaining_calls', 0) - 1)
    
    # Match suppliers
    cafe_prefs = get_default_preferences()
    matched_suppliers = match_suppliers(
        item_dict.get('name', ''),
        stockout_prediction.get('stock_days', 7) or 7,
        cafe_prefs,
        scraped_suppliers
    )

    reconciliation_records = []
    if not reconciliation_df.empty:
        reconciliation_copy = reconciliation_df.tail(7).copy()
        reconciliation_copy['date'] = pd.to_datetime(reconciliation_copy['date']).dt.strftime('%Y-%m-%d')
        reconciliation_records = reconciliation_copy.to_dict('records')
    
    result = {
        'item': item_dict,
        'cluster': cluster_result,
        'prediction': stockout_prediction,
        'reconciliation': reconciliation_records,
        'agent_result': insights_result,
        'recipe_forecast': recipe_forecast,
        'suppliers': matched_suppliers,
        'item_dict': item_dict,
        'usage_df': stock_usage_df,
        'reconciliation_df': reconciliation_df,
        'stockout_prediction': stockout_prediction,
        'insights_result': insights_result,
        'matched_suppliers': matched_suppliers
    }

    ITEM_INSIGHTS_CACHE[cache_key] = copy.deepcopy(result)
    return result


# ============================================================================
# ROUTES
# ============================================================================

@app.route('/')
def dashboard():
    """
    Dashboard: Overview of all inventory items with clustering and alerts.
    
    Pipeline:
    1. Build clustered inventory once
    2. Reuse clustered features per item (avoids duplicate usage recompute)
    3. Compute summary stats (cluster counts, projected waste)
    """
    inventory_view = get_food_inventory_df()
    if inventory_view.empty:
        return render_template('dashboard.html', items=[], summary={})

    clustered_df = get_clustered_inventory()
    
    items_enriched = []
    cluster_lookup: dict[str, dict] = {}
    if not clustered_df.empty:
        cluster_lookup = {
            str(row['item_id']): row.to_dict()
            for _, row in clustered_df.iterrows()
        }
    
    for idx, row in inventory_view.iterrows():
        item_id = row['item_id']
        item_dict = row.to_dict()
        
        cluster_row = cluster_lookup.get(str(item_id), {})
        stock_days_remaining = cluster_row.get('stock_days_remaining')
        avg_daily_usage = cluster_row.get('avg_daily_usage')

        # Approximate dashboard prediction from already-computed clustered features.
        stockout_days = None
        if (
            stock_days_remaining is not None
            and pd.notna(stock_days_remaining)
            and float(stock_days_remaining) < 999
            and avg_daily_usage is not None
            and pd.notna(avg_daily_usage)
            and float(avg_daily_usage) > 0
        ):
            stockout_days = int(min(60, max(1, math.ceil(float(stock_days_remaining)))))

        item_dict['stockout_days'] = stockout_days
        item_dict['has_alert'] = stockout_days is not None and stockout_days < 7
        item_dict['confidence'] = 'medium' if stockout_days is not None else 'insufficient_data'

        if cluster_row:
            item_dict['cluster_label'] = cluster_row.get('cluster_label')
            item_dict['cluster_color'] = cluster_row.get('cluster_color')
            item_dict['days_until_expiry'] = cluster_row.get('days_until_expiry')
            item_dict['stock_days_remaining'] = stock_days_remaining
        
        items_enriched.append(item_dict)
    
    # Compute summary stats
    expiry_risk_items = [
        i for i in items_enriched
        if i.get('cluster_label') == 'Expiry Risk'
    ]
    reorder_items = [
        i for i in items_enriched
        if i.get('cluster_label') == 'Reorder Now'
    ]
    stable_items = [
        i for i in items_enriched
        if i.get('cluster_label') in ['Stable', 'High Velocity']
    ]
    
    projected_waste_kg = sum([
        i.get('current_stock', 0) for i in expiry_risk_items
        if i.get('unit') == 'kg'
    ])
    projected_waste_l = sum([
        i.get('current_stock', 0) for i in expiry_risk_items
        if i.get('unit') == 'L'
    ])

    projected_waste_display_parts = []
    if projected_waste_kg > 0:
        projected_waste_display_parts.append(f"{round(projected_waste_kg, 2)} kg")
    if projected_waste_l > 0:
        projected_waste_display_parts.append(f"{round(projected_waste_l, 2)} L")
    projected_waste_display = " + ".join(projected_waste_display_parts) if projected_waste_display_parts else "0"
    
    summary = {
        'total_items': len(items_enriched),
        'expiry_risk_count': len(expiry_risk_items),
        'reorder_now_count': len(reorder_items),
        'stable_count': len(stable_items),
        'projected_waste_kg': round(projected_waste_kg, 2),
        'projected_waste_l': round(projected_waste_l, 2),
        'projected_waste_display': projected_waste_display
    }
    
    # Handle search filter
    query = request.args.get('q', '').lower()
    
    if query:
        items_enriched = [
            i for i in items_enriched
            if query in i.get('name', '').lower() or query in i.get('item_id', '')
        ]

    return render_template(
        'dashboard.html',
        items=items_enriched,
        summary=summary,
        search_query=query
    )


@app.route('/log-event')
def log_event_form():
    """Show form to log stock events (received/remaining)."""
    inventory_view = get_food_inventory_df()

    recipes = []
    if not recipe_df.empty:
        recipes = recipe_df[['recipe_id', 'recipe_name']].to_dict('records')

    if inventory_view.empty:
        return render_template(
            'log_event.html',
            items=[],
            recipes=recipes,
            errors=[],
            form_data={},
            preselected_item_id=None,
            today=str(get_today())
        )
    
    items = inventory_view[['item_id', 'name', 'current_stock', 'unit']].to_dict('records')
    return render_template(
        'log_event.html',
        items=items,
        recipes=recipes,
        errors=[],
        form_data={},
        preselected_item_id=request.args.get('item_id'),
        today=str(get_today())
    )


@app.route('/log-event', methods=['POST'])
def log_event_submit():
    """
    Validate and save stock event.
    
    Validation:
    - item_id exists in inventory
    - event_type is "received" or "remaining"
    - quantity is positive float
    - date is valid
    """
    global events_df
    
    try:
        item_id = normalize_item_id_input(request.form.get('item_id'))
        event_type = request.form.get('event_type')
        quantity = float(request.form.get('quantity'))
        event_date = request.form.get('date')
        notes = request.form.get('notes', '')
        
        # Validation
        errors = []
        
        valid_item_ids = set(get_food_inventory_df()['item_id'].astype(str))
        if not item_id or item_id not in valid_item_ids:
            errors.append("Invalid item ID")
        
        if event_type not in ['received', 'remaining']:
            errors.append("Event type must be 'received' or 'remaining'")
        
        if quantity <= 0:
            errors.append("Quantity must be positive")
        
        try:
            pd.to_datetime(event_date)
        except ValueError:
            errors.append("Invalid date format")
        
        if errors:
            inventory_view = get_food_inventory_df()
            items = inventory_view[['item_id', 'name', 'current_stock', 'unit']].to_dict('records')
            recipes = []
            if not recipe_df.empty:
                recipes = recipe_df[['recipe_id', 'recipe_name']].to_dict('records')
            return render_template(
                'log_event.html',
                items=items,
                recipes=recipes,
                errors=errors,
                form_data={
                    'item_id': item_id,
                    'event_type': event_type,
                    'quantity': request.form.get('quantity', ''),
                    'date': event_date,
                    'notes': notes
                },
                preselected_item_id=item_id,
                today=str(get_today())
            ), 400
        
        # Append new event
        if not events_df.empty and 'event_id' in events_df.columns:
            max_event_id = pd.to_numeric(events_df['event_id'], errors='coerce').max()
            new_event_id = int(max_event_id) + 1 if pd.notna(max_event_id) else 1
        else:
            new_event_id = 1
        new_row = pd.DataFrame({
            'event_id': [new_event_id],
            'item_id': [item_id],
            'date': [event_date],
            'event_type': [event_type],
            'quantity': [quantity],
            'notes': [notes]
        })
        
        updated_events_df = pd.concat([events_df, new_row], ignore_index=True)
        persist_dataframe(updated_events_df, 'events')
        events_df = updated_events_df
        update_inventory_stock(item_id, event_type, quantity)
        
        flash(f"Event logged: {event_type} {quantity} units of {item_id}", "success")
        return redirect(url_for('dashboard'))
    
    except Exception as e:
        inventory_view = get_food_inventory_df()
        items = inventory_view[['item_id', 'name', 'current_stock', 'unit']].to_dict('records')
        recipes = []
        if not recipe_df.empty:
            recipes = recipe_df[['recipe_id', 'recipe_name']].to_dict('records')
        return render_template(
            'log_event.html',
            items=items,
            recipes=recipes,
            errors=[str(e)],
            form_data={
                'item_id': request.form.get('item_id', ''),
                'event_type': request.form.get('event_type', ''),
                'quantity': request.form.get('quantity', ''),
                'date': request.form.get('date', str(get_today())),
                'notes': request.form.get('notes', '')
            },
            preselected_item_id=request.form.get('item_id'),
            today=str(get_today())
        ), 400


@app.route('/recipes')
def recipes():
    """
    Recipe management and demand forecasting.
    
    For each recipe, show:
    - Ingredients with current stock and uses remaining
    - Bottleneck ingredient (first to run out)
    - Demand forecast (7-day projection)
    """
    if recipe_df.empty or ingredients_df.empty:
        return render_template('recipes.html', recipes=[])
    
    recipes_with_forecast = []
    order_copy = order_df.copy()
    if not order_copy.empty and 'date' in order_copy.columns:
        order_copy['date'] = pd.to_datetime(order_copy['date'], errors='coerce')
    
    for idx, recipe in recipe_df.iterrows():
        recipe_dict = recipe.to_dict()
        recipe_id = recipe_dict['recipe_id']

        # Build ingredient summary for UI
        ingredient_summary = get_recipe_ingredient_summary(
            recipe_id,
            ingredients_df,
            inventory_df
        )
        recipe_dict['ingredients'] = [
            {
                'item_id': ing.get('item_id'),
                'name': ing.get('item_name', 'Unknown'),
                'quantity_per_drink': ing.get('quantity_per_unit', 0),
                'unit': ing.get('unit', 'units'),
                'current_stock': ing.get('current_stock', 0),
                'uses_remaining': ing.get('estimated_uses_remaining')
            }
            for ing in ingredient_summary
        ]

        # Recipe-level 7-day forecast (template expects recipe.forecast)
        recipe_orders = pd.DataFrame()
        if not order_copy.empty and 'recipe_id' in order_copy.columns:
            recipe_orders = order_copy[order_copy['recipe_id'] == recipe_id].copy()

        avg_daily_demand = 0.0
        insufficient_data = True
        if not recipe_orders.empty and recipe_orders['date'].notna().any():
            cutoff_date = get_today() - timedelta(days=14)
            recent_orders = recipe_orders[
                recipe_orders['date'].dt.date >= cutoff_date
            ].copy()

            if not recent_orders.empty:
                daily_sales = recent_orders.groupby(
                    recent_orders['date'].dt.date
                )['quantity_sold'].sum()

                if not daily_sales.empty:
                    avg_daily_demand = float(daily_sales.mean())
                    insufficient_data = len(daily_sales.index) < 5

        recipe_dict['forecast'] = {
            'avg_daily_demand': round(avg_daily_demand, 2),
            'projected_total': round(avg_daily_demand * 7, 2),
            'insufficient_data': insufficient_data,
            'contributing_recipes': [
                {
                    'recipe_id': recipe_id,
                    'recipe_name': recipe_dict.get('recipe_name', 'Unknown')
                }
            ] if avg_daily_demand > 0 else []
        }
        
        # Find bottleneck ingredient
        avg_orders = avg_daily_demand if avg_daily_demand > 0 else 10
        bottleneck = find_bottleneck_ingredient(
            recipe_id,
            ingredients_df,
            inventory_df,
            avg_orders
        )
        if bottleneck:
            bottleneck['name'] = bottleneck.get('item_name', 'Unknown')
        recipe_dict['bottleneck'] = bottleneck
        
        recipes_with_forecast.append(recipe_dict)
    
    return render_template('recipes.html', recipes=recipes_with_forecast)


@app.route('/orders', methods=['POST'])
def log_order():
    """
    Log cafe sales order (recipe sold).
    
    Validation:
    - recipe_id exists
    - quantity_sold is positive integer
    """
    global order_df
    
    try:
        recipe_id = str(request.form.get('recipe_id', '')).strip()
        quantity_sold = int(request.form.get('quantity_sold'))
        
        # Validation
        if recipe_id not in recipe_df['recipe_id'].values:
            flash("Invalid recipe ID", "error")
            return redirect(url_for('log_event_form'))
        
        if quantity_sold <= 0:
            flash("Quantity must be positive", "error")
            return redirect(url_for('log_event_form'))

        recipe_match = recipe_df[recipe_df['recipe_id'] == recipe_id]
        recipe_name = (
            recipe_match.iloc[0].get('recipe_name', 'Unknown')
            if not recipe_match.empty else 'Unknown'
        )
        
        # Append order
        if not order_df.empty and 'order_id' in order_df.columns:
            max_order_id = pd.to_numeric(order_df['order_id'], errors='coerce').max()
            new_order_id = int(max_order_id) + 1 if pd.notna(max_order_id) else 1
        else:
            new_order_id = 1
        new_row = pd.DataFrame({
            'order_id': [new_order_id],
            'date': [str(get_today())],
            'recipe_id': [recipe_id],
            'quantity_sold': [quantity_sold]
        })
        
        updated_order_df = pd.concat([order_df, new_row], ignore_index=True)
        persist_dataframe(updated_order_df, 'orders')
        order_df = updated_order_df
        apply_order_to_inventory(recipe_id, quantity_sold)
        
        flash(f"Order logged: {quantity_sold}x {recipe_id} - {recipe_name}", "success")
        return redirect(url_for('log_event_form'))
    
    except Exception as e:
        flash(f"Error: {str(e)}", "error")
        return redirect(url_for('log_event_form'))


@app.route('/recent-logs')
def recent_logs():
    """Show most recent stock events and order logs for quick auditing."""
    recent_events = []
    recent_orders = []

    if not events_df.empty:
        events_view = events_df.copy()
        events_view['date'] = pd.to_datetime(events_view['date'], errors='coerce')
        events_view = events_view.sort_values('date', ascending=False)

        if not inventory_df.empty and 'item_id' in inventory_df.columns:
            item_lookup = inventory_df[['item_id', 'name']].copy().rename(columns={'name': 'item_name'})
            events_view = events_view.merge(item_lookup, on='item_id', how='left')
        else:
            events_view['item_name'] = ''

        events_view['date'] = events_view['date'].dt.strftime('%Y-%m-%d').fillna('')
        recent_events = events_view.head(30).to_dict('records')

    if not order_df.empty:
        orders_view = order_df.copy()
        orders_view['date'] = pd.to_datetime(orders_view['date'], errors='coerce')
        orders_view = orders_view.sort_values('date', ascending=False)

        if not recipe_df.empty and 'recipe_id' in recipe_df.columns:
            recipe_lookup = recipe_df[['recipe_id', 'recipe_name']].copy()
            orders_view = orders_view.merge(recipe_lookup, on='recipe_id', how='left')
        else:
            orders_view['recipe_name'] = ''

        orders_view['date'] = orders_view['date'].dt.strftime('%Y-%m-%d').fillna('')
        recent_orders = orders_view.head(30).to_dict('records')

    return render_template(
        'recent_logs.html',
        recent_events=recent_events,
        recent_orders=recent_orders
    )


@app.route('/insights')
def insights_all():
    """
    Deep insights for all at-risk items (Reorder Now, Expiry Risk clusters).
    
    Runs full pipeline (usage reconciliation, clustering, LLM insights,
    supplier matching) for each at-risk item.
    """
    if inventory_df.empty:
        return render_template('insights.html', insight_items=[], insights_list=[])
    
    clustered_df = get_clustered_inventory()
    if clustered_df.empty:
        return render_template('insights.html', insight_items=[], insights_list=[])
    
    # Find at-risk items
    at_risk_ids = clustered_df[
        clustered_df['cluster_label'].isin(['Reorder Now', 'Expiry Risk'])
    ]['item_id'].tolist()
    
    insights_list = []
    ai_budget = {'remaining_calls': 3}
    for item_id in at_risk_ids:
        insights = get_item_insights(item_id, clustered_df, ai_budget=ai_budget)
        if 'error' not in insights:
            insights_list.append(insights)
    
    return render_template('insights.html', insight_items=insights_list, insights_list=insights_list)


@app.route('/insights/<item_id>')
def insights_single(item_id: str):
    """Deep dive insights for a single item."""
    insights = get_item_insights(item_id, get_clustered_inventory())
    
    if 'error' in insights:
        flash(insights['error'], "error")
        return redirect(url_for('dashboard'))
    
    return render_template('insights.html', insight_items=[insights], insights_list=[insights], single_item=True)


@app.route('/developer/advance-time', methods=['POST'])
def developer_advance_time():
    """
    Developer mode: Simulate time progression for testing.
    
    Gated: only available if DEVELOPER_MODE=true
    """
    global DATE_OFFSET_DAYS
    
    if not DEVELOPER_MODE:
        return "Forbidden", 403
    
    try:
        days = int(request.form.get('days', 0))
        DATE_OFFSET_DAYS += days
        
        new_date = get_today()
        flash(
            f"Developer mode: advanced {days} days. "
            f"Current simulated date: {new_date}",
            "info"
        )
    except ValueError:
        flash("Invalid days value", "error")
    
    return redirect(url_for('dashboard'))


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
