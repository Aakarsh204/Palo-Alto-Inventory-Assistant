"""
Recipe-based ingredient demand estimation for cafe inventory.

This module projects ingredient consumption by analyzing historical recipe
sales and ingredient compositions. Forward-looking demand estimates enable
proactive procurement and prevent stockouts on bottleneck ingredients.
"""

import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any


def estimate_ingredient_demand(
    item_id: str,
    order_df: pd.DataFrame,
    ingredients_df: pd.DataFrame,
    days_forward: int = 7
) -> Dict[str, Any]:
    """
    Estimate future ingredient demand based on recent recipe sales history.
    
    This function extrapolates future consumption by analyzing recent order
    patterns for recipes containing the target ingredient. A 3-day rolling
    average smooths daily order noise (weekday/weekend variance, one-off
    catering orders) to reveal underlying demand trend.
    
    Rolling average rationale: Same as predictor.py. Order counts fluctuate
    daily due to customer traffic patterns; a 3-day window captures the
    repeat-purchase cycle for most cafe items without losing week-level
    trends. With min_periods=1, the first 1-2 days are included (not NaN).
    
    Projection assumes average demand is stable (no seasonal events, holidays
    captured); intended for next 1-3 weeks ahead.
    
    Args:
        item_id (str): Target ingredient item ID (e.g., '001')
        order_df (pd.DataFrame): Daily recipe sales with columns: date,
                                recipe_id, quantity_sold
        ingredients_df (pd.DataFrame): Recipe-ingredient mapping with columns:
                                      recipe_id, item_id, quantity_per_unit, unit
        days_forward (int, optional): Days to project ahead. Defaults to 7.
    
    Returns:
        Dict[str, Any]: Demand forecast with keys:
            - item_id: Input item ID
            - avg_daily_demand: Smoothed average daily consumption (float)
            - projected_total: Total estimated consumption (days_forward days)
            - days_forward: Projection window (int)
            - contributing_recipes: List of recipe_ids using this item
            - insufficient_data: Boolean (True if < 5 days of order history)
    
    Example:
        >>> estimate_ingredient_demand('001', orders, ingredients, days_forward=7)
        {
            'item_id': '001',
            'avg_daily_demand': 1.25,
            'projected_total': 8.75,
            'days_forward': 7,
            'contributing_recipes': ['R001', 'R005'],
            'insufficient_data': False
        }
    """
    # Find all recipes using this item
    recipes_with_item = ingredients_df[
        ingredients_df['item_id'] == item_id
    ].copy()
    
    if recipes_with_item.empty:
        return {
            'item_id': item_id,
            'avg_daily_demand': 0.0,
            'projected_total': 0.0,
            'days_forward': days_forward,
            'contributing_recipes': [],
            'insufficient_data': True
        }
    
    recipe_ids = recipes_with_item['recipe_id'].unique().tolist()
    
    # Filter orders to last 30 days
    order_copy = order_df.copy()
    order_copy['date'] = pd.to_datetime(order_copy['date'])
    today = datetime.now().date()
    cutoff_date = today - timedelta(days=30)
    
    recent_orders = order_copy[
        order_copy['date'].dt.date >= cutoff_date
    ].copy()
    
    # Merge orders with ingredient quantities
    relevant_orders = recent_orders[
        recent_orders['recipe_id'].isin(recipe_ids)
    ].copy()
    
    # Check if we have sufficient data FOR THIS ITEM (after filtering)
    if relevant_orders.empty or len(relevant_orders['date'].dt.date.unique()) < 5:
        insufficient_data = True
    else:
        insufficient_data = False

    if relevant_orders.empty:
        return {
            'item_id': item_id,
            'avg_daily_demand': 0.0,
            'projected_total': 0.0,
            'days_forward': days_forward,
            'contributing_recipes': recipe_ids,
            'insufficient_data': insufficient_data
        }
    
    # Merge with ingredient quantities
    merged = relevant_orders.merge(
        recipes_with_item[['recipe_id', 'quantity_per_unit']],
        on='recipe_id',
        how='left'
    )
    
    # Calculate consumption per order
    merged['daily_consumption'] = (
        merged['quantity_sold'] * merged['quantity_per_unit']
    )
    
    # Group by date and sum consumption
    daily_demand = merged.groupby('date')[
        'daily_consumption'
    ].sum().reset_index()
    daily_demand.columns = ['date', 'demand']
    daily_demand = daily_demand.sort_values('date')
    
    # Apply 30-day rolling average (min_periods=1 to avoid NaN)
    smoothed = daily_demand['demand'].rolling(
        window=30,
        min_periods=1
    ).mean()
    
    avg_daily_demand = smoothed.mean()
    projected_total = avg_daily_demand * days_forward
    
    return {
        'item_id': item_id,
        'avg_daily_demand': round(avg_daily_demand, 3),
        'projected_total': round(projected_total, 3),
        'days_forward': days_forward,
        'contributing_recipes': recipe_ids,
        'insufficient_data': insufficient_data
    }


def get_recipe_ingredient_summary(
    recipe_id: str,
    ingredients_df: pd.DataFrame,
    inventory_df: pd.DataFrame
) -> List[Dict[str, Any]]:
    """
    Summarize all ingredients in a recipe with current stock levels.
    
    Enables quick visibility into recipe feasibility: How many complete
    servings can be made with current stock? Which ingredient is limiting?
    
    Args:
        recipe_id (str): Recipe ID (e.g., 'R001')
        ingredients_df (pd.DataFrame): With columns: recipe_id, item_id,
                                      quantity_per_unit, unit
        inventory_df (pd.DataFrame): With columns: item_id, name,
                                    current_stock, unit (for verification)
    
    Returns:
        List[Dict[str, Any]]: One dict per ingredient with keys:
            - item_id: Ingredient item ID
            - item_name: Human-readable name
            - quantity_per_unit: Amount used per recipe serving
            - unit: Unit of measurement
            - current_stock: Available quantity
            - estimated_uses_remaining: How many complete recipes can be made
              (integer: floor(current_stock / quantity_per_unit))
    
    Example:
        >>> get_recipe_ingredient_summary('R001', ingredients, inventory)
        [
            {
                'item_id': '001',
                'item_name': 'Coffee Beans',
                'quantity_per_unit': 0.018,
                'unit': 'kg',
                'current_stock': 3.2,
                'estimated_uses_remaining': 177
            },
            ...
        ]
    """
    # Find ingredients for this recipe
    recipe_ingredients = ingredients_df[
        ingredients_df['recipe_id'] == recipe_id
    ].copy()
    
    if recipe_ingredients.empty:
        return []
    
    # Join with inventory to get names, units, and stock
    inventory_columns = ['item_id', 'name', 'current_stock']
    if 'unit' in inventory_df.columns:
        inventory_columns.append('unit')

    inventory_lookup = inventory_df[inventory_columns].rename(
        columns={
            'name': 'item_name',
            'unit': 'stock_unit'
        }
    )

    summary = recipe_ingredients.merge(
        inventory_lookup,
        on='item_id',
        how='left'
    )
    
    results = []
    for idx, row in summary.iterrows():
        quantity_per_unit = row['quantity_per_unit']
        current_stock = row['current_stock']
        
        # Skip items where inventory data is missing (NaN from left join)
        if pd.isna(current_stock):
            continue

        # Calculate how many complete recipes can be made
        if quantity_per_unit > 0:
            uses_remaining = int(current_stock // quantity_per_unit)
        else:
            uses_remaining = 0
        
        results.append({
            'item_id': row['item_id'],
            'item_name': row['item_name'],
            'quantity_per_unit': row['quantity_per_unit'],
            'unit': row.get('stock_unit', row.get('unit')),
            'current_stock': row['current_stock'],
            'estimated_uses_remaining': uses_remaining
        })
    
    return results


def find_bottleneck_ingredient(
    recipe_id: str,
    ingredients_df: pd.DataFrame,
    inventory_df: pd.DataFrame,
    daily_orders: float
) -> Optional[Dict[str, Any]]:
    """
    Identify which ingredient will run out first given expected daily demand.
    
    For a recipe with specific daily order volume, this function calculates
    how many days each ingredient will last. The bottleneck ingredient
    (shortest duration) determines the recipe's survivability without reorder.
    
    Business use: If bottleneck ingredient runs out in 2 days but lead time
    is 3 days, immediate reorder is required to avoid service interruption.
    
    Args:
        recipe_id (str): Recipe ID
        ingredients_df (pd.DataFrame): Recipe-ingredient mapping
        inventory_df (pd.DataFrame): Current stock levels
        daily_orders (float): Expected daily order volume for this recipe
    
    Returns:
        Optional[Dict[str, Any]]: Bottleneck ingredient info with keys:
            - item_id: Ingredient item ID
            - item_name: Human-readable name
            - days_remaining: How many days this ingredient lasts
            - quantity_per_unit: Amount per recipe serving
            Or None if daily_orders is 0 or recipe has no ingredients
    
    Example:
        >>> find_bottleneck_ingredient('R001', ingredients, inventory, 20.0)
        {
            'item_id': '004',
            'item_name': 'Vanilla Syrup',
            'days_remaining': 3.2,
            'quantity_per_unit': 0.03
        }
    """
    if daily_orders <= 0:
        return None
    
    # Get recipe ingredients and stock
    summary = get_recipe_ingredient_summary(recipe_id, ingredients_df, inventory_df)
    
    if not summary:
        return None
    
    # Calculate days remaining per ingredient
    bottleneck = None
    min_days = float('inf')
    
    for ingredient in summary:
        quantity_per_unit = ingredient['quantity_per_unit']
        current_stock = ingredient['current_stock']
        
        # Days remaining = stock / (per-unit consumption × daily orders)
        daily_consumption = quantity_per_unit * daily_orders
        if daily_consumption > 0:
            days_remaining = current_stock / daily_consumption
        else:
            days_remaining = float('inf')
        
        if days_remaining < min_days:
            min_days = days_remaining
            bottleneck = {
                'item_id': ingredient['item_id'],
                'item_name': ingredient['item_name'],
                'days_remaining': round(days_remaining, 2),
                'quantity_per_unit': ingredient['quantity_per_unit']
            }
    
    return bottleneck
