import pandas as pd
from datetime import datetime


def derive_usage_from_events(item_id, events_df, reference_date=None):
    """
    Derive daily usage patterns from stock events (received/remaining records).
    
    This function reconstructs daily consumption by tracking opening stock,
    deliveries received, and end-of-day physical counts. When a remaining
    event is unavailable on a given day, the previous day's closing stock
    is carried forward and usage is recorded as zero (no tracking that day).
    
    Business logic:
    - Usage = (opening_stock + received_today) - closing_stock
    - Days without a "remaining" event carry forward the last known stock
      with zero usage recorded.
    - Handles initial days without recorded opening stock by assuming 0.
    
    Args:
        item_id (str): Item ID to filter events for (e.g., '001')
        events_df (pandas.DataFrame): Stock events with columns:
            event_id, item_id, date, event_type, quantity, notes
            - event_type must be either 'received' or 'remaining'
    
    Returns:
        pandas.DataFrame: Daily usage records with columns:
            - date: Calendar date of the event
            - usage: Calculated daily consumption (kg/L/units)
            - closing_stock: End-of-day remaining quantity
            - had_remaining_event: Boolean (True if physical count was recorded)
    """
    if events_df.empty:
        return pd.DataFrame(
            columns=['date', 'usage', 'closing_stock', 'had_remaining_event']
        )
    
    # Filter to item and parse dates
    item_events = events_df[events_df['item_id'] == item_id].copy()
    if item_events.empty:
        return pd.DataFrame(
            columns=['date', 'usage', 'closing_stock', 'had_remaining_event']
        )
    
    item_events['date'] = pd.to_datetime(item_events['date'])
    item_events = item_events.sort_values('date').reset_index(drop=True)
    
    # Build complete date range from first event to today
    first_date = item_events['date'].min()
    today = reference_date if reference_date is not None else datetime.now().date()
    last_date = min(item_events['date'].max(), pd.to_datetime(today))
    date_range = pd.date_range(first_date, last_date, freq='D')
    
    results = []
    opening_stock = 0.0
    last_closing_stock = 0.0
    
    for current_date in date_range:
        # Get all events for this date
        daily_events = item_events[item_events['date'] == current_date]
        
        # Sum received quantities for this day
        received_today = daily_events[
            daily_events['event_type'] == 'received'
        ]['quantity'].sum()
        
        # Look for remaining (physical count) event
        remaining_events = daily_events[
            daily_events['event_type'] == 'remaining'
        ]
        
        if not remaining_events.empty:
            # Use the closing stock from the remaining event
            closing_stock = remaining_events.iloc[-1]['quantity']
            had_remaining_event = True
        else:
            # Carry forward previous closing stock, no recount today
            closing_stock = last_closing_stock
            had_remaining_event = False
        
        # Calculate usage: opening + received - closing
        usage = opening_stock + received_today - closing_stock
        # Prevent negative usage from rounding/adjustment discrepancies
        usage = max(0.0, usage)
        
        results.append({
            'date': current_date.date(),
            'usage': round(usage, 2),
            'closing_stock': round(closing_stock, 2),
            'had_remaining_event': had_remaining_event
        })
        
        # Set up for next day
        opening_stock = closing_stock
        last_closing_stock = closing_stock
    
    return pd.DataFrame(results)


def derive_usage_from_orders(item_id, order_df, ingredients_df):
    """
    Derive daily usage from cafe sales orders and recipe ingredients.
    
    This function calculates expected consumption by identifying all recipes
    that use the specified item, then summing quantity_per_unit across all
    sales of those recipes on each day.
    
    Business logic:
    - Finds all recipes in ingredients_df that contain item_id
    - For each day, multiplies: quantity_sold × quantity_per_unit
    - Sums across all matching recipes to get daily consumption estimate
    - Useful for detecting spillage/waste (when stock usage >> order usage)
    
    Args:
        item_id (str): Item ID to calculate usage for (e.g., '001')
        order_df (pandas.DataFrame): Daily cafe sales with columns:
            order_id, date, recipe_id, quantity_sold
        ingredients_df (pandas.DataFrame): Recipe composition with columns:
            recipe_id, item_id, quantity_per_unit, unit
    
    Returns:
        pandas.DataFrame: Daily estimated usage with columns:
            - date: Calendar date
            - estimated_usage: Theoretical item consumption from orders
    """
    if order_df.empty or ingredients_df.empty:
        return pd.DataFrame(columns=['date', 'estimated_usage'])
    
    # Find all recipes that use this item
    recipes_with_item = ingredients_df[
        ingredients_df['item_id'] == item_id
    ].copy()
    if recipes_with_item.empty:
        return pd.DataFrame(columns=['date', 'estimated_usage'])
    
    recipe_ids = recipes_with_item['recipe_id'].unique()
    
    # Filter orders to only those recipes
    order_copy = order_df.copy()
    order_copy['date'] = pd.to_datetime(order_copy['date'])
    relevant_orders = order_copy[
        order_copy['recipe_id'].isin(recipe_ids)
    ].copy()
    
    if relevant_orders.empty:
        return pd.DataFrame(columns=['date', 'estimated_usage'])
    
    # Merge orders with ingredient quantities (one row per recipe-per-order)
    merged = relevant_orders.merge(
        recipes_with_item[['recipe_id', 'quantity_per_unit']],
        on='recipe_id',
        how='left'
    )
    
    # Calculate consumption per order (quantity sold × per-unit consumption)
    merged['daily_consumption'] = (
        merged['quantity_sold'] * merged['quantity_per_unit']
    )
    
    # Group by date and sum all consumption for this item
    daily_usage = merged.groupby('date')[
        'daily_consumption'
    ].sum().reset_index()
    daily_usage.columns = ['date', 'estimated_usage']
    daily_usage['estimated_usage'] = daily_usage['estimated_usage'].round(2)
    daily_usage['date'] = daily_usage['date'].dt.date
    
    return daily_usage


def reconcile_usage(stock_derived_df, order_derived_df):
    """
    Reconcile stock-derived usage against order-derived usage estimates.
    
    This function compares actual stock consumption (from physical counts)
    with theoretical consumption derived from sales orders. Discrepancies
    indicate spillage, waste, theft, data errors, or untracked movement.
    
    Business logic:
    - Merges both DataFrames on date (left join keeps all stock days)
    - Computes discrepancy = actual - estimated
    - Positive discrepancy: actual use > orders (spillage/waste)
    - Negative discrepancy: actual use < orders (shouldn't happen; data error)
    - Flags rows where abs(discrepancy) > 20% of actual usage
    
    Args:
        stock_derived_df (pandas.DataFrame): Output from derive_usage_from_events
            with columns: date, usage, closing_stock, had_remaining_event
        order_derived_df (pandas.DataFrame): Output from derive_usage_from_orders
            with columns: date, estimated_usage
    
    Returns:
        pandas.DataFrame: Reconciled usage with columns:
            - date: Calendar date
            - usage: Actual stock-derived usage
            - estimated_usage: Order-derived estimated usage (0 if no orders)
            - discrepancy: usage - estimated_usage
            - variance_flagged: Boolean (True if abs(discrepancy) > 0.2 × usage)
    """
    if stock_derived_df.empty:
        return pd.DataFrame(
            columns=['date', 'usage', 'estimated_usage', 'discrepancy',
                     'variance_flagged']
        )
    
    # Ensure date columns are datetime for merging
    stock_copy = stock_derived_df.copy()
    order_copy = order_derived_df.copy()
    
    stock_copy['date'] = pd.to_datetime(stock_copy['date'])
    order_copy['date'] = pd.to_datetime(order_copy['date'])
    
    # Merge with left join (keep all stock days, add orders where available)
    merged = stock_copy.merge(
        order_copy[['date', 'estimated_usage']],
        on='date',
        how='left'
    )
    
    # Fill missing estimated_usage with 0 (no orders recorded for that day)
    merged['estimated_usage'] = merged['estimated_usage'].fillna(0.0)
    
    # Calculate discrepancy (positive = more used than ordered)
    merged['discrepancy'] = (
        merged['usage'] - merged['estimated_usage']
    ).round(2)
    
    # Flag variance: abs(discrepancy) > 20% of actual usage
    # (only flag when usage > 0 to avoid division-by-zero issues)
    merged['variance_flagged'] = (
        (merged['usage'] > 0) &
        (abs(merged['discrepancy']) > 0.2 * merged['usage'])
    )
    
    # Return only essential columns
    result = merged[[
        'date', 'usage', 'estimated_usage', 'discrepancy', 'variance_flagged'
    ]].copy()
    
    result['date'] = result['date'].dt.date
    
    return result
