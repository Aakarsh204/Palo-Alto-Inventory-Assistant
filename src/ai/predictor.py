import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from typing import List, Optional, Dict, Union


def smooth_usage(usage_list: List[float], window: int = 3) -> List[float]:
    """
    Apply a rolling mean filter to smooth noisy daily usage data.
    
    This function reduces short-term fluctuations in usage patterns,
    making trend analysis more reliable for stockout prediction. Uses
    min_periods=1 to ensure the first few days (which have fewer
    observations) are still smoothed rather than producing NaN values.
    
    min_periods=1 is critical because without it, the first (window-1)
    days would be NaN, truncating the data and losing early-period
    information. With min_periods=1, day 1 becomes the average of [day1],
    day 2 becomes the average of [day1, day2], etc., preserving all data.
    
    Args:
        usage_list (List[float]): Chronological list of daily usage values
        window (int, optional): Rolling window size. Defaults to 3.
    
    Returns:
        List[float]: Smoothed usage values as a plain Python list
    
    Example:
        >>> smooth_usage([5.0, 3.0, 7.0, 6.0, 8.0], window=3)
        [5.0, 4.0, 5.0, 5.33..., 7.0]
    """
    if not usage_list:
        return []
    
    series = pd.Series(usage_list)
    smoothed = series.rolling(window=window, min_periods=1).mean()
    return smoothed.tolist()


def predict_stockout(
    usage_history: List[float],
    current_stock: float
) -> Optional[int]:
    """
    Predict the number of days until stockout using cumulative usage regression.
    
    This function uses linear regression on cumulative consumption trends to
    forecast when stock will reach zero. Cumulative sum transforms point-in-time
    usage into a monotonically increasing line, whose slope represents the
    average daily consumption rate. Extrapolating this line to zero stock
    predicts stockout date.
    
    Business logic:
    - Smooths usage to reduce daily noise and capture true trend
    - Fits a line to cumulative consumption vs. day number
    - Solves for the day when cumulative usage equals current_stock
    - Returns days remaining (capped at 60 for "60+ days" display)
    
    Edge cases handled:
    - Insufficient history (< 5 days) → None
    - Non-positive stock → None
    - Non-positive consumption rate (not being used) → None
    
    Args:
        usage_history (List[float]): Chronological daily usage values
        current_stock (float): Current remaining quantity in inventory
    
    Returns:
        Optional[int]: Days until stockout (0-60), or None if prediction
                       cannot be made (insufficient data, zero stock, etc.)
    
    Example:
        >>> predict_stockout([1.0, 1.2, 0.9, 1.1, 1.0], 10.0)
        10  # Approximately 10 days remaining
    """
    if len(usage_history) < 5:
        return None
    
    if current_stock <= 0:
        return None
    
    # Smooth usage to reduce daily variance
    smoothed = smooth_usage(usage_history, window=7)
    
    # Build X (day numbers) and y (cumulative sum of usage)
    # Day numbers start at 1, 2, 3, ... to match business day counting
    n_days = len(smoothed)
    X = np.array(range(1, n_days + 1)).reshape(-1, 1)
    y = np.array(np.cumsum(smoothed)).reshape(-1, 1)
    
    # Fit linear regression: cumulative_usage = intercept + slope * day
    model = LinearRegression()
    model.fit(X, y)
    
    slope = model.coef_[0][0]
    intercept = model.intercept_[0]
    
    # If slope is non-positive, item is not being consumed
    if slope <= 0:
        return None
    
    # Solve for stockout day: 
    # Cumulative usage at day n (end of history) + remaining stock = total consumable
    # We need: (cumsum_at_day_n + current_stock) = intercept + slope * day_of_stockout
    cumsum_at_day_n = y[-1][0]  # Total consumed through today
    total_consumable = cumsum_at_day_n + current_stock
    
    day_of_stockout = (total_consumable - intercept) / slope
    days_remaining = day_of_stockout - n_days
    
    # For active consumption with positive stock, surface at least 1 day
    # so UI doesn't show a confusing "0 days" while stock is still > 0.
    result = max(1, round(days_remaining))
    result = min(result, 60)
    
    return int(result)


def compare_predictions(
    stock_prediction: Optional[int],
    order_prediction: Optional[int]
) -> Dict[str, Union[str, int, None]]:
    """
    Compare stockout predictions from stock-derived and order-derived sources.
    
    This function evaluates confidence in the overall prediction by comparing
    two independent estimates. High agreement between methods increases
    confidence; divergence suggests anomalies (spillage, untracked usage,
    forecast errors, or data quality issues).
    
    Confidence levels:
    - "high": Both methods agree within 2 days (tight alignment)
    - "medium": Both methods agree within 5 days (reasonable alignment)
    - "low": Both methods available but diverge >5 days, or only one available
    - "insufficient_data": No predictions available from either method
    
    Args:
        stock_prediction (Optional[int]): Predicted days to stockout from
                                          stock-derived usage (0-60 or None)
        order_prediction (Optional[int]): Predicted days to stockout from
                                          order-derived usage (0-60 or None)
    
    Returns:
        Dict[str, Union[str, int, None]]: Dictionary with keys:
            - stock_days: stock_prediction value or None
            - order_days: order_prediction value or None
            - confidence: confidence level string
            - divergence_days: absolute difference (or None if unavailable)
    
    Example:
        >>> compare_predictions(10, 11)
        {'stock_days': 10, 'order_days': 11, 'confidence': 'high',
         'divergence_days': 1}
        
        >>> compare_predictions(10, None)
        {'stock_days': 10, 'order_days': None, 'confidence': 'low',
         'divergence_days': None}
    """
    # Determine confidence and divergence
    if stock_prediction is None and order_prediction is None:
        confidence = "insufficient_data"
        divergence = None
    elif stock_prediction is None or order_prediction is None:
        confidence = "low"
        divergence = None
    else:
        # Both predictions available, calculate divergence
        divergence = abs(stock_prediction - order_prediction)
        
        if divergence <= 2:
            confidence = "high"
        elif divergence <= 5:
            confidence = "medium"
        else:
            confidence = "low"
    
    return {
        'stock_days': stock_prediction,
        'order_days': order_prediction,
        'confidence': confidence,
        'divergence_days': divergence
    }
