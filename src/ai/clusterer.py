"""
Inventory clustering module for grouping cafe items by consumption behavior.

K-Means clustering requires feature scaling because the algorithm uses Euclidean
distance, which is sensitive to feature magnitude. Without StandardScaler:
- days_until_expiry (0-100s) would dominate over avg_daily_usage (0-50s)
- Clustering would be biased toward shelf-life patterns
- Performance and label interpretation would be unreliable

StandardScaler transforms all features to mean=0, std=1, ensuring equal
contribution to distance calculations and improving cluster coherence.

random_state=42 ensures deterministic clustering across runs, critical for
reproducible inventory analysis and consistent reorder recommendations.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from typing import Tuple

from .usage_calculator import derive_usage_from_events


def build_feature_matrix(
    inventory_df: pd.DataFrame,
    events_df: pd.DataFrame,
    reference_date: datetime.date | None = None
) -> pd.DataFrame:
    """
    Extract consumption behavior features for each inventory item.
    
    This function builds a feature matrix for clustering by computing:
    - Average daily usage (from physical stock tracking via events)
    - Days remaining before expiry
    - Stock coverage in days (how long current stock lasts at current consumption)
    
    These three features enable clustering into prioritized action groups
    (e.g., "Reorder Now", "Expiry Risk", "Stable Stock").
    
    Business logic:
    - If no usage history exists, assumes zero consumption (999-day coverage)
    - Expiry dates clipped at 0 (already expired items = 0 days to expiry)
    - Stock coverage = inventory / daily_consumption
    
    Args:
        inventory_df (pd.DataFrame): Inventory with columns: item_id, 
                                     current_stock, expiry_date, unit
        events_df (pd.DataFrame): Stock events with columns: event_id, item_id,
                                  date, event_type, quantity, notes
    
    Returns:
        pd.DataFrame: Feature matrix with columns: item_id, avg_daily_usage,
                     days_until_expiry, stock_days_remaining
    """
    results = []
    today = reference_date if reference_date is not None else datetime.now().date()
    
    for idx, row in inventory_df.iterrows():
        item_id = row['item_id']
        current_stock = row['current_stock']
        expiry_date = pd.to_datetime(row['expiry_date']).date()
        
        # Derive usage history from stock events
        usage_df = derive_usage_from_events(item_id, events_df, reference_date=today)
        
        # Calculate average daily usage
        if usage_df.empty or usage_df['usage'].sum() == 0:
            avg_daily_usage = 0.0
        else:
            avg_daily_usage = usage_df['usage'].mean()
        
        # Calculate days until expiry (minimum 0)
        days_until_expiry = max(0, (expiry_date - today).days)
        
        # Calculate stock coverage in days
        if avg_daily_usage > 0:
            stock_days_remaining = current_stock / avg_daily_usage
        else:
            stock_days_remaining = 999.0  # No consumption, won't stock out
        
        results.append({
            'item_id': item_id,
            'avg_daily_usage': round(avg_daily_usage, 2),
            'days_until_expiry': days_until_expiry,
            'stock_days_remaining': round(stock_days_remaining, 2)
        })
    
    return pd.DataFrame(results)


def label_cluster(centroid_real: np.ndarray, all_centroids_real: np.ndarray) -> Tuple[str, str]:
    """
    Assign a human-readable label and color to a cluster based on its centroid.
    
     Labeling heuristic (applied in priority order):

     1. "Expiry Risk" (#f59e0b amber): Item expires very soon and should be
         consumed/promoted first to reduce waste.

     2. "Reorder Now" (#ef4444 red): Stock will run out within 7 days.
         Action: place reorder immediately to avoid stockout.

     3. "High Velocity" (#2563eb blue): Usage is high relative to peers but
         stock coverage is healthy. Action: monitor closely.

     4. "Stable" (#22c55e green): All other items. Action: monitor routinely.
    
    Note: This heuristic is calibrated for a small 150-200 customer/day Indian
    cafe. Real-world thresholds should be recalibrated based on:
    - Actual reorder lead times and batch sizes
    - Historical waste patterns and spillage rates
    - Shelf-life utilization rates
    
    Args:
        centroid_real (np.ndarray): 1D array of real-unit feature values:
                                    [avg_daily_usage, days_until_expiry,
                                     stock_days_remaining]
        all_centroids_real (np.ndarray): All cluster centroids for comparison
                                         (shape: n_clusters x 3)
    
    Returns:
        Tuple[str, str]: (cluster_label, color_hex)
    """
    avg_daily_usage, days_until_expiry, stock_days_remaining = centroid_real
    
    # Rule 1: Expiry Risk
    if days_until_expiry < 3:
        return ("Expiry Risk", "#f59e0b")
    
    # Rule 2: Reorder Now (low stock)
    if stock_days_remaining < 7:
        return ("Reorder Now", "#ef4444")

    # Rule 3: High Velocity (fast moving but still well-covered)
    if all_centroids_real.shape[0] > 1:
        avg_usage_across_clusters = all_centroids_real[:, 0].mean()
        if avg_usage_across_clusters > 0 and \
           avg_daily_usage > 1.5 * avg_usage_across_clusters and \
           stock_days_remaining >= 14:
            return ("High Velocity", "#2563eb")

    # Rule 4: Stable (default)
    return ("Stable", "#22c55e")


def cluster_items(feature_df: pd.DataFrame, n_clusters: int = 4) -> Tuple[pd.DataFrame, KMeans]:
    """
    Cluster inventory items using K-Means on consumption behavior features.
    
    This function applies StandardScaler to ensure all features contribute
    equally to clustering (see module docstring for why this is critical),
    then fits K-Means with a deterministic random state for reproducibility.
    
    Business flow:
    - Scales features to unit variance (required for distance-based clustering)
    - Fits K-Means with k clusters, interpreting results in real units
    - Labels each cluster with priority action and visualization color
    - Assigns each item to its nearest cluster
    
    Edge cases:
    - Fewer items than requested clusters → automatically reduces k
    - Empty feature_df → returns empty DataFrame with cluster columns
    
    Args:
        feature_df (pd.DataFrame): Output from build_feature_matrix with columns:
                                   item_id, avg_daily_usage, days_until_expiry,
                                   stock_days_remaining
        n_clusters (int, optional): Number of clusters. Defaults to 4.
                                   Automatically reduced if fewer items exist.
    
    Returns:
        Tuple[pd.DataFrame, KMeans]:
            - Enriched feature_df with columns: cluster_id, cluster_label,
              cluster_color (hex string)
            - Fitted KMeans object for diagnostics/inspection
    
    Example:
        >>> clustered_df, kmeans = cluster_items(feature_df, n_clusters=4)
        >>> clustered_df[['item_id', 'cluster_label']]
           item_id cluster_label
        0      001    Reorder Now
        1      002      Stable
    """
    # Handle edge case: fewer items than clusters
    actual_n_clusters = min(n_clusters, len(feature_df))
    
    if len(feature_df) == 0:
        # Return empty result with proper columns
        empty_result = feature_df.copy()
        empty_result['cluster_id'] = pd.Series(dtype='int64')
        empty_result['cluster_label'] = pd.Series(dtype='str')
        empty_result['cluster_color'] = pd.Series(dtype='str')
        return empty_result, None
    
    # Extract feature columns
    features = feature_df[
        ['avg_daily_usage', 'days_until_expiry', 'stock_days_remaining']
    ].values
    
    # Scale features (required for K-Means to work properly)
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Fit K-Means with deterministic seed
    kmeans = KMeans(
        n_clusters=actual_n_clusters,
        random_state=42,
        n_init=15
    )
    cluster_ids = kmeans.fit_predict(features_scaled)
    
    # Inverse transform centroids back to real units for labeling
    centroids_real = scaler.inverse_transform(kmeans.cluster_centers_)
    
    # Label each cluster based on centroid position
    cluster_labels = []
    cluster_colors = []
    for centroid in centroids_real:
        label, color = label_cluster(centroid, centroids_real)
        cluster_labels.append(label)
        cluster_colors.append(color)
    
    # Enrich feature_df with cluster assignments
    result_df = feature_df.copy()
    result_df['cluster_id'] = cluster_ids
    result_df['cluster_label'] = result_df['cluster_id'].map(
        lambda cid: cluster_labels[cid]
    )
    result_df['cluster_color'] = result_df['cluster_id'].map(
        lambda cid: cluster_colors[cid]
    )

    # Apply item-level risk overrides so urgent conditions are never masked by
    # centroid-level cluster labels.
    expiry_mask = result_df['days_until_expiry'] < 3
    reorder_mask = (~expiry_mask) & (result_df['stock_days_remaining'] < 7)

    result_df.loc[expiry_mask, 'cluster_label'] = 'Expiry Risk'
    result_df.loc[expiry_mask, 'cluster_color'] = '#f59e0b'

    result_df.loc[reorder_mask, 'cluster_label'] = 'Reorder Now'
    result_df.loc[reorder_mask, 'cluster_color'] = '#ef4444'
    
    return result_df, kmeans
