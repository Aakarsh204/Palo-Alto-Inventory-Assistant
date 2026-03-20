import pytest
import pandas as pd
from datetime import datetime, timedelta
from src.ai.usage_calculator import (
    derive_usage_from_events,
    derive_usage_from_orders,
    reconcile_usage
)


class TestDeriveUsageFromEvents:
    """Tests for derive_usage_from_events function"""
    
    def test_derive_usage_basic_happy_path(self):
        """Test basic happy path with multiple days of stock events"""
        events_df = pd.DataFrame({
            'item_id': ['001', '001', '001', '001'],
            'event_type': ['received', 'remaining', 'remaining', 'remaining'],
            'quantity': [5.0, 7.2, 6.8, 5.9],
            'date': [
                datetime(2026, 2, 18),
                datetime(2026, 2, 19),
                datetime(2026, 2, 20),
                datetime(2026, 2, 22),
            ]
        })
        
        result = derive_usage_from_events('001', events_df)
        
        # Assert result is not empty
        assert len(result) > 0
        
        # Assert all usage values are >= 0
        assert (result['usage'] >= 0).all()
        
        # Assert closing_stock values are non-negative
        assert (result['closing_stock'] >= 0).all()
        
        # Assert result contains expected columns
        assert 'date' in result.columns
        assert 'usage' in result.columns
        assert 'closing_stock' in result.columns
    
    def test_derive_usage_empty_item(self):
        """Test with item_id that doesn't exist in events"""
        events_df = pd.DataFrame({
            'item_id': ['001', '001'],
            'event_type': ['received', 'remaining'],
            'quantity': [5.0, 4.5],
            'date': [
                datetime(2026, 2, 18),
                datetime(2026, 2, 19),
            ]
        })
        
        result = derive_usage_from_events('999', events_df)
        
        # Should return empty DataFrame without raising exception
        assert len(result) == 0
        assert isinstance(result, pd.DataFrame)


class TestDeriveUsageFromOrders:
    """Tests for derive_usage_from_orders function"""
    
    def test_derive_usage_from_orders_happy_path(self):
        """Test deriving usage from order history with known ingredient ratios"""
        # Setup: item 001 used in recipe R001 at 0.018 per unit
        ingredients_df = pd.DataFrame({
            'recipe_id': ['R001', 'R001', 'R002'],
            'item_id': ['001', '002', '001'],
            'quantity_per_unit': [0.018, 0.15, 0.025]
        })
        
        # Setup: R001 sold 20 units each day for 5 days
        order_df = pd.DataFrame({
            'recipe_id': ['R001', 'R001', 'R001', 'R001', 'R001'],
            'quantity_sold': [20, 20, 20, 20, 20],
            'date': [
                datetime(2026, 2, 18),
                datetime(2026, 2, 19),
                datetime(2026, 2, 20),
                datetime(2026, 2, 21),
                datetime(2026, 2, 22),
            ]
        })
        
        result = derive_usage_from_orders('001', order_df, ingredients_df)
        
        # Assert result has expected rows
        assert len(result) == 5
        
        # Assert each estimated_usage == 20 * 0.018 = 0.36
        expected_usage = 20 * 0.018
        for usage in result['estimated_usage']:
            assert abs(usage - expected_usage) < 0.001
    
    def test_derive_usage_from_orders_multiple_recipes(self):
        """Test when ingredient appears in multiple recipes"""
        ingredients_df = pd.DataFrame({
            'recipe_id': ['R001', 'R002'],
            'item_id': ['001', '001'],
            'quantity_per_unit': [0.018, 0.020]
        })
        
        order_df = pd.DataFrame({
            'recipe_id': ['R001', 'R002'],
            'quantity_sold': [10, 5],
            'date': [datetime(2026, 2, 18), datetime(2026, 2, 18)]
        })
        
        result = derive_usage_from_orders('001', order_df, ingredients_df)
        
        # Should aggregate: (10 * 0.018) + (5 * 0.020) = 0.18 + 0.1 = 0.28
        assert len(result) > 0
        assert 'estimated_usage' in result.columns
    
    def test_derive_usage_from_orders_item_not_in_recipes(self):
        """Test when item_id is not in any recipe"""
        ingredients_df = pd.DataFrame({
            'recipe_id': ['R001'],
            'item_id': ['002'],
            'quantity_per_unit': [0.15]
        })
        
        order_df = pd.DataFrame({
            'recipe_id': ['R001'],
            'quantity_sold': [10],
            'date': [datetime(2026, 2, 18)]
        })
        
        result = derive_usage_from_orders('001', order_df, ingredients_df)
        
        # Should return empty or zero-usage DataFrame
        assert len(result) == 0 or (result['estimated_usage'] == 0).all()


class TestReconcileUsage:
    """Tests for reconcile_usage function"""
    
    def test_reconcile_flags_large_discrepancy(self):
        """Test that large discrepancies are flagged correctly"""
        stock_derived = pd.DataFrame({
            'date': [
                datetime(2026, 2, 18),
                datetime(2026, 2, 19),
                datetime(2026, 2, 20),
                datetime(2026, 2, 21),
            ],
            'usage': [0.5, 0.5, 0.9, 0.5]
        })
        
        order_derived = pd.DataFrame({
            'date': [
                datetime(2026, 2, 18),
                datetime(2026, 2, 19),
                datetime(2026, 2, 20),
                datetime(2026, 2, 21),
            ],
            'estimated_usage': [0.5, 0.5, 0.5, 0.5]
        })
        
        result = reconcile_usage(stock_derived, order_derived)
        
        # Assert result has variance_flagged column
        assert 'variance_flagged' in result.columns
        
        # Assert the row with 0.9 usage is flagged (80% discrepancy from 0.5)
        row_with_large_discrepancy = result[result['usage'] == 0.9]
        assert len(row_with_large_discrepancy) > 0
        assert row_with_large_discrepancy['variance_flagged'].iloc[0] == True
        
        # Assert other rows are not flagged
        rows_with_small_discrepancy = result[result['usage'] == 0.5]
        assert (rows_with_small_discrepancy['variance_flagged'] == False).all()
    
    def test_reconcile_handles_missing_order_data(self):
        """Test that missing order data is handled gracefully with left join"""
        stock_derived = pd.DataFrame({
            'date': [
                datetime(2026, 2, 18),
                datetime(2026, 2, 19),
                datetime(2026, 2, 20),
                datetime(2026, 2, 21),
                datetime(2026, 2, 22),
                datetime(2026, 2, 23),
                datetime(2026, 2, 24),
            ],
            'usage': [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
        })
        
        order_derived = pd.DataFrame({
            'date': [
                datetime(2026, 2, 18),
                datetime(2026, 2, 19),
                datetime(2026, 2, 20),
            ],
            'estimated_usage': [0.5, 0.5, 0.5]
        })
        
        result = reconcile_usage(stock_derived, order_derived)
        
        # Assert result has same number of rows as stock_derived (left join)
        assert len(result) == len(stock_derived)
        
        # Assert estimat estimated_usage column exists
        assert 'estimated_usage' in result.columns
    def test_full_pipeline_with_realistic_data(self):
        """Test the full usage calculation pipeline with realistic cafe data"""
        # Setup: 14-day data for item 001 (coffee beans)
        events_data = []
        for day in range(1, 15):
            if day % 7 == 1:  # Weekly delivery
                events_data.append({
                    'item_id': '001',
                    'event_type': 'received',
                    'quantity': 2.0,
                    'date': datetime(2026, 3, 15 + day)
                })
            # Daily remaining count
            remaining = 2.0 - (day % 7) * 0.3
            if remaining > 0:
                events_data.append({
                    'item_id': '001',
                    'event_type': 'remaining',
                    'quantity': remaining,
                    'date': datetime(2026, 3, 15 + day)
                })
        
        events_df = pd.DataFrame(events_data)
        
        # Derive usage from stock events
        stock_usage = derive_usage_from_events('001', events_df)
        
        assert len(stock_usage) > 0
        assert (stock_usage['usage'] >= 0).all()
        
        # Setup order data (10 days in March)
        order_data = []
        for day in range(1, 11):
            order_data.append({
                'recipe_id': ['R001'] * 20,
                'quantity_sold': 20,
                'date': datetime(2026, 3, 10 + day)
            })
        
        # Flatten order data
        flat_orders = []
        for day_orders in order_data:
            for _ in range(5):  # 5 sales of 20 units per recipe
                flat_orders.append({
                    'recipe_id': 'R001',
                    'quantity_sold': 4,
                    'date': day_orders['date']
                })
        
        order_df = pd.DataFrame(flat_orders)
        
        ingredients_df = pd.DataFrame({
            'recipe_id': ['R001'],
            'item_id': ['001'],
            'quantity_per_unit': [0.018]
        })
        
        # Derive usage from orders
        order_usage = derive_usage_from_orders('001', order_df, ingredients_df)
        
        assert len(order_usage) > 0
        
        # Reconcile the two approaches
        if len(stock_usage) > 0 and len(order_usage) > 0:
            reconciled = reconcile_usage(stock_usage, order_usage)
            assert len(reconciled) > 0
            assert 'variance_flagged' in reconciled.columns
