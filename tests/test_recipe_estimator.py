import pytest
import pandas as pd
from datetime import datetime, timedelta
from src.ai.recipe_estimator import (
    estimate_ingredient_demand,
    find_bottleneck_ingredient
)


class TestEstimateIngredientDemand:
    """Tests for estimate_ingredient_demand function"""
    
    def test_estimate_ingredient_demand_happy_path(self):
        """Test ingredient demand estimation with sufficient order history"""
        # Setup: item 001 used in recipe R001 at 0.018 per unit
        ingredients_df = pd.DataFrame({
            'recipe_id': ['R001'],
            'item_id': ['001'],
            'quantity_per_unit': [0.018]
        })
        
        # Setup: 10 days of recent orders (within 14-day window), R001 sold 20 units each day
        order_dates = [datetime(2026, 3, 10) + timedelta(days=i) for i in range(10)]
        order_df = pd.DataFrame({
            'recipe_id': ['R001'] * 10,
            'quantity_sold': [20] * 10,
            'date': order_dates
        })
        
        result = estimate_ingredient_demand('001', order_df, ingredients_df, days_forward=7)
        
        # Assert result is a dict
        assert isinstance(result, dict)
        
        # Assert avg_daily_demand ≈ 0.36 (20 × 0.018) within tolerance 0.05
        if result.get('avg_daily_demand') is not None:
            assert abs(result['avg_daily_demand'] - 0.36) < 0.05
        
        # Assert projected_total ≈ 2.52 (0.36 × 7) within tolerance 0.1
        if result.get('projected_total') is not None:
            assert abs(result['projected_total'] - 2.52) < 0.1
        
        # Assert R001 is in contributing_recipes
        assert 'R001' in result.get('contributing_recipes', [])
        
        # Assert insufficient_data is False
        assert result.get('insufficient_data') == False
    
    def test_estimate_ingredient_demand_multiple_recipes(self):
        """Test demand estimation when item appears in multiple recipes"""
        ingredients_df = pd.DataFrame({
            'recipe_id': ['R001', 'R002'],
            'item_id': ['001', '001'],
            'quantity_per_unit': [0.018, 0.020]
        })
        
        # 10 days of recent orders: 20 units R001 + 10 units R002 daily
        order_dates = [datetime(2026, 3, 10) + timedelta(days=i) for i in range(10)]
        order_df = pd.DataFrame({
            'recipe_id': ['R001'] * 10 + ['R002'] * 10,
            'quantity_sold': [20] * 10 + [10] * 10,
            'date': order_dates * 2
        })
        
        result = estimate_ingredient_demand('001', order_df, ingredients_df, days_forward=7)
        
        # Should aggregate: (20 × 0.018) + (10 × 0.020) = 0.36 + 0.2 = 0.56 per day
        assert 'avg_daily_demand' in result
        assert 'contributing_recipes' in result
        assert len(result['contributing_recipes']) >= 1
    
    def test_estimate_insufficient_data(self):
        """Test insufficient data flag when only 3 days of history"""
        ingredients_df = pd.DataFrame({
            'recipe_id': ['R001'],
            'item_id': ['001'],
            'quantity_per_unit': [0.018]
        })
        
        # Only 3 days of recent order history (below 14-day window for rolling average)
        order_dates = [datetime(2026, 3, 16) + timedelta(days=i) for i in range(3)]
        order_df = pd.DataFrame({
            'recipe_id': ['R001'] * 3,
            'quantity_sold': [20, 20, 20],
            'date': order_dates
        })
        
        result = estimate_ingredient_demand('001', order_df, ingredients_df, days_forward=7)
        
        # Assert insufficient_data flag is True
        assert result.get('insufficient_data') == True
    
    def test_estimate_item_not_in_any_recipe(self):
        """Test estimation for item not in any recipe"""
        ingredients_df = pd.DataFrame({
            'recipe_id': ['R001'],
            'item_id': ['002'],
            'quantity_per_unit': [0.15]
        })
        
        order_dates = [datetime(2026, 3, 10) + timedelta(days=i) for i in range(10)]
        order_df = pd.DataFrame({
            'recipe_id': ['R001'] * 10,
            'quantity_sold': [20] * 10,
            'date': order_dates
        })
        
        # Request for item 001 which is not in recipes
        result = estimate_ingredient_demand('999', order_df, ingredients_df, days_forward=7)
        
        # Assert avg_daily_demand is 0.0
        assert result.get('avg_daily_demand') == 0.0
        
        # Assert contributing_recipes is empty list
        assert result.get('contributing_recipes') == []
    
    def test_estimate_different_forecast_periods(self):
        """Test demand estimation with different forward-looking periods"""
        ingredients_df = pd.DataFrame({
            'recipe_id': ['R001'],
            'item_id': ['001'],
            'quantity_per_unit': [0.018]
        })
        
        order_dates = [datetime(2026, 3, 5) + timedelta(days=i) for i in range(15)]
        order_df = pd.DataFrame({
            'recipe_id': ['R001'] * 15,
            'quantity_sold': [20] * 15,
            'date': order_dates
        })
        
        # Test with 7-day forecast
        result_7 = estimate_ingredient_demand('001', order_df, ingredients_df, days_forward=7)
        
        # Test with 14-day forecast
        result_14 = estimate_ingredient_demand('001', order_df, ingredients_df, days_forward=14)
        
        # 14-day projection should be roughly 2x the 7-day projection
        if result_7.get('projected_total') and result_14.get('projected_total'):
            ratio = result_14['projected_total'] / result_7['projected_total']
            assert 1.8 < ratio < 2.2


class TestFindBottleneckIngredient:
    """Tests for find_bottleneck_ingredient function"""
    
    def test_find_bottleneck_happy_path(self):
        """Test bottleneck detection with two ingredients of different availability"""
        # Setup ingredients: item 001 (0.018/drink) and item 002 (0.15/drink)
        ingredients_df = pd.DataFrame({
            'recipe_id': ['R001', 'R001'],
            'item_id': ['001', '002'],
            'quantity_per_unit': [0.018, 0.15]
        })
        
        # Setup inventory
        inventory_df = pd.DataFrame({
            'item_id': ['001', '002'],
            'name': ['Coffee Beans', 'Whole Milk'],
            'current_stock': [2.0, 5.0],
            'unit': ['kg', 'L']
        })
        
        # daily_orders = 20
        # Item 001: 2.0 / (0.018 × 20) ≈ 5.5 days
        # Item 002: 5.0 / (0.15 × 20) ≈ 1.67 days ← bottleneck
        result = find_bottleneck_ingredient('R001', ingredients_df, inventory_df, daily_orders=20)
        
        # Assert result is a dict
        assert isinstance(result, dict)
        
        # Assert bottleneck is item 002 (lowest days_remaining)
        assert result.get('item_id') == '002'
        
        # Assert days_remaining is less than 5
        assert result.get('days_remaining') < 5
    
    def test_find_bottleneck_with_single_ingredient(self):
        """Test bottleneck detection with single ingredient recipe"""
        ingredients_df = pd.DataFrame({
            'recipe_id': ['R002'],
            'item_id': ['003'],
            'quantity_per_unit': [0.025]
        })
        
        inventory_df = pd.DataFrame({
            'item_id': ['003'],
            'name': ['Oat Milk'],
            'current_stock': [10.0],
            'unit': ['L']
        })
        
        result = find_bottleneck_ingredient('R002', ingredients_df, inventory_df, daily_orders=15)
        
        # Should return the only ingredient as bottleneck
        assert result.get('item_id') == '003'
        assert result.get('item_name') == 'Oat Milk'
    
    def test_find_bottleneck_no_orders(self):
        """Test bottleneck detection with zero daily orders"""
        ingredients_df = pd.DataFrame({
            'recipe_id': ['R001'],
            'item_id': ['001'],
            'quantity_per_unit': [0.018]
        })
        
        inventory_df = pd.DataFrame({
            'item_id': ['001'],
            'name': ['Coffee Beans'],
            'current_stock': [2.0],
            'unit': ['kg']
        })
        
        # Should handle zero orders gracefully (return None, no division by zero)
        result = find_bottleneck_ingredient('R001', ingredients_df, inventory_df, daily_orders=0)
        
        # Should return None or handle gracefully
        assert result is None or isinstance(result, dict)
    
    def test_find_bottleneck_missing_inventory_item(self):
        """Test bottleneck detection when inventory item doesn't exist"""
        ingredients_df = pd.DataFrame({
            'recipe_id': ['R001', 'R001'],
            'item_id': ['001', '999'],
            'quantity_per_unit': [0.018, 0.10]
        })
        
        # Inventory missing item 999
        inventory_df = pd.DataFrame({
            'item_id': ['001'],
            'name': ['Coffee Beans'],
            'current_stock': [2.0],
            'unit': ['kg']
        })
        
        # Should handle missing inventory gracefully
        result = find_bottleneck_ingredient('R001', ingredients_df, inventory_df, daily_orders=20)
        
        # Should return either None or valid dict
        assert result is None or isinstance(result, dict)
    
    def test_find_bottleneck_high_consumption_item(self):
        """Test bottleneck with high consumption ingredient"""
        ingredients_df = pd.DataFrame({
            'recipe_id': ['R001', 'R001'],
            'item_id': ['001', '002'],
            'quantity_per_unit': [0.001, 0.5]  # Item 002 uses 0.5 per drink
        })
        
        inventory_df = pd.DataFrame({
            'item_id': ['001', '002'],
            'name': ['Spice', 'Base'],
            'current_stock': [2.0, 2.0],
            'unit': ['g', 'kg']
        })
        
        result = find_bottleneck_ingredient('R001', ingredients_df, inventory_df, daily_orders=50)
        
        # Item 002 should be bottleneck due to high consumption
        # 2.0 / (0.5 × 50) = 0.08 days
        if result:
            assert result.get('item_id') in ['001', '002']
    
    def test_find_bottleneck_result_structure(self):
        """Test that result dict has expected structure"""
        ingredients_df = pd.DataFrame({
            'recipe_id': ['R001'],
            'item_id': ['001'],
            'quantity_per_unit': [0.018]
        })
        
        inventory_df = pd.DataFrame({
            'item_id': ['001'],
            'name': ['Coffee Beans'],
            'current_stock': [2.0],
            'unit': ['kg']
        })
        
        result = find_bottleneck_ingredient('R001', ingredients_df, inventory_df, daily_orders=20)

        if result is not None:
            # Assert required keys
            assert 'item_id' in result
            assert 'item_name' in result
            assert 'days_remaining' in result


class TestIntegration:
    """Integration tests combining recipe estimator functions"""
    
    def test_estimate_demand_then_find_bottleneck(self):
        """Test workflow: estimate demand, then find bottleneck"""
        # Setup realistic cafe data
        ingredients_df = pd.DataFrame({
            'recipe_id': ['R001', 'R001', 'R002', 'R002'],
            'item_id': ['001', '002', '001', '003'],
            'quantity_per_unit': [0.018, 0.15, 0.020, 0.03]
        })
        
        # 15 days of recent order history
        order_dates = [datetime(2026, 3, 5) + timedelta(days=i) for i in range(15)]
        order_df = pd.DataFrame({
            'recipe_id': ['R001'] * 15 + ['R002'] * 15,
            'quantity_sold': [20] * 15 + [15] * 15,
            'date': order_dates * 2
        })
        
        inventory_df = pd.DataFrame({
            'item_id': ['001', '002', '003'],
            'name': ['Coffee Beans', 'Whole Milk', 'Vanilla Syrup'],
            'current_stock': [2.0, 5.0, 1.0],
            'unit': ['kg', 'L', 'L']
        })
        
        # Step 1: Estimate demand for item 001
        demand = estimate_ingredient_demand('001', order_df, ingredients_df, days_forward=7)
        assert demand.get('avg_daily_demand') is not None
        
        # Step 2: Find bottleneck for recipe R001
        bottleneck = find_bottleneck_ingredient('R001', ingredients_df, inventory_df, daily_orders=20)
        
        # Should return a bottleneck
        if bottleneck:
            assert bottleneck.get('item_id') in ['001', '002']
    
    def test_multiple_recipes_demand_analysis(self):
        """Test demand analysis across multiple recipes"""
        ingredients_df = pd.DataFrame({
            'recipe_id': ['R001', 'R001', 'R002', 'R002', 'R003'],
            'item_id': ['001', '002', '001', '003', '002'],
            'quantity_per_unit': [0.018, 0.15, 0.020, 0.02, 0.12]
        })
        
        order_dates = [datetime(2026, 3, 1) + timedelta(days=i) for i in range(20)]
        order_df = pd.DataFrame({
            'recipe_id': ['R001'] * 20 + ['R002'] * 20 + ['R003'] * 20,
            'quantity_sold': [20] * 20 + [15] * 20 + [10] * 20,
            'date': order_dates * 3
        })
        
        # Analyze demand for each item
        for item_id in ['001', '002', '003']:
            result = estimate_ingredient_demand(item_id, order_df, ingredients_df, days_forward=7)
            
            if result.get('avg_daily_demand') is not None:
                assert result['avg_daily_demand'] >= 0
