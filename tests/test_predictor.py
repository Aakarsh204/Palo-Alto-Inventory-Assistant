import pytest
from src.ai.predictor import smooth_usage, predict_stockout, compare_predictions


class TestSmoothUsage:
    """Tests for smooth_usage function"""
    
    def test_smooth_usage_reduces_spike(self):
        """Test that smoothing reduces spikes in usage data"""
        input_data = [2.0, 20.0, 3.0]
        result = smooth_usage(input_data, window=3)
        
        # Assert result has same length
        assert len(result) == 3
        
        # Assert middle value is smoothed (less than spike, more than neighbors)
        assert result[1] < 20.0
        assert result[1] > 2.0
        
        # Assert no NaN values
        assert all(v is not None and not (isinstance(v, float) and v != v) for v in result)
    
    def test_smooth_usage_preserves_length(self):
        """Test that smoothing preserves list length"""
        input_data = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = smooth_usage(input_data, window=3)
        
        assert len(result) == len(input_data)
    
    def test_smooth_usage_min_periods(self):
        """Test smoothing with single value (min_periods=1)"""
        input_data = [5.0]
        result = smooth_usage(input_data, window=3)
        
        # Should return [5.0] without crashing
        assert result == [5.0] or len(result) == 1
        assert result[0] == 5.0
    
    def test_smooth_usage_constant_values(self):
        """Test smoothing with constant usage values"""
        input_data = [5.0, 5.0, 5.0, 5.0, 5.0]
        result = smooth_usage(input_data, window=3)
        
        # All smoothed values should be ~5.0
        assert len(result) == 5
        assert all(abs(v - 5.0) < 0.01 for v in result)
    
    def test_smooth_usage_returns_list(self):
        """Test that smoothing returns a list"""
        input_data = [1.0, 2.0, 3.0]
        result = smooth_usage(input_data, window=2)
        
        assert isinstance(result, list)
    
    def test_smooth_usage_large_window(self):
        """Test smoothing with window larger than data"""
        input_data = [1.0, 2.0]
        result = smooth_usage(input_data, window=5)
        
        # Should not crash, should handle gracefully
        assert isinstance(result, list)
        assert len(result) == 2


class TestPredictStockout:
    """Tests for predict_stockout function"""
    
    def test_predict_stockout_happy_path(self):
        """Test stock prediction with sufficient data"""
        usage_history = [0.4] * 10  # Consistent daily usage
        current_stock = 2.0
        
        result = predict_stockout(usage_history, current_stock)
        
        # Assert result is not None
        assert result is not None
        
        # Assert result is an integer
        assert isinstance(result, int)
        
        # Assert result is reasonable (between 0 and 60)
        assert 0 <= result <= 60
    
    def test_predict_stockout_returns_none_insufficient_data(self):
        """Test that prediction returns None with < 5 days of data"""
        usage_history = [0.4, 0.5, 0.3]  # Only 3 days
        current_stock = 2.0
        
        result = predict_stockout(usage_history, current_stock)
        
        # Should return None for insufficient data
        assert result is None
    
    def test_predict_stockout_zero_stock(self):
        """Test prediction with zero current stock"""
        usage_history = [0.4] * 10
        current_stock = 0.0
        
        result = predict_stockout(usage_history, current_stock)
        
        # Should return None when stock is zero
        assert result is None
    
    def test_predict_stockout_minimum_data(self):
        """Test prediction with exactly 5 days of data (threshold)"""
        usage_history = [0.5] * 5  # Exactly 5 days
        current_stock = 3.0
        
        result = predict_stockout(usage_history, current_stock)
        
        # Should work with exactly 5 days
        assert result is not None
        assert isinstance(result, int)
    
    def test_predict_stockout_high_usage(self):
        """Test prediction with high daily usage"""
        usage_history = [2.0] * 10  # Heavy daily usage
        current_stock = 5.0
        
        result = predict_stockout(usage_history, current_stock)
        
        # Result should be low (stockout soon)
        if result is not None:
            assert result < 10  # Less than 10 days
    
    def test_predict_stockout_low_usage(self):
        """Test prediction with low daily usage"""
        usage_history = [0.1] * 10  # Light daily usage
        current_stock = 5.0
        
        result = predict_stockout(usage_history, current_stock)
        
        # Result should be high (long stock duration)
        if result is not None:
            assert result > 20  # More than 20 days
    
    def test_predict_stockout_negative_usage_ignored(self):
        """Test that negative usage values are handled"""
        usage_history = [0.5, -0.1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
        current_stock = 2.0
        
        # Should not crash with negative values
        result = predict_stockout(usage_history, current_stock)
        
        assert result is None or isinstance(result, int)
    
    def test_predict_stockout_very_high_stock(self):
        """Test prediction with very high stock level"""
        usage_history = [0.5] * 10
        current_stock = 100.0
        
        result = predict_stockout(usage_history, current_stock)
        
        # Result should be capped at display limit (60 days)
        if result is not None:
            assert result <= 60


class TestComparePredictions:
    """Tests for compare_predictions function"""
    
    def test_compare_predictions_high_confidence(self):
        """Test high confidence when predictions are very close"""
        stock_days = 5
        order_days = 6
        
        result = compare_predictions(stock_days, order_days)
        
        # Assert result is a dict
        assert isinstance(result, dict)
        
        # Assert confidence is high (≤2 days difference)
        assert result.get('confidence') == 'high'
        
        # Assert divergence_days is correct
        assert result.get('divergence_days') == 1
    
    def test_compare_predictions_medium_confidence(self):
        """Test medium confidence when predictions differ by 3-5 days"""
        stock_days = 5
        order_days = 8
        
        result = compare_predictions(stock_days, order_days)
        
        assert result.get('confidence') == 'medium'
        assert result.get('divergence_days') == 3
    
    def test_compare_predictions_low_confidence(self):
        """Test low confidence when predictions differ by >5 days"""
        stock_days = 5
        order_days = 15
        
        result = compare_predictions(stock_days, order_days)
        
        assert result.get('confidence') == 'low'
        assert result.get('divergence_days') == 10
    
    def test_compare_predictions_insufficient_data(self):
        """Test confidence when both predictions are None"""
        stock_days = None
        order_days = None
        
        result = compare_predictions(stock_days, order_days)
        
        assert result.get('confidence') == 'insufficient_data'
        assert result.get('stock_days') is None
        assert result.get('order_days') is None
    
    def test_compare_predictions_one_none(self):
        """Test confidence when one prediction is None"""
        stock_days = 5
        order_days = None
        
        result = compare_predictions(stock_days, order_days)
        
        assert result.get('confidence') == 'low'
        assert result.get('stock_days') == 5
        assert result.get('order_days') is None
    
    def test_compare_predictions_other_none(self):
        """Test confidence when the other prediction is None"""
        stock_days = None
        order_days = 7
        
        result = compare_predictions(stock_days, order_days)
        
        assert result.get('confidence') == 'low'
        assert result.get('stock_days') is None
        assert result.get('order_days') == 7
    
    def test_compare_predictions_identical(self):
        """Test high confidence when both predictions are identical"""
        stock_days = 10
        order_days = 10
        
        result = compare_predictions(stock_days, order_days)
        
        assert result.get('confidence') == 'high'
        assert result.get('divergence_days') == 0
    
    def test_compare_predictions_result_structure(self):
        """Test that result has expected keys"""
        result = compare_predictions(5, 7)
        
        assert 'stock_days' in result
        assert 'order_days' in result
        assert 'confidence' in result
        assert 'divergence_days' in result
    
    def test_compare_predictions_with_zero(self):
        """Test predictions with zero days (immediate stockout)"""
        stock_days = 0
        order_days = 0
        
        result = compare_predictions(stock_days, order_days)
        
        # Both zero should have high confidence, 0 divergence
        assert result.get('confidence') == 'high'
        assert result.get('divergence_days') == 0


class TestIntegration:
    """Integration tests combining predictor functions"""
    
    def test_smooth_then_predict(self):
        """Test smoothing usage data then predicting stockout"""
        raw_usage = [0.3, 0.5, 0.8, 2.0, 0.4, 0.5, 0.6, 0.4, 0.5, 0.6]
        current_stock = 4.0
        
        # Smooth the usage
        smoothed = smooth_usage(raw_usage, window=3)
        
        # Predict stockout
        prediction = predict_stockout(smoothed, current_stock)
        
        # Should produce valid prediction
        assert prediction is not None or len(smoothed) < 5
    
    def test_compare_two_methods(self):
        """Test comparing predictions from two different methods"""
        # Simulate stock-event based prediction
        stock_usage = [0.4] * 10
        stock_prediction = predict_stockout(stock_usage, 3.0)
        
        # Simulate order-history based prediction
        order_usage = [0.45] * 10
        order_prediction = predict_stockout(order_usage, 3.0)
        
        # Compare them
        if stock_prediction is not None and order_prediction is not None:
            comparison = compare_predictions(stock_prediction, order_prediction)
            
            assert comparison.get('confidence') in ['high', 'medium', 'low']
            assert comparison.get('divergence_days') >= 0
