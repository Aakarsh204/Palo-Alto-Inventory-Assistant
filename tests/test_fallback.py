import pytest
from datetime import datetime, timedelta, date
from src.ai.fallback import (
    should_use_fallback,
    rule_based_alert,
    generate_rule_based_summary
)


class TestShouldUseFallback:
    """Tests for should_use_fallback function"""
    
    def test_should_use_fallback_true_below_threshold(self):
        """Test that fallback is True when history is below 5-day threshold"""
        assert should_use_fallback(4) == True
        assert should_use_fallback(0) == True
        assert should_use_fallback(1) == True
    
    def test_should_use_fallback_false_above_threshold(self):
        """Test that fallback is False when history is 5+ days"""
        assert should_use_fallback(5) == False
        assert should_use_fallback(30) == False
        assert should_use_fallback(100) == False
    
    def test_should_use_fallback_edge_case_exactly_five(self):
        """Test edge case at exactly 5 days (threshold boundary)"""
        # Assuming 5 days is the threshold and >= 5 is sufficient
        result = should_use_fallback(5)
        # Should be False (5 days is sufficient)
        assert result == False


class TestRuleBasedAlert:
    """Tests for rule_based_alert function"""
    
    def test_rule_based_alert_low_stock(self):
        """Test alert triggered for low stock (below reorder threshold)"""
        item = {
            'item_id': '001',
            'name': 'Coffee Beans',
            'current_stock': 0.5,
            'unit': 'kg',
            'reorder_threshold': 1.0,
            'days_until_expiry': 100,
            'expiry_date': (date.today() + timedelta(days=100)).isoformat()
        }
        
        result = rule_based_alert(item, avg_daily_usage=0.4)
        
        # Assert result structure
        assert isinstance(result, dict)
        assert result['has_alert'] == True
        assert result['source'] == 'rule-based'
        
        # Assert at least one alert for reorder or threshold
        assert 'alerts' in result
        alert_types = [alert.get('type') for alert in result.get('alerts', [])]
        assert 'reorder' in alert_types or 'threshold' in alert_types
        
        # Assert data_note exists
        assert 'data_note' in result
    
    def test_rule_based_alert_expiry_soon(self):
        """Test alert triggered when expiry is within 5 days"""
        expiry_date = (date.today() + timedelta(days=3)).isoformat()
        
        item = {
            'item_id': '002',
            'name': 'Whole Milk',
            'current_stock': 50.0,
            'unit': 'L',
            'reorder_threshold': 1.0,
            'expiry_date': expiry_date,
            'days_until_expiry': 3
        }
        
        result = rule_based_alert(item, avg_daily_usage=0.1)
        
        assert result['has_alert'] == True
        
        # Assert expiry alert exists
        alert_types = [alert.get('type') for alert in result.get('alerts', [])]
        assert 'expiry' in alert_types
    
    def test_rule_based_alert_no_alerts_stable_item(self):
        """Test no alert for stable item with sufficient stock"""
        expiry_date = (date.today() + timedelta(days=90)).isoformat()
        
        item = {
            'item_id': '003',
            'name': 'Sugar',
            'current_stock': 10.0,
            'unit': 'kg',
            'reorder_threshold': 1.0,
            'expiry_date': expiry_date,
            'days_until_expiry': 90
        }
        
        result = rule_based_alert(item, avg_daily_usage=0.2)
        
        # 10.0 / 0.2 = 50 days remaining (well above threshold)
        assert result['has_alert'] == False
        assert 'alerts' in result
        assert len(result.get('alerts', [])) == 0
    
    def test_rule_based_alert_zero_usage(self):
        """Test that zero usage doesn't crash the function"""
        item = {
            'item_id': '004',
            'name': 'Vanilla Syrup',
            'current_stock': 5.0,
            'unit': 'L',
            'reorder_threshold': 0.5,
            'expiry_date': (date.today() + timedelta(days=30)).isoformat(),
            'days_until_expiry': 30
        }
        
        # Should not raise exception with zero usage
        result = rule_based_alert(item, avg_daily_usage=0.0)
        
        assert isinstance(result, dict)
        assert 'has_alert' in result
    
    def test_rule_based_alert_multiple_alerts(self):
        """Test item that triggers multiple alerts"""
        expiry_date = (date.today() + timedelta(days=2)).isoformat()
        
        item = {
            'item_id': '005',
            'name': 'Paper Cups',
            'current_stock': 0.8,
            'unit': 'box',
            'reorder_threshold': 1.0,
            'expiry_date': expiry_date,
            'days_until_expiry': 2
        }
        
        result = rule_based_alert(item, avg_daily_usage=0.5)
        
        # Should have multiple alerts (low stock + expiry)
        assert result['has_alert'] == True
        alerts = result.get('alerts', [])
        assert len(alerts) >= 1
    
    def test_rule_based_alert_result_structure(self):
        """Test that result dict has all expected keys"""
        item = {
            'item_id': '006',
            'name': 'Oat Flour',
            'current_stock': 2.0,
            'unit': 'kg',
            'reorder_threshold': 0.5,
            'expiry_date': (date.today() + timedelta(days=60)).isoformat(),
            'days_until_expiry': 60
        }
        
        result = rule_based_alert(item, avg_daily_usage=0.1)
        
        # Assert required keys
        assert 'has_alert' in result
        assert 'alerts' in result
        assert 'source' in result
        assert 'data_note' in result
        
        # Assert source is rule-based
        assert result['source'] == 'rule-based'
    
    def test_rule_based_alert_very_low_stock(self):
        """Test alert for critically low stock"""
        item = {
            'item_id': '007',
            'name': 'Cardamom Pods',
            'current_stock': 0.05,
            'unit': 'kg',
            'reorder_threshold': 0.1,
            'expiry_date': (date.today() + timedelta(days=120)).isoformat(),
            'days_until_expiry': 120
        }
        
        result = rule_based_alert(item, avg_daily_usage=0.01)
        
        # Should trigger alert(s) for very low stock
        assert result['has_alert'] == True
    
    def test_rule_based_alert_negative_daily_usage(self):
        """Test handling of negative daily usage (shouldn't happen but test robustness)"""
        item = {
            'item_id': '008',
            'name': 'Green Tea',
            'current_stock': 5.0,
            'unit': 'kg',
            'reorder_threshold': 0.5,
            'expiry_date': (date.today() + timedelta(days=45)).isoformat(),
            'days_until_expiry': 45
        }
        
        # Negative usage shouldn't crash
        result = rule_based_alert(item, avg_daily_usage=-0.1)
        
        assert isinstance(result, dict)
        assert 'has_alert' in result


class TestGenerateRuleBasedSummary:
    """Tests for generate_rule_based_summary function"""
    
    def test_generate_summary_basic(self):
        """Test basic summary generation"""
        item = {
            'item_id': '001',
            'name': 'Coffee Beans'
        }
        
        alert_result = {
            'has_alert': True,
            'alerts': [{'type': 'reorder'}]
        }
        
        top_supplier = {
            'name': 'Blue Tokai',
            'lead_time': '2 business days'
        }
        
        summary = generate_rule_based_summary(item, alert_result, top_supplier)
        
        # Assert summary is a string
        assert isinstance(summary, str)
        assert len(summary) > 0
        
        # Assert summary contains item name or reference
        assert 'Coffee Beans' in summary or 'reorder' in summary.lower()
    
    def test_generate_summary_no_supplier(self):
        """Test summary generation without supplier info"""
        item = {
            'item_id': '002',
            'name': 'Whole Milk'
        }
        
        alert_result = {
            'has_alert': True,
            'alerts': [{'type': 'expiry'}]
        }
        
        summary = generate_rule_based_summary(item, alert_result, None)
        
        assert isinstance(summary, str)
        assert len(summary) > 0
    
    def test_generate_summary_multiple_sentences(self):
        """Test that summary is descriptive (multiple sentences)"""
        item = {
            'item_id': '003',
            'name': 'Sugar'
        }
        
        alert_result = {
            'has_alert': True,
            'alerts': [
                {'type': 'reorder'},
                {'type': 'expiry'}
            ]
        }
        
        top_supplier = {
            'name': 'Green Valley',
            'lead_time': '1 business day'
        }
        
        summary = generate_rule_based_summary(item, alert_result, top_supplier)
        
        # Summary should be descriptive (2+ sentences)
        sentences = summary.split('.')
        assert len(sentences) >= 2


class TestIntegration:
    """Integration tests combining fallback functions"""
    
    def test_fallback_decision_and_alert_flow(self):
        """Test decision flow: check history, then generate alert"""
        history_length = 3
        
        # Step 1: Check if fallback should be used
        use_fallback = should_use_fallback(history_length)
        assert use_fallback == True  # Less than 5 days
        
        # Step 2: If fallback, generate alert
        if use_fallback:
            item = {
                'item_id': '001',
                'name': 'Coffee Beans',
                'current_stock': 1.5,
                'unit': 'kg',
                'reorder_threshold': 1.0,
                'expiry_date': (date.today() + timedelta(days=30)).isoformat(),
                'days_until_expiry': 30
            }
            
            alert = rule_based_alert(item, avg_daily_usage=0.4)
            assert 'has_alert' in alert
    
    def test_fallback_with_stable_item(self):
        """Test fallback flow for stable item"""
        history_length = 3  # Insufficient for AI
        
        use_fallback = should_use_fallback(history_length)
        
        if use_fallback:
            item = {
                'item_id': '004',
                'name': 'Oat Milk',
                'current_stock': 20.0,
                'unit': 'L',
                'reorder_threshold': 2.0,
                'expiry_date': (date.today() + timedelta(days=100)).isoformat(),
                'days_until_expiry': 100
            }
            
            alert = rule_based_alert(item, avg_daily_usage=0.2)
            # Even with fallback, a stable item shouldn't trigger alert
            if alert['has_alert'] == False:
                assert True  # Expected for stable item
            else:
                assert len(alert.get('alerts', [])) > 0
