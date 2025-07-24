from tradeutils.technical_analysis.Indicator import hurst_exponent
import numpy as np
from tradeutils.technical_analysis.Indicator import forward_fill



def test_hurst_exponent():
    # 生成测试数据
    np.random.seed(42)
    
    # 1. 随机游走序列 (理论H=0.5)
    random_walk = np.cumsum(np.random.randn(1000))
    hurst_random = hurst_exponent(random_walk)
    
    # 2. 趋势性序列 (理论H>0.5)
    trend = np.cumsum(np.random.randn(1000) + 0.1)  # 加入正向漂移
    hurst_trend = hurst_exponent(trend)
    
    # 3. 反持续性序列 (理论H<0.5)
    anti_persistent = np.zeros(1000)
    for i in range(1, 1000):
        anti_persistent[i] = -0.8 * anti_persistent[i-1] + np.random.randn()
    hurst_anti = hurst_exponent(anti_persistent)
    
    print(f"随机游走序列赫斯特指数: {hurst_random:.4f}")
    print(f"趋势性序列赫斯特指数: {hurst_trend:.4f}")
    print(f"反持续性序列赫斯特指数: {hurst_anti:.4f}")



def test_middle_nans_filled():
    """Test filling NaN values between valid numbers"""
    input_array = np.array([1.0, np.nan, np.nan, 4.0, np.nan, 6.0])
    expected = np.array([1.0, 1.0, 1.0, 4.0, 4.0, 6.0])
    result = forward_fill(input_array)
    np.testing.assert_array_equal(result, expected)

def test_fill_first_true():
    """Test filling leading NaN values when fill_first=True"""
    input_array = np.array([np.nan, np.nan, 3.0, np.nan, 5.0])
    expected = np.array([3.0, 3.0, 3.0, 3.0, 5.0])
    result = forward_fill(input_array, fill_first=True)
    np.testing.assert_array_equal(result, expected)

def test_fill_first_false():
    """Test preserving leading NaN values when fill_first=False"""
    input_array = np.array([np.nan, np.nan, 3.0, np.nan, 5.0])
    expected = np.array([np.nan, np.nan, 3.0, 3.0, 5.0])
    result = forward_fill(input_array)
    np.testing.assert_array_equal(result, expected)

def test_all_nans():
    """Test array with all NaN values"""
    input_array = np.array([np.nan, np.nan, np.nan])
    result = forward_fill(input_array)
    np.testing.assert_array_equal(result, input_array)

def test_empty_array():
    """Test empty array input"""
    input_array = np.array([])
    result = forward_fill(input_array)
    np.testing.assert_array_equal(result, input_array)

def test_no_nans():
    """Test array with no NaN values"""
    input_array = np.array([1.0, 2.0, 3.0])
    result = forward_fill(input_array)
    np.testing.assert_array_equal(result, input_array)

def test_single_nan():
    """Test array with single NaN value"""
    input_array = np.array([1.0, np.nan, 3.0])
    expected = np.array([1.0, 1.0, 3.0])
    result = forward_fill(input_array)
    np.testing.assert_array_equal(result, expected)

def test_different_data_types():
    """Test with different data types"""
    input_array = np.array([1, np.nan, 3], dtype=float)
    expected = np.array([1, 1, 3], dtype=float)
    result = forward_fill(input_array)
    np.testing.assert_array_equal(result, expected)