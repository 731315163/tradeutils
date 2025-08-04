import pytest
import numpy as np
from tradeutils.technical_analysis.volatility_indicators import PTRP
def test_PTRP_basic_case():
    """测试基本功能：正常输入下的计算结果是否正确"""
    high = [10, 12, 15, 14, 16]  # 最高价序列
    low = [8, 9, 11, 10, 12]     # 最低价序列
    period = 3
    
    # 手动计算预期结果：
    # 周期3的滑动窗口：
    # 窗口1（索引0-2）：max_high=15, min_low=8 → 波动范围=7 → 平均价格=(15+8)/2=11.5 → 7/11.5*100≈60.8696
    # 窗口2（索引1-3）：max_high=15, min_low=9 → 波动范围=6 → 平均价格=(15+9)/2=12 → 6/12*100=50.0
    # 窗口3（索引2-4）：max_high=16, min_low=10 → 波动范围=6 → 平均价格=(16+10)/2=13 → 6/13*100≈46.1538
    expected = np.array([np.nan, np.nan, (15-8)/((15+8)/2), (15-9)/((15+9)/2), (16-10)/((16+10)/2)])
    
    result = PTRP(high, low, period)
    np.testing.assert_allclose(result, expected, rtol=1e-6)  # 允许微小浮点误差

def test_PTRP_period_one():
    """测试周期为1的情况（每个时刻自身的波动）"""
    high = [5, 8, 10]
    low = [3, 7, 9]
    period = 1
    
    # 每个时刻的波动：(high - low) / ((high+low)/2) * 100
    expected = np.array([(2/4), (1/7.5), (1/9.5)])
    
    result = PTRP(high, low, period)
    np.testing.assert_allclose(result, expected, rtol=1e-6)

def test_PTRP_zero_price():
    """测试价格为0的极端情况（确保无除零错误）"""
    high = [0, 0, 2]
    low = [0, 0, 0]
    period = 2
    
    # 窗口1（0-1）：max_high=0, min_low=0 → 平均价格用1e-6替代 → 0/1e-6*100=0
    # 窗口2（1-2）：max_high=2, min_low=0 → 平均价格=(2+0)/2=1 → (2-0)/1*100=200
    expected = np.array([np.nan, 0.0, 200.0])
    
    result = PTRP(high, low, period)
    np.testing.assert_allclose(result, expected, rtol=1e-6)

def test_PTRP_period_equal_length():
    """测试周期等于序列长度的情况"""
    high = [10, 20, 30]
    low = [5, 15, 25]
    period = 3  # 序列长度为3
    
    # 整个序列的波动：max_high=30, min_low=5 → 范围25 → 平均价格17.5 → 25/17.5*100≈142.857
    expected = np.array([np.nan, np.nan, 142.85714286])
    
    result = PTRP(high, low, period)
    np.testing.assert_allclose(result, expected, rtol=1e-6)

def test_PTRP_invalid_input_length_mismatch():
    """测试high和low长度不匹配的错误处理"""
    high = [1, 2, 3]
    low = [1, 2]  # 长度不匹配
    period = 2
    
    with pytest.raises(ValueError, match="high and low must be same length"):
        PTRP(high, low, period)

def test_PTRP_invalid_input_short_length():
    """测试序列长度小于周期的错误处理"""
    high = [1, 2]
    low = [1, 2]
    period = 3  # 周期3 > 序列长度2
    
    with pytest.raises(ValueError, match="have at least period elements"):
        PTRP(high, low, period)

def test_PTRP_numpy_input():
    """测试输入为numpy数组的兼容性"""
    high = np.array([100, 110, 120, 115])
    low = np.array([90, 95, 105, 100])
    period = 2
    
    expected = np.array([
        np.nan,
        (110-90)/((110+90)/2)*100,  # 20/100*100=20
        (120-95)/((120+95)/2)*100,   # 25/107.5*100≈23.2558
        (115-100)/((115+100)/2)*100  # 15/107.5*100≈13.9535
    ])
    
    result = PTRP(high, low, period)
    np.testing.assert_allclose(result, expected, rtol=1e-6)


# Test data cases
test_cases = [
    # Normal case
    {
        'name': 'normal_case',
        'high': [1, 2, 3, 4, 5],
        'low': [0, 1, 2, 3, 4],
        'period': 2,
        'expected': np.array([np.nan, 200.0, 100.0, 66.66666666666667, 50.0])
    },
    # Zero price range case
    {
        'name': 'zero_price_range',
        'high': [2, 2, 2],
        'low': [2, 2, 2],
        'period': 2,
        'expected': np.array([np.nan, 0.0, 0.0])
    },
    # Edge case - period=1
    {
        'name': 'period_one',
        'high': [1, 3, 2],
        'low': [1, 2, 1],
        'period': 2,
        'expected': np.array([0.0, 33.33333333333333, 50.0])
    }
]

@pytest.mark.parametrize("test_case", test_cases, ids=[tc['name'] for tc in test_cases])
def test_PPTRP_normal_cases(test_case):
    """Test PPTRP with various normal inputs using parameterized testing"""
    result = PTRP(test_case['high'], test_case['low'], test_case['period'])
    np.testing.assert_allclose(result, test_case['expected'], rtol=1e-5, equal_nan=True)

def test_PPTRP_insufficient_length():
    """Test PPTRP with insufficient input length"""
    with pytest.raises(ValueError) as exc_info:
        PTRP([1, 2], [0, 1], 3)
    
    assert "high and low must be same length and have at least period elements" in str(exc_info.value)

def test_PPTRP_length_mismatch():
    """Test PPTRP with mismatched input lengths"""
    with pytest.raises(ValueError) as exc_info:
        PTRP([1, 2, 3], [0, 1], 2)
    
    assert "high and low must be same length and have at least period elements" in str(exc_info.value)

def test_PPTRP_zero_avg_price():
    """Test PPTRP with zero average price scenario"""
    high = [1, 1, 1]
    low = [1, 1, 1]
    period = 2
    expected = np.array([np.nan, 0.0, 0.0])
    
    result = PTRP(high, low, period)
    np.testing.assert_allclose(result, expected, rtol=1e-5, equal_nan=True)

def test_PPTRP_with_numpy_arrays():
    """Test PPTRP with numpy array inputs"""
    high = np.array([1, 2, 3])
    low = np.array([0, 1, 2])
    period = 2
    expected = np.array([np.nan, 200.0, 100.0])
    
    result = PTRP(high, low, period)
    np.testing.assert_allclose(result, expected, rtol=1e-5, equal_nan=True)