from tradeutils.technical_analysis.Indicator import hurst_exponent
import numpy as np
import talib


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



def test_tanh():
    a = 1000
    b=10
    c= 1
    d= 0.001
    tanh= talib.TANH( np.array([1000,10 ,1,d,np.pi]))
    print(f"tanh(1000) = {tanh[0]}")
    print(f"tanh(10) = {tanh[1]}")
    print(f"tanh(1) = {tanh[2]}")
    print(f"tanh(0.001) = {tanh[3]}")
    print(f"tanh(pi) = {tanh[4]}")
    assert True


# tests/test_indicator.py
import pytest
import numpy as np
from tradeutils.technical_analysis.Indicator import linear_regression_periods

def test_normal_input():
    """测试正常输入数据的处理"""
    close = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]
    periods = [3, 5]
    
    result = linear_regression_periods(close, periods)
    
    # 验证输出结构
    assert len(result) == len(periods)
    for i, period in enumerate(periods):
        assert len(result[i]) == len(close)
        # 验证数值范围在[-1, 1]之间
        assert np.all(result[i] >= -1.0) and np.all(result[i] <= 1.0)

def test_empty_input():
    """测试空输入的处理"""
    with pytest.raises(ValueError):
        linear_regression_periods([], [3, 5])

def test_short_sequence():
    """测试序列长度小于周期的场景"""
    close = [100, 101, 102]
    periods = [5, 10]
    
    result = linear_regression_periods(close, periods)
    
    # 验证输出结构
    assert len(result) == len(periods)
    for i, period in enumerate(periods):
        assert len(result[i]) == len(close)
        # 验证前(period-1)个值为NaN
        assert np.isnan(result[i][:period-1]).all()
        # 验证最后一个值有效
        assert not np.isnan(result[i][-1])

def test_invalid_period():
    """测试无效周期参数的处理"""
    close = [100, 101, 102, 103, 104]
    
    # 测试包含0的周期
    with pytest.raises(ValueError):
        linear_regression_periods(close, [0, 3])

    # 测试包含负数的周期
    with pytest.raises(ValueError):
        linear_regression_periods(close, [-5, 3])
@pytest.mark.parametrize("r", [
    100,
    1,
    10000
])
def test_single_period(r):
    """测试单个周期的处理"""
    close = np.random.rand(100) * r  # 随机生成100个价格
    
    result = linear_regression_periods(close, [10])
    
    # 验证输出结构
    assert len(result) == 1
    assert len(result[0]) == len(close)
    # 验证非NaN值的数量
    valid_values = np.sum(~np.isnan(result[0]))
    assert valid_values == len(close) - 9  # 前9个值应为NaN

@pytest.mark.parametrize("periods", [
    [3, 5, 7],
    [10],
    [20, 30]
])
def test_multiple_period_configs(periods):
    """测试不同的周期配置"""
    close = np.random.rand(200) * 100
    
    result = linear_regression_periods(close, periods)
    
    assert len(result) == len(periods)
    for i, period in enumerate(periods):
        assert len(result[i]) == len(close)
        assert np.all(result[i] >= -1.0) and np.all(result[i] <= 1.0)
        valid_count = np.sum(~np.isnan(result[i]))
        assert valid_count == len(close) - (period - 1)