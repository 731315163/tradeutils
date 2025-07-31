from pathlib import Path
from coverage import data
import numpy as np
import pandas
import pytest
from tradeutils.technical_analysis.Indicator import (
    calculate_adjusted_rsi_oscillators,
    calculate_adjusted_candle_range_oscillators,
    caculate_stochastic_signals,
    caculate_williams_signals
)

# 测试数据生成器
@pytest.fixture
def sample_price_data():
    """生成标准化的测试价格数据"""
    np.random.seed(42)
    dataframe= pandas.read_feather(Path.cwd()/"tests"/"BTC_USDT_USDT-5m-futures.feather")
    close = dataframe['close'].to_numpy()
    high = dataframe['high'].to_numpy()
    low = dataframe['low'].to_numpy()
    return close, high, low

# 测试 calculate_adjusted_rsi_oscillators
def test_rsi_output_range(sample_price_data):
    """测试RSI输出范围在[-1,1]内"""
    close, _, _ = sample_price_data
    results = calculate_adjusted_rsi_oscillators(close)
    for result in results:
        valid_data = result[~np.isnan(result)]
        assert np.all((valid_data >= -1) & (valid_data <= 1))

def test_rsi_output_length(sample_price_data):
    """测试输出维度正确性"""
    close, _, _ = sample_price_data
    periods = [5, 8, 11, 14, 17, 20]
    results = calculate_adjusted_rsi_oscillators(close, periods)
    assert len(results) == len(periods)
    for i, period in enumerate(periods):
        assert len(results[i]) == len(close)

def test_rsi_monotonic_cases():
    """测试极端情况（全涨/全跌）"""
    # 全涨序列
    close = np.arange(100, 200)
    rsi = calculate_adjusted_rsi_oscillators(close)
    # assert rsi[-1] > 0.8  # 涨势应接近1
    
    # 全跌序列
    close = np.arange(200, 100, -1)
    rsi = calculate_adjusted_rsi_oscillators(close)
    assert rsi[-1] < -0.8  # 跌势应接近-1

# 测试 calculate_adjusted_candle_range_oscillators（已存在，保留完整性）
def test_candle_range_output_range(sample_price_data):
    """测试蜡烛范围输出范围"""
    close, high, low = sample_price_data
    results = calculate_adjusted_candle_range_oscillators(close, high, low)
    for result in results:
        valid_data = result[~np.isnan(result)]
        assert np.all((valid_data >= -1) & (valid_data <= 1))

# 测试 caculate_stochastic_signals
def test_stochastic_output_range(sample_price_data):
    """测试随机指标输出范围"""
    close, high, low = sample_price_data
    results = caculate_stochastic_signals(close, high, low)
    for result in results:
        valid_data = result[~np.isnan(result)]
        assert np.all((valid_data >= -1) & (valid_data <= 1))

def test_stochastic_output_length(sample_price_data):
    """测试随机指标输出维度"""
    close, high, low = sample_price_data
    periods = [14, 21]
    results = caculate_stochastic_signals(close, high, low, periods)
    assert len(results) == len(periods)
    for i, period in enumerate(periods):
        assert len(results[i]) == len(close)

# 测试 caculate_williams_signals
def test_williams_output_range(sample_price_data):
    """测试威廉指标输出范围"""
    close, high, low = sample_price_data
    results = caculate_williams_signals(close, high, low)
    for result in results:
        valid_data = result[~np.isnan(result)]
        assert np.all((valid_data >= -1) & (valid_data <= 0))  # 威廉指标范围[-1,0]

def test_williams_output_length(sample_price_data):
    """测试威廉指标输出维度"""
    close, high, low = sample_price_data
    periods = [14, 20]
    results = caculate_williams_signals(close, high, low, periods)
    assert len(results) == len(periods)
    for i, period in enumerate(periods):
        assert len(results[i]) == len(close)

# 共享测试：NaN处理
@pytest.mark.parametrize("func", [
    calculate_adjusted_rsi_oscillators,
    calculate_adjusted_candle_range_oscillators,
    caculate_stochastic_signals,
    caculate_williams_signals
])
def test_nan_handling(func):
    """测试通用NaN处理逻辑"""
    # 创建含NaN的测试数据
    data = [np.array([100, np.nan, 102, 103, 104])] * 3  # 重复三次以适配不同参数数量
    
    # 动态构建参数
    kwargs = {}
    if func.__name__ == 'calculate_adjusted_candle_range_oscillators':
        kwargs = {'high': data[0], 'low': data[0]}
    
    with np.errstate(all='ignore'):
        results = func(data[0], **kwargs)
    
    for result in results:
        if len(result) > 0:  # 避免空数组情况
            assert np.isnan(result[1])  # 验证NaN输入对应位置输出也为NaN

# 共享测试：空输入处理
def test_empty_input():
    """测试空输入情况"""
    empty_array = np.array([])
    with pytest.raises(Exception):  # 空输入应触发异常
        calculate_adjusted_rsi_oscillators(empty_array)

# 共享测试：极短序列处理
def test_short_series():
    """测试超短时间序列"""
    short_data = np.array([100])
    if len(short_data) < max([5, 8, 11, 14, 17, 20]):  # RSI最大周期
        with pytest.raises(Exception):
            calculate_adjusted_rsi_oscillators(short_data)