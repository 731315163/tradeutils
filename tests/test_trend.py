import pytest
import numpy as np

from tradeutils.technical_analysis import (  # 请替换为实际的模块名
    roc_periods,
    sma_periods,
    sma_crossover_periods,
    linear_regression_periods,
    calculate_trend_score, average_arrays_strict_nan
)
@pytest.fixture
def sample_data():
    # Generate synthetic price data
    return np.linspace(100, 200, 500)  # Linearly increasing prices

def generate_test_data(length):
    """生成测试用的收盘价序列"""
    return np.random.randn(length).cumsum() + 100  # 生成随机游走序列

# 测试roc_periods函数
def test_roc_periods_basic():
    """测试roc_periods的基本功能"""
    close = generate_test_data(1000)
    result = roc_periods(close)
    
    # 检查返回类型和长度
    assert isinstance(result, list)
    assert len(result) == 10  # 默认有10个周期
    
    # 检查每个元素的类型和长度
    for roc in result:
        assert isinstance(roc, np.ndarray)
        assert len(roc) == len(close)

def test_roc_periods_custom_periods():
    """测试自定义周期的roc_periods"""
    close = generate_test_data(100)
    custom_periods = [5, 10, 15]
    result = roc_periods(close, custom_periods)
    
    assert len(result) == len(custom_periods)

def test_roc_periods_short_data():
    """测试数据长度短于周期的情况"""
    close = generate_test_data(50)  # 数据长度50
    result = roc_periods(close)  # 包含96等长周期
    
    # 应该能正常运行，只是结果中会有NaN值
    assert isinstance(result, list)
    assert len(result) == 10

# 测试sma_periods函数
def test_sma_periods_basic():
    """测试sma_periods的基本功能"""
    close = generate_test_data(1000)
    result = sma_periods(close)
    
    
    assert len(result) == 10
    

def test_sma_periods_equal_case():
    """测试收盘价等于SMA的情况"""
    # 创建收盘价等于SMA的数据（水平直线）
    close = np.ones(1000) * 100
    result = sma_periods(close)
    
    # 所有信号都应该是0
    for signal in result:
        for s in signal:
            assert s == 0.0 or np.isnan(s)  # 允许NaN值

# 测试sma_crossover_periods函数
def test_sma_crossover_periods_basic():
    """测试sma_crossover_periods的基本功能"""
    close = generate_test_data(1000)
    result = sma_crossover_periods(close)
    
    assert isinstance(result, list)
    assert len(result) == 10  # 默认有10对周期
    
    # 检查每个信号都是-1, 0, 或1
    for signal in result:
        assert signal in (-1.0, 0.0, 1.0)

def test_sma_crossover_custom_pairs():
    """测试自定义周期对的情况"""
    close = generate_test_data(1000)
    custom_pairs = [(10, 20), (30, 60)]
    result = sma_crossover_periods(close, custom_pairs)
    
    assert len(result) == len(custom_pairs)

def test_sma_crossover_equal_case():
    """测试短期SMA等于长期SMA的情况"""
    close = np.ones(1000) * 100  # 水平直线，所有SMA相等
    result = sma_crossover_periods(close)
    
    # 所有信号都应该是0
    for signal in result:
        assert signal == 0.0

# 测试linear_regression_periods函数
def test_linear_regression_periods_basic():
    """测试linear_regression_periods的基本功能"""
    close = generate_test_data(1000)
    result = linear_regression_periods(close)
    
    assert isinstance(result, list)
    assert len(result) == 10  # 默认有10个周期
    
    # 检查每个结果都是在[-1, 1]之间的数组
    for lr in result:
        assert isinstance(lr, np.ndarray)
        assert np.all(lr >= -1.0) and np.all(lr <= 1.0)

def test_linear_regression_rising_trend():
    """测试上升趋势的情况，期望得到正的信号"""
    # 创建明确上升的序列
    close = np.arange(1000)  # 完美上升趋势
    result = linear_regression_periods(close)
    
    # 大多数最新的信号应该是正的
    for lr in result:
        assert lr[-1] > 0  # 最后一个值应该为正

def test_linear_regression_falling_trend():
    """测试下降趋势的情况，期望得到负的信号"""
    # 创建明确下降的序列
    close = np.arange(1000, 0, -1)  # 完美下降趋势
    result = linear_regression_periods(close)
    
    # 大多数最新的信号应该是负的
    for lr in result:
        assert lr[-1] < 0  # 最后一个值应该为负

# 测试calculate_trend_score函数
def test_calculate_trend_score_basic():
    """测试calculate_trend_score的基本功能"""
    close = generate_test_data(1000)
    result = calculate_trend_score(close)
    
    assert isinstance(result, np.ndarray)
    # 检查结果是否在[-1, 1]范围内
    assert np.all(result >= -1.0) and np.all(result <= 1.0)

def test_calculate_trend_score_rising_market():
    """测试明显上升市场的情况，期望得到接近1的分数"""
    close = np.arange(1000)  # 完美上升趋势
    result = calculate_trend_score(close)
    
    # 最后一个分数应该接近1
    assert result[-1] > 0.5

def test_calculate_trend_score_falling_market():
    """测试明显下降市场的情况，期望得到接近-1的分数"""
    close = np.arange(1000, 0, -1)  # 完美下降趋势
    result = calculate_trend_score(close)
    
    # 最后一个分数应该接近-1
    assert result[-1] < -0.5

def test_calculate_trend_score_flat_market():
    """测试平稳市场的情况，期望得到接近0的分数"""
    close = np.ones(1000) * 100  # 平稳市场
    result = calculate_trend_score(close)
    
    # 最后一个分数应该接近0
    assert abs(result[-1]) < 0.1

# 参数化测试：测试不同长度的输入数据
@pytest.mark.parametrize("data_length", [100, 500, 1000, 2000])
def test_different_data_lengths(data_length):
    """测试不同长度的输入数据是否能正确处理"""
    close = generate_test_data(data_length)
    
    # 测试各个函数
    roc_result = roc_periods(close)
    sma_result = sma_periods(close)
    crossover_result = sma_crossover_periods(close)
    lr_result = linear_regression_periods(close)
    score_result = calculate_trend_score(close)
    
    # 基本断言
    assert len(roc_result) == 10
    assert len(sma_result) == 10
    assert len(crossover_result) == 10
    assert len(lr_result) == 10
    assert len(score_result) == data_length

# 参数化测试：测试不同类型的输入
@pytest.mark.parametrize("input_type", [list, tuple, np.ndarray])
def test_different_input_types(input_type):
    """测试不同类型的输入（list, tuple, np.ndarray）是否能正确处理"""
    base_data = generate_test_data(1000)
    close = input_type(base_data)  # 转换为不同类型
    
    # 测试各个函数
    roc_result = roc_periods(close)
    sma_result = sma_periods(close)
    crossover_result = sma_crossover_periods(close)
    lr_result = linear_regression_periods(close)
    score_result = calculate_trend_score(close)
    
    # 基本断言
    assert len(roc_result) == 10
    assert len(sma_result) == 10
    assert len(crossover_result) == 10
    assert len(lr_result) == 10
    assert len(score_result) == 1000
# 测试roc_periods函数
def test_roc_periods():
    # 构造明确趋势的数据
    # 1. 持续上涨数据 (确保ROC为正)
    a = np.linspace(10, 20, 100)  # 从10线性增长到20
    b = np.linspace(20, 0, 100)  # 从10线性增长到20
    close_rising = np.concatenate([a, b])  # 前半段上涨，后半段下跌
    roc_rising = roc_periods(close_rising)
    
   

# 测试sma_periods函数
def test_sma_periods():
    # 1. 价格高于SMA (最近价格上涨)
    close_above = np.concatenate([
        np.full(500, 10.0),  # 前500个数据保持10
        np.linspace(10, 20, 500)  # 后500个数据涨到20
    ])
    sma_above = sma_periods(close_above)
    assert all(signal == 1.0 for signal in sma_above)
    
    # 2. 价格低于SMA (最近价格下跌)
    close_below = np.concatenate([
        np.full(500, 20.0),  # 前500个数据保持20
        np.linspace(20, 10, 500)  # 后500个数据跌到10
    ])
    sma_below = sma_periods(close_below)
    assert all(signal == -1.0 for signal in sma_below)
    
    # 3. 价格等于SMA (平稳)
    close_equal = np.full(1000, 15.0)
    sma_equal = sma_periods(close_equal)
    assert all(signal == 0.0 for signal in sma_equal)

# 测试sma_crossover_periods函数
def test_sma_crossover_periods():
    # 1. 短期SMA高于长期SMA (上涨交叉)
    # 构建近期快速上涨的数据
    close_bullish = np.concatenate([
        np.linspace(10, 15, 800),  # 缓慢上涨
        np.linspace(15, 30, 200)   # 快速上涨，拉动短期SMA
    ])
    crossover_bullish = sma_crossover_periods(close_bullish)
    assert all(signal == 1.0 for signal in crossover_bullish)
    
    # 2. 短期SMA低于长期SMA (下跌交叉)
    close_bearish = np.concatenate([
        np.linspace(30, 15, 800),  # 缓慢下跌
        np.linspace(15, 10, 200)   # 快速下跌，拉动短期SMA
    ])
    crossover_bearish = sma_crossover_periods(close_bearish)
    assert all(signal == -1.0 for signal in crossover_bearish)
    
    # 3. 短期SMA等于长期SMA
    close_flat = np.full(1000, 15.0)
    crossover_flat = sma_crossover_periods(close_flat)
    assert all(signal == 0.0 for signal in crossover_flat)

# 测试linear_regression_periods函数
def test_linear_regression_periods():
    # 1. 明确上涨趋势 (斜率>标准误差)
    close_rising = np.linspace(10, 30, 2000)  # 持续上涨
    lr_rising = linear_regression_periods(close_rising)
    print(lr_rising)
    
    # 2. 明确下跌趋势 (斜率<标准误差)
    close_falling = np.linspace(30, 10, 2000)  # 持续下跌
    lr_falling = linear_regression_periods(close_falling)
    
    
    # 3. 平稳趋势 (斜率接近0，绝对值小于等于标准误差)
    close_flat = np.full(2000, 20.0) + np.random.normal(0, 0.01, 2000)  # 微小波动
    lr_flat = linear_regression_periods(close_flat)
    assert all(signal == 0.0 for signal in lr_flat)
    
    # 4. 测试最小周期数据
    min_period = 3 * 21  # 63
    close_min = np.linspace(10, 15, min_period)  # 刚好满足最小周期
    lr_min = linear_regression_periods(close_min)
    assert len(lr_min) == 10  # 应返回10个信号
def test_output_length():
    """测试输出信号数量与周期数量一致"""
    test_data = np.random.rand(1000)
    periods = [5, 10, 15, 20]
    result = linear_regression_periods(test_data, periods)
    
    # 验证返回的信号数量与周期数量一致
    assert len(result) == len(periods)
    
    # 验证每个信号数组的长度与输入数据长度一致
    for signal in result:
        assert len(signal) == len(test_data)

def test_output_with_constant_input():
    """测试常数输入应返回接近零的斜率"""
    constant_data = np.ones(100)
    periods = [5, 10, 15]
    result = linear_regression_periods(constant_data, periods)
    
    # 对于常数序列，线性回归斜率应接近0
    for signal in result:
        # 允许小数点后6位的误差
        assert np.allclose(signal, np.zeros_like(signal), atol=1e-6)

def test_output_with_linear_input():
    """测试线性输入应返回正斜率"""
    x = np.arange(100).astype(float)
    periods = [5, 10, 15]
    result = linear_regression_periods(x, periods)
    
    # 对于严格线性序列，所有周期都应返回正斜率
    for signal in result:
        # 排除NaN值后检查是否所有值都为正
        valid_values = signal[~np.isnan(signal)]
        assert np.all(valid_values > 0)

def test_output_with_decreasing_input():
    """测试递减输入应返回负斜率"""
    x = np.arange(100, 0, -1).astype(float)
    periods = [5, 10, 15]
    result = linear_regression_periods(x, periods)
    
    # 对于严格递减序列，所有周期都应返回负斜率
    for signal in result:
        # 排除NaN值后检查是否所有值都为负
        valid_values = signal[~np.isnan(signal)]
        assert np.all(valid_values < 0)

def test_output_shape_with_different_periods():
    """测试不同周期下的输出形状"""
    test_data = np.random.rand(500)
    period_options = [
        [20, 40, 60],
        [5, 10],
        [100, 200, 300, 400]
    ]
    
    for periods in period_options:
        result = linear_regression_periods(test_data, periods)
        assert len(result) == len(periods)
        for signal, period in zip(result, periods):
            assert len(signal) == len(test_data)

def test_nan_handling():
    """测试包含NaN值的输入处理"""
    test_data = np.random.rand(100).astype(float)
    # 在中间插入NaN值
    test_data[40:60] = np.nan
    periods = [5, 10, 15]
    
    result = linear_regression_periods(test_data, periods)
    
    for signal in result:
        # 验证NaN值位置对应的输出也为NaN
        np.testing.assert_array_equal(np.isnan(test_data), np.isnan(signal))

def test_empty_input():
    """测试空输入处理"""
    empty_data = np.array([])
    result = linear_regression_periods(empty_data)
    
    # 验证返回空列表
    assert result == []
    
    # 测试非空周期列表但输入为空的情况
    result_with_periods = linear_regression_periods(empty_data, [5, 10, 15])
    assert result_with_periods == []
# 测试calculate_trend_score函数
def test_calculate_trend_score():
    # 1. 强烈上涨趋势 (所有信号应为正)
    close_strong_up = np.linspace(10, 50, 2000)
    score_up = calculate_trend_score(close_strong_up)
    assert 0.0 < score_up <= 1.0  # 应为正值且被限制在1.0以内
    
    # 2. 强烈下跌趋势 (所有信号应为负)
    close_strong_down = np.linspace(50, 10, 2000)
    score_down = calculate_trend_score(close_strong_down)
    assert -1.0 <= score_down < 0.0  # 应为负值且被限制在-1.0以内
    
    # 3. 平衡趋势 (正负信号混合)
    # 创建先涨后跌的数据，形成平衡信号
    close_balanced = np.concatenate([
        np.linspace(10, 30, 1000),  # 上涨阶段
        np.linspace(30, 10, 1000)   # 下跌阶段
    ])
    score_balanced = calculate_trend_score(close_balanced)
    assert -1.0 <= score_balanced <= 1.0  # 应在范围内
    assert abs(score_balanced) < 0.5  # 应接近0

# 参数化测试输入类型兼容性
@pytest.mark.parametrize("input_data", [
    list(range(1, 100)),  # Python列表(整数)
    [float(x) for x in range(1, 100)],  # Python列表(浮点数)
    np.array(range(1, 100), dtype=np.float64),  # NumPy数组
])
def test_input_type_compatibility(input_data):
    # 验证不同输入类型都能被处理而不报错
    roc_periods(input_data)
    sma_periods(input_data)
    sma_crossover_periods(input_data)
    linear_regression_periods(input_data)
    calculate_trend_score(input_data)

# 测试边界情况：数据长度刚好满足最小周期要求
def test_minimum_data_length():
    # 计算所有函数需要的最小数据长度
    min_roc_period = min([24, 32, 48, 64, 96, 128, 192, 256, 384, 512])
    min_sma_period = min([24, 32, 48, 64, 96, 128, 192, 256, 384, 512])
    min_lr_period = min([3*21, 4*21, 5*21, 6*21, 7*21, 8*21, 9*21, 12*21, 15*21, 18*21])
    
    # 刚好满足所有周期要求的数据长度
    min_required_length = max(min_roc_period, min_sma_period, min_lr_period) + 1
    
    # 创建刚好满足要求长度的数据
    close_min = np.linspace(10, 20, min_required_length)
    
    # 验证所有函数都能处理这个长度的数据
    assert len(roc_periods(close_min)) == 10
    assert len(sma_periods(close_min)) == 10
    assert len(sma_crossover_periods(close_min)) == 10
    assert len(linear_regression_periods(close_min)) == 10
    assert isinstance(calculate_trend_score(close_min), float)




def test_average_arrays_with_nan():
    # 示例用法
    a = np.array([1.0, 2.0, np.nan, 4.0])
    b = np.array([5.0, np.nan, 6.0, 7.0])
    c = np.array([np.nan, 8.0, 9.0, 10.0])

    result = average_arrays_strict_nan([a, b, c])
    assert np.allclose(result,[np.nan,np.nan,np.nan,7.0],equal_nan=True)  # 输出: [3.         5.         7.5        7.        ]