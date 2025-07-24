

from collections.abc import MutableSequence, Sequence
import numpy as np
import talib as ta
from mathematics.stata import corrcoef
from mathematics.type import SequenceType
from scipy import stats
def get_corrcoef( datapoint: SequenceType, windows: int = 7):
    slope = np.zeros(len(datapoint))
    x = np.arange(1, windows + 1)
    
    x_norm = (x - 1) / windows
    for i in range(windows - 1, len(datapoint)):
        idx = i + 1
        slope[i] = corrcoef(x=x_norm, y=datapoint[idx-windows:idx])
    return slope






def roc_periods(close: SequenceType, periods=[24, 32, 48, 64, 96, 128, 192, 256, 384, 512]):
    signals = []
    for period in periods:
        if len(close) < period:
            signals.append(0.0)
            continue
        roc = ta.ROC(close, timeperiod=period)[-1]
        signals.append(1.0 if roc > 0 else (-1.0 if roc < 0 else 0.0))
    return signals

def sma_periods(close: SequenceType, periods=[24, 32, 48, 64, 96, 128, 192, 256, 384, 512]):
    signals = []
    for period in periods:
        if len(close) < period:
            signals.append(0.0)
            continue
        sma = ta.SMA(close, timeperiod=period)[-1]
        signals.append(1.0 if close[-1] > sma else (-1.0 if close[-1] < sma else 0.0))
    return signals

def sma_crossover_periods(close: SequenceType, pairs=[
    (20, 400), (50, 400), (100, 400), (200, 400),
    (20, 200), (50, 200), (100, 200),
    (20, 100), (50, 100), (20, 50)
]):
    signals = []
    for short, long in pairs:
        if len(close) < long:
            signals.append(0.0)
            continue
        sma_short = ta.SMA(close, timeperiod=short)[-1]
        sma_long = ta.SMA(close, timeperiod=long)[-1]
        signals.append(1.0 if sma_short > sma_long else (-1.0 if sma_short < sma_long else 0.0))
    return signals

def linear_regression_periods(close: SequenceType, periods=[3*21, 4*21, 5*21, 6*21, 7*21, 8*21, 9*21, 12*21, 15*21, 18*21]):
    signals = []
    for period in periods:
        if len(close) < period:
            signals.append(0.0)
            continue
        prices = close[-period:]
        slope, _, _, _, std_err = stats.linregress(np.arange(period), prices)
        signals.append(1.0 if abs(slope) > std_err and slope > 0 else 
                      (-1.0 if abs(slope) > std_err else 0.0))
    return signals
def calculate_trend_score(close: SequenceType):
    signals = []
    
    # 1. Rate of Change Signals
    roc_signals = roc_periods(close)
    signals.extend(roc_signals)
    
    # 2. Simple Moving Average Signals
    sma_signals = sma_periods(close)
    signals.extend(sma_signals)
    
    # 3. SMA Crossover Signals
    crossover_signals = sma_crossover_periods(close)
    signals.extend(crossover_signals)
    
    # 4. Linear Regression Slope Signals
    lr_signals = linear_regression_periods(close)
    signals.extend(lr_signals)
    
    return np.clip(np.mean(signals), -1.0, 1.0)
def calculate_adjusted_rsi_oscillators(close: SequenceType,periods=[5, 8, 11, 14, 17, 20]):
    """
    计算6个时间框架的调整后RSI振荡器，范围[-1,1]
    对应论文中情绪指数的RSI成分（图2中rescaled RSI）{insert\_element\_0\_}
    """
    adjusted_rsis = []
    for period in periods:
        if len(close) < period:
            # 数据不足时用0.0填充
            adjusted_rsis.append(0.0)
            continue
        # 计算原始RSI并调整至[-1,1]范围
        rsi = ta.RSI(close, timeperiod=period)[-1]
        adjusted_rsi = 2 * (rsi / 100) - 1  # 论文公式：2.0*RSI - 1.0
        adjusted_rsis.append(adjusted_rsi)
    return adjusted_rsis
def calculate_adjusted_candle_range_oscillators(close: np.ndarray, high: np.ndarray, low: np.ndarray,periods = [3, 6, 9, 12, 15, 18]  ):
    """
    计算6个时间框架的调整后蜡烛范围振荡器，范围[-1,1]
    对应论文中情绪指数的蜡烛范围成分（图2中rescaled CandleRange）{insert\_element\_1\_}
    """
    
    adjusted_candle_ranges = []
    for period in periods:
        if len(close) < period or len(high) < period or len(low) < period:
            # 数据不足时用0.0填充
            adjusted_candle_ranges.append(0.0)
            continue
        # 计算指定时间框架内的最高价和最低价
        min_low = np.min(low[-period:])
        max_high = np.max(high[-period:])
        if max_high == min_low:
            # 避免除零，用0.0填充
            adjusted_candle_ranges.append(0.0)
            continue
        # 计算蜡烛范围并调整至[-1,1]范围
        candle_range = 2 * (close[-1] - min_low) / (max_high - min_low) - 1  # 论文公式
        adjusted_candle_ranges.append(np.clip(candle_range, -1.0, 1.0))  # 确保在范围内
    return adjusted_candle_ranges
def caculate_stochastic_signals(close, high, low, periods=[14, 21]):
    signals = []
    for period in periods:
        slowk, slowd = ta.STOCH(high, low, close, fastk_period=period, slowk_period=3, slowd_period=3)
        signals.append(2 * (slowd[-1]/100) - 1)  # 用慢线更平滑
    return signals

def caculate_williams_signals(close, high, low, periods=[14, 20]):
    signals = []
    for period in periods:
        williams = ta.WILLR(high, low, close, timeperiod=period)
        signals.append(williams[-1] / 100)  # 直接转换为[-1,0]
    return signals

def calculate_emotion_index(close: SequenceType, high: SequenceType, low: SequenceType):
    """
    计算情绪指数：平均两类振荡器的结果，范围[-1,1]
    对应论文公式：EmotionIndex = (1/12) * Σ(oscillator_i){insert\_element\_2\_}
    """
    # 获取两类振荡器的结果
    rsi_oscillators = calculate_adjusted_rsi_oscillators(close)
    candle_oscillators = calculate_adjusted_candle_range_oscillators(close, high, low)
    stochastic_signals = caculate_stochastic_signals(close, high, low)
    
    williams_signals = caculate_williams_signals(close, high, low)
    # 合并所有12个振荡器并取平均
    all_oscillators = rsi_oscillators + candle_oscillators+ stochastic_signals + williams_signals
    return np.clip(np.mean(all_oscillators), -1.0, 1.0)


def calculate_anchored_trend_score(close: np.ndarray, emotion_index: float, emotion_threshold: float = 0.1, current_anchored_trend: float = 0.0):
    """
    计算锚定趋势分数，基于当前趋势分数和情绪指数更新锚定趋势值
    当情绪指数接近0（绝对值≤情绪阈值）时，使用当前趋势分数更新锚定趋势分数，否则保持原有锚定趋势分数
    """
    current_trend = calculate_trend_score(close)  # 调用已实现的趋势分数计算函数
    if abs(emotion_index) <= emotion_threshold:
        return current_trend
    else:
        return current_anchored_trend

def calculate_timing_indicator(anchored_trend: SequenceType| float, emotion_index: SequenceType| float):
    """计算时机指标：锚定趋势分数 - 情绪指数
       大于1.0时做多信号，小于-1.0时表示做空信号
    """
    anchored_trend = np.array(anchored_trend)
    emotion_index = np.array(emotion_index)
    ary= np.clip(anchored_trend - emotion_index, -2.0, 2.0)
    if len(ary)==1:
        return ary[0]
    return ary

def hurst_exponent(price_array:np.ndarray|Sequence|MutableSequence, max_lag=20):
    """
    使用numpy数组计算赫斯特指标
    参数:
        price_array: 一维numpy数组，包含价格序列
        max_lag: 最大滞后阶数
    返回:
        hurst: 赫斯特指数
    """

    
    # 计算收益率序列 (一阶差分)
    returns = np.diff(price_array)
    n = len(returns)
    
    # 生成滞后阶数序列
    lags = np.arange(2, max_lag + 1)
    rs_values = np.zeros(len(lags))
    
    for i, lag in enumerate(lags):
        # 计算可完整划分的子序列数量
        num_sub = n // lag
        total_length = num_sub * lag
        
        # 重塑为二维数组 (子序列数量 x 滞后阶数)
        sub_series = returns[:total_length].reshape(num_sub, lag)
        
        # 计算每个子序列的均值偏差
        mean_sub = np.mean(sub_series, axis=1, keepdims=True)
        deviations = sub_series - mean_sub
        
        # 计算累积偏差
        cumulative = np.cumsum(deviations, axis=1)
        
        # 计算每个子序列的极差 (max - min)
        range_vals = np.max(cumulative, axis=1) - np.min(cumulative, axis=1)
        
        # 计算每个子序列的标准差
        std_vals = np.std(sub_series, axis=1)
        
        # 计算R/S值 (避免除零错误)
        with np.errstate(invalid='ignore'):
            rs = range_vals / std_vals
        rs = rs[~np.isnan(rs)]  # 过滤无效值
        
        # 保存平均R/S值
        rs_values[i] = np.mean(rs)
    
    # 对数转换并进行线性回归
    log_lags = np.log10(lags)
    log_rs = np.log10(rs_values)
    
    # 计算回归斜率 (赫斯特指数)
    hurst = np.polyfit(log_lags, log_rs, 1)[0]
    
    return hurst

