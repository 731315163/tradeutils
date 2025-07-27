

import numpy as np
import talib as ta
from mathematics.impute import forward_fill
from mathematics.stata import corrcoef
from tradeutils.type import SequenceType
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
    close = np.asarray(close, dtype=np.float64)
    for period in periods:
        roc = ta.ROCR(close, timeperiod=period)
        signals.append(roc)
    return signals

def sma_periods(close: SequenceType, periods=[24, 32, 48, 64, 96, 128, 192, 256, 384, 512]):
    signals = []
    close = np.asarray(close, dtype=np.float64)
    for period in periods:
       
        sma = ta.SMA(close, timeperiod=period)
          # 检查是否有NaN值（close或sma中）
        has_nan =  np.isnan(sma)
        
        # 生成信号：保留NaN，否则根据比较结果赋值1.0/-1.0/0.0
        period_signals = np.where(
            has_nan,
            np.nan,  # 存在NaN时保留NaN
            np.where(close > sma, 1.0,
                     np.where(close < sma, -1.0, 0.0))
        )
        signals.append(period_signals)
    return signals

def sma_crossover_periods(close: SequenceType, pairs=[
    (20, 400), (50, 400), (100, 400), (200, 400),
    (20, 200), (50, 200), (100, 200),
    (20, 100), (50, 100), (20, 50)
]):
    signals = []
    close = np.asarray(close, dtype=np.float64)
    for short, long in pairs:
        
        sma_short = ta.SMA(close, timeperiod=short)
        sma_long = ta.SMA(close, timeperiod=long)
        
        # 检查是否有NaN值（任一SMA存在NaN则结果为NaN）
        has_nan = np.isnan(sma_short) | np.isnan(sma_long)
        
        # 使用np.where实现三重条件判断
        period_signals = np.where(
            has_nan,
            np.nan,  # 存在NaN时保留NaN
            np.where(sma_short > sma_long, 1.0,
                     np.where(sma_short < sma_long, -1.0, 0.0))
        )
        
        signals.append(period_signals)
    return signals

def linear_regression_periods(close: SequenceType, periods=[3*21, 4*21, 5*21, 6*21, 7*21, 8*21, 9*21, 12*21, 15*21, 18*21]):
    signals = []
    close = np.asarray(close, dtype=np.float64)
    for period in periods:
      
        """
        计算标准化后的线性回归斜率
        
        参数:
            close: 收盘价序列
            timeperiod: 计算周期
            
        返回:
            标准化后的斜率列表，每个值都在[-1, 1]之间
        """
        
        
        # 计算原始斜率
        slopes = ta.LINEARREG_SLOPE(close, period)
        
        # 标准化每个斜率值
        normalized = np.tanh(slopes)
        signals.append( normalized)
    return signals

def average_arrays_strict_nan(arrays:list[SequenceType]):
    """
    计算多个等长数组的平均值，只要对应位置有一个值为NaN，结果就为NaN
    
    参数:
        arrays: 数组的列表或元组，所有数组必须长度相同
        
    返回:
        平均值数组，形状与输入数组相同，只要输入中有NaN则对应位置为NaN
    """
    # 将输入的数组列表转换为二维数组（每行一个数组）
    stacked = np.vstack(arrays)
    
    # 检查每个位置是否存在NaN（任何一个数组在该位置为NaN则标记为True）
    has_nan = np.any(np.isnan(stacked), axis=0)
    
    # 计算每一列的总和（忽略NaN）
    sums = np.nansum(stacked, axis=0)
    
    # 计算每一列中非NaN值的数量
    counts = np.sum(~np.isnan(stacked), axis=0)
    
    # 计算平均值（非NaN位置）
    averages = np.divide(sums,counts) 
    
    # 将存在NaN的位置赋值为NaN
    averages[has_nan] = np.nan
    
    return averages
def calculate_trend_score(close: SequenceType):
    
    # 1. Rate of Change Signals
    roc_signals = roc_periods(close)
    
    
    # 2. Simple Moving Average Signals
    sma_signals = sma_periods(close)
    
    # 3. SMA Crossover Signals
    crossover_signals = sma_crossover_periods(close)
    
    # 4. Linear Regression Slope Signals
    lr_signals = linear_regression_periods(close)

    signals = average_arrays_strict_nan(roc_signals+sma_signals+crossover_signals+lr_signals)
    return signals
    return np.clip(signals, -1.0, 1.0)
def calculate_adjusted_rsi_oscillators(close: SequenceType,periods=[5, 8, 11, 14, 17, 20]):
    """
    计算6个时间框架的调整后RSI振荡器，范围[-1,1]
    对应论文中情绪指数的RSI成分（图2中rescaled RSI）{insert\_element\_0\_}
    """
    close = np.asarray(close, dtype=np.float64)
   
    adjusted_rsis = []
    for period in periods:
        # 计算原始RSI并调整至[-1,1]范围
        rsi = ta.RSI(close, timeperiod=period)
        adjusted_rsi = np.where(
            np.isnan(rsi),
            np.nan,
             (rsi / 50) - 1  # 论文公式：2.0*RSI - 1.0
        ) 
        adjusted_rsis.append(adjusted_rsi)
    return adjusted_rsis
def calculate_adjusted_candle_range_oscillators(close: SequenceType, high: SequenceType, low: SequenceType,periods = [3, 6, 9, 12, 15, 18]  ):
    """
    计算6个时间框架的调整后蜡烛范围振荡器，范围[-1,1]
    对应论文中情绪指数的蜡烛范围成分（图2中rescaled CandleRange）{insert\_element\_1\_}
    """
    close = np.asarray(close, dtype=np.float64)
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    adjusted_candle_ranges = []
    for period in periods:
      
        min_low = ta.MIN(low, period)
        max_high = ta.MAX(high, period)
        
        # 向量化计算（避免逐元素操作）
        with np.errstate(divide='ignore', invalid='ignore'):
            candle_range = 2 * (close - min_low) / (max_high - min_low) - 1
            
        # 统一NaN处理（保持与输入长度一致）
        candle_range = np.where(
            (max_high == min_low) | np.isnan(min_low) | np.isnan(max_high),
            np.nan,
            candle_range
        )
        adjusted_candle_ranges.append(candle_range)  # 确保在范围内
    return adjusted_candle_ranges
def caculate_stochastic_signals(close: SequenceType, high: SequenceType, low: SequenceType, periods=[14, 21]):
    signals = []
    close = np.asarray(close, dtype=np.float64)
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    for period in periods:
        slowk, slowd = ta.STOCH(high, low, close, fastk_period=period, slowk_period=3, slowd_period=3)
        signal = np.where(
            np.isnan(slowd),
            np.nan,
            np.divide( slowd , 50) - 1  # 使用慢线更平滑
        )
        signals.append(signal)
    return signals

def caculate_williams_signals(close: SequenceType, high: SequenceType, low: SequenceType, periods=[14, 21]):
    signals = []
    close = np.asarray(close, dtype=np.float64)
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    for period in periods:
        williams = ta.WILLR(high, low, close, timeperiod=period)
        signal = np.where(
            np.isnan(williams),
            np.nan,
            np.divide(williams , 50)+1  # 直接转换为[-1,0]
        )
        signals.append(signal)
    return signals

def calculate_emotion_index(close: SequenceType, high: SequenceType, low: SequenceType):
    """
    计算情绪指数：平均两类振荡器的结果，范围[-1,1]
    calculate_adjusted_rsi_oscillators(close)
    calculate_adjusted_candle_range_oscillators(close, high, low)
    caculate_stochastic_signals(close, high, low)
    对应论文公式：EmotionIndex = (1/12) * Σ(oscillator_i){insert\_element\_2\_}
    """
    # 获取两类振荡器的结果
    rsi_oscillators = calculate_adjusted_rsi_oscillators(close)
    candle_oscillators = calculate_adjusted_candle_range_oscillators(close, high, low)
    stochastic_signals = caculate_stochastic_signals(close, high, low)
    
    williams_signals = caculate_williams_signals(close, high, low)
    # 合并所有12个振荡器并取平均
    all_oscillators =average_arrays_strict_nan( rsi_oscillators + candle_oscillators)
    return np.clip(all_oscillators, -1.0, 1.0)


# def calculate_anchored_trend_index(close: SequenceType, emotion_index: float, emotion_threshold: float = 0.1, pre_anchored_trend: float = 0.0):
#     """
#     计算锚定趋势分数，基于当前趋势分数和情绪指数更新锚定趋势值
#     当情绪指数接近0（绝对值≤情绪阈值）时，使用当前趋势分数更新锚定趋势分数，否则保持原有锚定趋势分数
#     """
#     current_trend = calculate_trend_score(close)  # 调用已实现的趋势分数计算函数
#     if abs(emotion_index) <= emotion_threshold:
#         return current_trend
#     else:
#         return pre_anchored_trend


def calculate_anchored_trend(close: SequenceType, emotion_index: SequenceType, emotion_threshold: float = 0.1):
    """
    计算锚定趋势分数。
    """
    close = np.asarray(close, dtype=np.float64)
    emotion_index = np.array(emotion_index)
  
    
    if close.shape != emotion_index.shape:
        raise ValueError("close和emotion_index必须具有相同的形状")
    
    trends = calculate_trend_score(close)
    ary = np.where(np.abs(emotion_index) <= abs(emotion_threshold), trends, np.nan)
    filled_nan_ary = forward_fill(ary)
    
    return filled_nan_ary
    

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

def hurst_exponent(price_array:SequenceType, max_lag=20):
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

