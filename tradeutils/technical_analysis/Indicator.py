

import numpy as np
import talib as ta
from mathematics.stata import (corrcoef)
from pandasutils import (SequenceType)
from scipy import stats
def get_corrcoef( datapoint: SequenceType, windows: int = 7):
    slope = np.zeros(len(datapoint))
    x = np.arange(1, windows + 1)
    
    x_norm = (x - 1) / windows
    for i in range(windows - 1, len(datapoint)):
        idx = i + 1
        slope[i] = corrcoef(x=x_norm, y=datapoint[idx-windows:idx])
    return slope


def calculate_trend_score(df):
    close = df['close'].values
    n = len(close)
    signals = []  # 存储40个指标的信号
    
    # 1. Rate of Change (ROC)：10个时间框架
    roc_periods = [24, 32, 48, 64, 96, 128, 192, 256, 384, 512]
    for period in roc_periods:
        if n < period:
            signals.append(0.0)  # 数据不足
            continue
        roc = ta.ROC(close, timeperiod=period)
        # 当前价格 > period天前价格：1.0；否则：-1.0（最后一个有效值）
        current_roc = roc[-1]
        signal = 1.0 if current_roc > 0 else (-1.0 if current_roc < 0 else 0.0)
        signals.append(signal)
    
    # 2. Simple Moving Average (SMA)：10个时间框架
    sma_periods = [24, 32, 48, 64, 96, 128, 192, 256, 384, 512]
    for period in sma_periods:
        if n < period:
            signals.append(0.0)
            continue
        sma = ta.SMA(close, timeperiod=period)
        # 当前价格 > SMA：1.0；否则：-1.0
        current_price = close[-1]
        current_sma = sma[-1]
        signal = 1.0 if current_price > current_sma else (-1.0 if current_price < current_sma else 0.0)
        signals.append(signal)
    
    # 3. SMA交叉系统：10组短期/长期组合（文档图1中的Crossover参数）
    crossover_pairs = [
        (20, 400), (50, 400), (100, 400), (200, 400),
        (20, 200), (50, 200), (100, 200),
        (20, 100), (50, 100),
        (20, 50)
    ]
    for short, long in crossover_pairs:
        if n < long:
            signals.append(0.0)
            continue
        sma_short = ta.SMA(close, timeperiod=short)
        sma_long = ta.SMA(close, timeperiod=long)
        # 短期SMA > 长期SMA：1.0；否则：-1.0
        signal = 1.0 if sma_short[-1] > sma_long[-1] else (-1.0 if sma_short[-1] < sma_long[-1] else 0.0)
        signals.append(signal)
    
    # 4. 线性回归斜率：10个时间框架（月→交易日，假设1月=21天）
    lr_periods = [3*21, 4*21, 5*21, 6*21, 7*21, 8*21, 9*21, 12*21, 15*21, 18*21]
    for period in lr_periods:
        if n < period:
            signals.append(0.0)
            continue
        # 取最近period天的价格进行线性回归
        prices = close[-period:]
        x = np.arange(period)
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, prices)
        # 斜率>0且绝对值>标准误差：1.0；斜率<0且绝对值>标准误差：-1.0
        if abs(slope) > std_err:
            signal = 1.0 if slope > 0 else -1.0
        else:
            signal = 0.0
        signals.append(signal)
    
    # 计算TrendScore（40个信号的平均值）
    trend_score = np.mean(signals)
    return np.clip(trend_score, -1.0, 1.0)  # 确保在[-1,1]范围内



def extended_emotion_index(df):
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    volume = df['volume'].values  # 新增成交量数据（用于MFI等）
    oscillators = []
    
    # 原有指标：RSI和Candle Range（保持不变）
    # ...（省略原文RSI和Candle Range的计算）
    
    # 新增指标1：随机指标（Stochastic）
    stoch_periods = [14, 21]
    for period in stoch_periods:
        slowk, slowd = ta.STOCH(high, low, close, fastk_period=period, slowk_period=3, slowd_period=3)
        adjusted_stoch = 2 * (slowd[-1]/100) - 1  # 用慢线更平滑
        oscillators.append(adjusted_stoch)
    
    # 新增指标2：威廉指标（Williams %R）
    williams_periods = [14, 20]
    for period in williams_periods:
        williams = ta.WILLR(high, low, close, timeperiod=period)
        adjusted_williams = williams[-1] / 100  # 直接转换为[-1,0]
        oscillators.append(adjusted_williams)
    
    # 计算扩展后的情绪指数（16个指标的平均值）
    return np.clip(np.mean(oscillators), -1.0, 1.0)







def hurst_exponent_numpy(price_array, max_lag=20):
    """
    使用numpy数组计算赫斯特指标
    参数:
        price_array: 一维numpy数组，包含价格序列
        max_lag: 最大滞后阶数
    返回:
        hurst: 赫斯特指数
    """
    # 确保输入是numpy数组
    if not isinstance(price_array, np.ndarray):
        raise ValueError("输入必须是numpy数组")
    
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

# 示例使用
if __name__ == "__main__":
    # 生成测试数据
    np.random.seed(42)
    
    # 1. 随机游走序列 (理论H=0.5)
    random_walk = np.cumsum(np.random.randn(1000))
    hurst_random = hurst_exponent_numpy(random_walk)
    
    # 2. 趋势性序列 (理论H>0.5)
    trend = np.cumsum(np.random.randn(1000) + 0.1)  # 加入正向漂移
    hurst_trend = hurst_exponent_numpy(trend)
    
    # 3. 反持续性序列 (理论H<0.5)
    anti_persistent = np.zeros(1000)
    for i in range(1, 1000):
        anti_persistent[i] = -0.8 * anti_persistent[i-1] + np.random.randn()
    hurst_anti = hurst_exponent_numpy(anti_persistent)
    
    print(f"随机游走序列赫斯特指数: {hurst_random:.4f}")
    print(f"趋势性序列赫斯特指数: {hurst_trend:.4f}")
    print(f"反持续性序列赫斯特指数: {hurst_anti:.4f}")