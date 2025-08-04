import numpy as np
import talib 
from mathematics.impute import forward_fill
from tradeutils.type import SequenceType



def PTR( high: SequenceType, low: SequenceType,period:int):
    if len(high) != len(low) or len(high)< period:
        raise ValueError("high and low must be same length and have at least period elements")
     # 转换为numpy数组便于计算
    high_np = np.asarray(high,dtype=np.float64)
    low_np = np.asarray(low,dtype=np.float64)
    min_low = talib.MIN(low_np, period)
    max_high = talib.MAX(high_np, period)
    return max_high - min_low

def PTRP(high: SequenceType, low: SequenceType, period: int) -> np.ndarray:
    """
    基于周期内最高价和最低价的波动幅度指标（保留原始逻辑，优化标准化方式）
    
    参数:
        high: 最高价序列
        low: 最低价序列
        period: 计算周期（如"周内"则period=5，对应5个交易日）
    
    返回:
        周期内的波动幅度（数值越大，波动越剧烈）
    """
    # 输入校验
    if len(high) != len(low) or len(high) < period:
        raise ValueError("high and low must be same length and have at least period elements")
    
    # 转换为numpy数组便于计算
    high_np = np.asarray(high,dtype=np.float64)
    low_np = np.asarray(low,dtype=np.float64)
    
    # 计算周期内的最高价和最低价（核心逻辑：保留你要的"周期内极值"）
    max_high = talib.MAX(high_np, timeperiod=period)  # 周期内最高价
    min_low = talib.MIN(low_np, timeperiod=period)    # 周期内最低价
    
    # 计算周期内的价格波动范围（最高价 - 最低价）
    price_range = max_high - min_low
    
    # 优化标准化方式：用"周期内平均价格"替代"最低价"作为分母
    # 避免最低价过小导致的数值失真（例如价格接近0时）
    avg_price = (max_high + min_low) / 2  # 周期内平均价格（高低点均值）
    avg_price = np.where(avg_price == 0, 1e-6, avg_price)  # 避免除零错误
    
    # 最终波动幅度：(波动范围 / 周期内平均价格) * 100（转为百分比）
    trp_value = price_range / avg_price
    
    return trp_value