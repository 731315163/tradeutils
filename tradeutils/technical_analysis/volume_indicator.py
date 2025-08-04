
from typing import overload
import numpy as np
from tradeutils.type import SequenceType
import talib
from pandas import Series

@overload
def VWAP(prices: Series,volume:Series,window:int):...


@overload
def VWAP(prices:SequenceType, volume:SequenceType, window:int):...


def VWAP(prices: Series|SequenceType,volume:Series|SequenceType,window:int):
    if isinstance(prices,Series) and isinstance(volume,Series):
         return _VWAP_Series(prices,volume,window) 
    elif isinstance(prices,SequenceType) and isinstance(volume,SequenceType):
        return _VWAP_numpy(prices,volume,window)
    else:
        raise TypeError(f"Invalid input type prices:{type(prices)}, volume:{type(volume)}")


def _VWAP_Series(prices: Series,volume:Series,window:int):
    amount = prices * volume
    vwap = (
    amount.rolling(window=window, min_periods=1).sum() /
    volume.rolling(window=window, min_periods=1).sum()
    )
    return vwap

def _VWAP_numpy(prices:SequenceType, volume:SequenceType, window:int):
    # 转换为 NumPy 数组
    prices = np.array(prices)
    volume = np.array(volume)
    
    n = len(prices)
    assert len(volume) == n, "Length of prices and volume must be the same"

    # 计算每一点的 amount
    amount = prices * volume

    # 计算累计和，并在前面添加一个 0
    cum_amount = np.cumsum(amount)
    cum_volume = np.cumsum(volume)
    cum_amount_padded = np.zeros(n + 1)
    cum_volume_padded = np.zeros(n + 1)
    cum_amount_padded[1:] = cum_amount
    cum_volume_padded[1:] = cum_volume

    # 生成每个窗口的起始索引
    start_indices = np.arange(n) - window + 1
    start_indices = np.maximum(start_indices, 0)

    # 计算窗口内的 amount 和 volume 总和
    sum_amount = cum_amount_padded[np.arange(n) + 1] - cum_amount_padded[start_indices]
    sum_volume = cum_volume_padded[np.arange(n) + 1] - cum_volume_padded[start_indices]

    # 防止除以零，可选：设置除零结果为 NaN
    with np.errstate(divide='ignore', invalid='ignore'):
        vwap = np.true_divide(sum_amount, sum_volume)
        vwap[np.isnan(vwap)] = np.inf  # 或设置为 0、NaN 等，根据需求处理

    return vwap

@overload
def filter_volume(volume:SequenceType,period:int,short_period:int):...
@overload
def filter_volume(volume:SequenceType,period:int):...

def filter_volume(volume:SequenceType,period:int,short_period:int|None= None):
    if period > len(volume): raise ValueError("period must be less than or equal to the length of volume")
     
    if short_period is None:
        average_volume = talib.SMA(volume, timeperiod=period)
        masks = volume > average_volume
        # masks = np.where(np.isnan(average_volume), False, masks)
    else:
        short_volume = talib.SMA(volume, timeperiod=short_period)
        average_volume = talib.SMA(short_volume, timeperiod=period)
        masks = short_volume > average_volume
        # masks = np.where(np.isnan(average_volume), False, masks)
    return masks