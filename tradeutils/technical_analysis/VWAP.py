
import numpy as np
from tradeutils.type import SequenceType

from pandas import Series
def VWAP(prices: Series,volume:Series,windows:int):
    amount = prices * volume
    vwap = (
    amount.rolling(window=windows, min_periods=1).sum() /
    volume.rolling(window=windows, min_periods=1).sum()
    )
    return vwap




def VWAP_numpy(prices:SequenceType, volume:SequenceType, window:int):
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