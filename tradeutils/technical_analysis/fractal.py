from tradeutils.type import SequenceType
import math
import numpy as np
def TopFractal(seq: SequenceType):
    n = len(seq)
    result =np.full(n,False) 
    # 从第2个元素开始遍历（索引2），避免使用未来数据
    for i in range(2, n):
        # 判断前一个元素是否为顶分型（需要当前和前两个数据点）
        if seq[i-1] > seq[i-2] and seq[i-1] > seq[i]:
            result[i-1] = True
    return result

def BottomFractal(seq: SequenceType):
    n = len(seq)
    result =np.full(n,False) 
    # 从第2个元素开始遍历（索引2），避免使用未来数据
    for i in range(2, n):
        # 判断前一个元素是否为底分型（需要当前和前两个数据点）
        if seq[i-1] < seq[i-2] and seq[i-1] < seq[i]:
            result[i-1] = True
    return result


def OverlapFractal(low:SequenceType, high:SequenceType, period=3):
    """
    计算重叠分形指标 (Overlap Fractal)
    
    该指标基于前 period 根K线的最高价和最低价，结合当前K线的高低价，
    判断价格突破方向并计算归一化的强度值。

    参数:
        low: 序列类型，各K线的最低价
        high: 序列类型，各K线的最高价
        close: 序列类型，各K线的收盘价（当前实现中未直接使用，保留用于扩展）
        period: 向前看的K线数量，默认为 3

    返回:
        np.ndarray: 与输入等长的数组，前 period 个值为 NaN，之后每个位置为:
                    - math.inf: 当前高点突破前期高点，且低点跌破前期低点
                    - -math.inf: 当前低点未跌破前期低点，且高点未突破前期高点
                    - 其他: 基于波动距离归一化的突破强度值 (正值表示向上突破，负值表示向下突破)
    """
    n = len(low)

    if len(high) != n:
        raise ValueError("low and high must be the same length")

    result = np.full(n, np.nan)

    for i in range(period, n):

        # 前 period 根K线（不包含当前K线）
        prev_high = np.max(high[i-period:i])
        prev_low = np.min(low[i-period:i])

        h = high[i] - prev_high
        l =   low[i]-prev_low

       
        if np.sign(h) > 0  and np.sign(l) < 0:
            result[i] = math.inf
        elif np.sign(h) < 0 and np.sign(l) > 0:
            result[i] = -math.inf
        else:
            distance = (prev_high - prev_low+high[i]-low[i])/2
            if np.sign(h) > 0:
                result[i] = h / distance
            else:
                result[i] = l / distance
    return result

