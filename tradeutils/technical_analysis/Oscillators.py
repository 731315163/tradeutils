from tradeutils.type import SequenceType
import talib
import numpy as np
def bias(close: SequenceType ,periods=[20, 32, 48, 64, 96, 128, 192, 256, 384, 512]):
    '''
    正向大于0,负向小于0
    '''
    signals = []
    close = np.asarray(close, dtype=np.float64)
    for period in periods:
       
        sma = talib.SMA(close, timeperiod=period)
          # 检查是否有NaN值（close或sma中）
        has_nan =  np.isnan(sma)
        
        # 生成信号：保留NaN，否则根据比较结果赋值1.0/-1.0/0.0
        period_signals = np.where(
            has_nan,
            np.nan,  # 存在NaN时保留NaN
            (close - sma)/sma
        )
        signals.append(period_signals)
    return signals