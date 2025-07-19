from pandas import Series

def VWAP(prices: Series,volume:Series,windows:int):
    amount = prices * volume
    vwap = (
    amount.rolling(window=windows, min_periods=1).sum() /
    volume.rolling(window=windows, min_periods=1).sum()
    )
    return vwap