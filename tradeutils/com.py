import pandas as pd


def CrossUnder(x: pd.Series, y) -> pd.Series:
    if isinstance(y, pd.Series):
        y_s = y.reindex(x.index)
        prev_x = x.shift(1)
        prev_y = y_s.shift(1)
        return (prev_x >= prev_y) & (x < y_s)
    else:
        prev_x = x.shift(1)
        return (prev_x >= y) & (x < y)

def CrossOver(x: pd.Series, y) -> pd.Series:
    if isinstance(y, pd.Series):
        y_s = y.reindex(x.index)
        prev_x = x.shift(1)
        prev_y = y_s.shift(1)
        return (prev_x <= prev_y) & (x > y_s)
    else:
        prev_x = x.shift(1)
        return (prev_x <= y) & (x > y)