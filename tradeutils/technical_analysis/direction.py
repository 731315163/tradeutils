
import numpy as np
from mathematics import SequenceType
from mathematics.stata import (linregress,
                               slopeR,min_max)

def linear_regression( datapoint: SequenceType, err: SequenceType|None=None, windows: int = 7):
    slope = np.zeros(len(datapoint))
    x = np.arange(1, windows + 1)
    
    x_norm = (x - 1) / windows
    for i in range(windows - 1, len(datapoint)):
        idx = i + 1
        y_normal,_,_ = min_max(datapoint[idx - windows : idx])
        slope[i] = slopeR(x=x_norm, y=y_normal)
    return slope
def Linregress(y:SequenceType,windows=7):
    slopeR = np.zeros(len(y))
    slop = np.zeros(len(y))
    intercept = np.zeros(len(y))
    r = np.zeros(len(y))
    x = np.arange(1, windows + 1)
    x_min_max = min_max(x)
    for i in range(windows - 1, len(y)):
        idx = i + 1
        slope_original, intercept_original,r_value,r_slope = linregress(x=x_min_max, y=y[idx - windows : idx])
        slop[i]=slope_original
        intercept[i]= intercept_original
        r[i]=r_value # type: ignore
        slopeR[i] = r_slope
    return slop,intercept,r,slopeR
def trend_momentum_hlatr(
    high: SequenceType,
    low: SequenceType,
    atr: SequenceType,
    start_from: int = 0,
    Cid=None,
    HV=None,
    LV=None,
    threshold=1e-9,
):
    # very good
    N = len(high)
    if not (N == len(low) == len(atr)):
        raise ValueError("Input sequences must have equal length")
    HT_idx, LT_idx = -1, -1
    if Cid is None or HV is None or LV is None:
        HT_idx = int(np.argmax(high[0:start_from]))
        LT_idx = int(np.argmin(low[0:start_from]))
        HV = high[HT_idx]  # Highest price
        LV = low[LT_idx]  # Lowest price
        Cid = 0
    start_from = max(start_from, 0)
    Directions = np.zeros(N, dtype=float)  # Label vector initialized with zeros
    H_Value = np.zeros(N, dtype=float)
    L_Value = np.zeros(N, dtype=float)
    for i in range(start_from, N):
        xhi = high[i]
        xli = low[i]
        delta = atr[i]

        if Cid > threshold:  # Current trend is up
            if xhi > HV:
                HV, HT_idx = xhi, i
            elif xhi < HV - delta:
                # Label the range from LT+1 to HT as up (inclusive of LT but exclusive of HT)

                # Update lowest price, time, and change trend direction to down
                LV, LT_idx, Cid = xli, i, (xhi - HV) / delta

        elif Cid < -threshold:  # Current trend is down
            if xli < LV:
                LV, LT_idx = xli, i
            elif xli > LV + delta:
                # Label the range from HT+1 to LT as down (inclusive of HT but exclusive of LT)

                HV, HT_idx, Cid = xhi, i, (xli - LV) / delta
        else:
            if xhi > HV + delta:
                HV, HT_idx, Cid = xhi, i, (xhi - HV) / delta
            elif xli < LV - delta:
                LV, LT_idx, Cid = xli, i, (xli - LV) / delta
        H_Value[i] = HV
        L_Value[i] = LV
        Directions[i] = Cid

    return Directions, H_Value, L_Value
# def get_direction(
#     # loadf_dir_path: Path | str,
#     pair: str,
#     df: pd.DataFrame,
#     timeindex: str = "date",
#     high_n="high",
#     low_n="low",
#     omega_n="atr",
#     atr_p=1.25,
#     window: int = 7,
#     threshold=1e-9,
#     load_last: bool = False,
# ):
#     """
#     局部最大值:LocalMaximum
#     局部最小值:LocalMinimum
#     """

#     direction_n, LMax_n, LMin_n = IndicatorName.direction, "LocalMaximum", "LocalMinimum"
#     N = len(df)
#     if window > N or N < 2 or window <= 0:
#         raise ValueError(
#             f"window must be less than N and greater than 2,window:{window},df length:{N}"
#         )
#     if (df.loc[window:, omega_n] * atr_p <= threshold).any():
#         raise ValueError("ATR values must be greater than threshold")
#     timeframe = df[timeindex].iloc[1] - df[timeindex].iloc[0]
#     # loadf_dir_path = pathutil.create_dir(loadf_dir_path)
#     # history_df, history_df_path = load_df(
#     #     dir=loadf_dir_path, pair=pair, time_frame=timeframe, part_filename=direction_n
#     # )

#     df_firstime = timeutil.to_utctz(df[timeindex].iloc[0])
#     df_lastime = timeutil.to_utctz(df[timeindex].iloc[-1])

#     if (
#         history_df.empty
#         or timeutil.to_utctz(history_df[timeindex].iloc[0]) >= df_firstime
#     ):
#         D, H, L = trend_momentum_hlatr(
#             high=df[high_n].to_numpy(),
#             low=df[low_n].to_numpy(),
#             atr=df[omega_n].to_numpy() * atr_p,
#             start_from=window,
#         )

#         history_df = df.copy(deep=True)
#         succ = dfutil.set_timeidx(history_df, timeindex, timezone=timezone.utc)
#         history_df.loc[:, direction_n] = D
#         history_df.loc[:, LMax_n] = H
#         history_df.loc[:, LMin_n] = L
#         dfutil.writepd(df=history_df, p=history_df_path)

#     else:
#         if timeutil.to_utctz(history_df[timeindex].iloc[-1]) < df_lastime:
#             history_df = dfutil.combinefirst_bytime(
#                 timeindex, history_df, df, reset_index=True
#             )
#             if history_df.empty:
#                 raise ValueError(
#                     "Invalid parameters: combinefirst function is missing, history_df cannot be empty."
#                 )
#             hisdory_duplicated = history_df.index.duplicated(keep=False)
#             if hisdory_duplicated.any():
#                 raise ValueError(
#                     f"Duplicate dates found in combined DataFrame: {history_df.index[hisdory_duplicated].unique()}\n"
#                     "Use `df.drop_duplicates(subset='date', keep='first')` to resolve."
#                 )
#             ALLD, ALLH, ALLL = trend_momentum_hlatr(
#                 high=history_df["high"].to_numpy(),
#                 low=history_df["low"].to_numpy(),
#                 atr=(history_df[omega_n] * atr_p).to_numpy(),
#                 start_from=window,
#             )
#             # 正确写法（行范围 pos 到 pos+df_length-1）
#             # 行索引（含 pos 到 pos+df_length-1）  # 列名列表
#             # print(len(D))
#             history_df[direction_n] = ALLD
#             history_df[LMax_n] = ALLH
#             history_df[LMin_n] = ALLL
#             dfutil.writepd(df=history_df, p=history_df_path)
#         dfutil.set_timeidx(history_df, timeindex, timezone.utc)
#         pos = cast(int, history_df.index.get_loc(df_firstime))
#         D = history_df[direction_n].iloc[pos : pos + N].to_numpy()

#     if len(D) != N:
#         raise ValueError(
#             f"Invalid parameters: direction length:{len(D)} be must big than df length {N}."
#         )
#     if pd.isna(D).any():
#         raise ValueError(f"Invalid parameters: na in  {D}.")
#     return D




    