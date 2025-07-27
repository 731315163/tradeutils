import math
from typing import Literal

import numpy as np
# 现在你可以导入共享文件夹中的模块了
from tradeutils.volumeprofile import get_analyze_volumeprofiles
from tradeutils.type import SequenceType



# def search_dfindex(
#     time: DatetimeType, df: pd.DataFrame, indexname="date", isclamp: bool = False
# ):
#     """
#     在 DataFrame 中搜索特定时间的索引位置。

#     本函数旨在查找给定时间在 DataFrame 中的插入位置，可以选择性地对结果索引进行边界检查以防止越界。
#     查找插入位置
#     new <= old
#     参数:
#     - time: datetime 类型，指定要搜索的时间。
#     - df: pd.DataFrame 类型，包含时间相关索引或列的数据表。
#     - indexname: 字符串类型，指定 DataFrame 中的时间相关索引或列名，默认为 "date"。
#     - isclamp: 布尔类型，指示是否对结果索引进行边界检查，默认为 False。

#     返回:
#     - int 类型，时间在 DataFrame 中的插入位置索引。
#     """
#     time = timeutil.to_utctz(time=time)

#     if indexname == df.index.name:
#         column = df.index
#     elif indexname in df.columns:
#         column = df[indexname]
#     else:
#         raise ValueError("Invalid indexname: indexname must be in df.columns or index.")
#     column = timeutil.to_utctz(column)
#     idx = cast(int, column.searchsorted(value=time, side="left"))
#     if isclamp:
        
#         idx = np.clip(a=idx, a_min=0, a_max=len(df) - 1)
#     return cast(int, idx)



def pricesgridnum(up_p: int | float, low_p: int | float, interval: int | float):
    return math.ceil((up_p - low_p) / abs(interval))


def get_pricesgrid(low_p: int | float, up_p: int | float, interval: int | float):
    """
    为看涨策略生成价格网格。

    :param up: 最高价格。
    :param low: 最低价格。
    :param interval: 价格间隔。
    :param stablecoin: 稳定币的总价值。
    :return: 包含看涨策略的每个价格点的持仓量。
    """
    if up_p <= low_p:
        raise ValueError(f"Invalid parameters: up_p={up_p}, low_p={low_p}.")
    if interval < 0:
        start_p = up_p
    elif interval > 0:
        start_p = low_p
    else:
        raise ValueError(
            f"Invalid interval:  interval={interval}, interval must be non-zero."
        )

    prices = [start_p]
    num = pricesgridnum(up_p=up_p, low_p=low_p, interval=interval)
    # todo: 如何用新的循环替代< >

    for i in range(0, num):
        start_p += interval
        prices.append(start_p)
    return prices


def getprices(
    side: Literal["long", "short"],
    up_p: int | float,
    low_p: int | float,
    interval: int | float,
):

    if side == "long":
        interval = abs(interval)
    elif side == "short":
        interval = -abs(interval)
    else:
        raise ValueError("Invalid side: side must be 'long' or 'short'.")
    return get_pricesgrid(low_p=low_p, up_p=up_p, interval=interval)


def getgrid(
    stablecoin: int | float,
    up_p: int | float,
    low_p: int | float,
    interval: int | float,
    side: Literal["long", "short"],
):
    """
    Generate a grid of stakes based on the given parameters.
    Parameters:
    stablecoin (int | float): The total amount of stablecoin to be distributed across the grid.
    up_p (int | float): The upper price limit for the grid.
    low_p (int | float): The lower price limit for the grid.
    interval (int | float): The interval between each price point in the grid.
    side (Literal["long", "short"]): The side of the market, either 'long' or 'short'.
    Returns:
    np.ndarray: An array representing the stake distribution across the grid.
    Raises:
    ValueError: If the side parameter is not 'long' or 'short'.
    """
    if side != "long" and side != "short":
        raise ValueError("getgrids : Invalid side: side must be 'long' or 'short'.")
    pricesgrid = getprices(side=side, up_p=up_p, low_p=low_p, interval=interval)

    length = len(pricesgrid)
    interval_stablecoinvalue = stablecoin / (len(pricesgrid) - 1)
    stake_grids = [0.0] * length
    for index in range(length - 2, -1, -1):
        stake_grids[index] = (
            interval_stablecoinvalue / pricesgrid[index]
        ) + stake_grids[index + 1]
    return stake_grids


def get_grid_index(
    cur_p: int | float,
    low_p: int | float,
    high_p: int | float,
    interval: int | float,
    side: Literal["long", "short"],
):
    """
    Calculate the index positions for the current price within the grid.

    :param cur_p: Current price.
    :param low_p: Lowest price in the grid.
    :param high_p: Highest price in the grid.
    :param interval: Price interval between grid levels.
    :param side: Strategy side, either 'long' or 'short'.
    :return: A tuple containing the lower and upper index positions.
    :edge            [ 100 ,        90 ,         80 ,            0 ]
    two side            0,1,
                        0,0

    """
    if side == "long":
        small = math.floor((cur_p - low_p) / abs(interval))
    elif side == "short":
        small = math.floor((high_p - cur_p) / abs(interval))
    else:
        raise ValueError("Invalid side: side must be 'long' or 'short'.")
    # num = pricesgridnum(up_p=high_p, low_p=low_p, interval=interval)
    # small = int(clamp(x=small, min_v=0, max_v=num - 1))
    return small


def get_position(
    stack_grid: SequenceType,
    cur_p: int | float,
    cur_amount: int | float,
    low_p: int | float,
    high_p: int | float,
    interval: int | float,
    side: Literal["long", "short"],
):
    """
    + 加仓，- 减仓
    在stack_grid中
    例如 [100,90,80,60]
    [100,90,80,_]为买入仓位
    [_,90,80,60]为卖出仓位
    当前持有仓位大于大的卖出 ,小于小的买入
    """
    max_length = len(stack_grid) - 1
    # 向下舍入，
    bigstake_idx = get_grid_index(
        cur_p=cur_p, low_p=low_p, high_p=high_p, interval=interval, side=side
    )

    bigstake_idx = max(bigstake_idx, 0)
    smallstake_idx = min(bigstake_idx + 1, max_length)
    bigstake_idx = min(bigstake_idx, max_length)

    # buy amount > sell amount
    bigamount = stack_grid[bigstake_idx]
    smamount = stack_grid[smallstake_idx]
    # [ 100 ,        90 ,         80 ,            0]
    # [      cur > sellindex, buyindex  > cur,  allsell]
    # 大于大的 ，小于小的,100不会被买入，0不会被卖出

    if cur_amount > bigamount:
        return bigamount - cur_amount  # (-)
    elif cur_amount < smamount:
        return smamount - cur_amount  # (+)
    else:
        return 0


# def merge_predict_df(
#     pair: str,
#     time: datetime,
#     informative: pd.DataFrame,
#     predict_df: pd.DataFrame | None,
#     indexname="date",
# ):

#     # 加载预测数据
#     if informative is None or informative.empty:
#         raise ValueError("Invalid parameters: informative cannot be None or empty.")
#     if predict_df is None or predict_df.empty:
#         return informative

#     else:
#         length = len(informative)
#         combined_df = _combined_df(
#             time=time,
#             informative=informative,
#             predict_df=predict_df,
#             indexname=indexname,
#         )

#         # 计算移动长度
#         shift_length = len(combined_df) - length
#         if shift_length <= 0:
#             return combined_df
#         names = [n for n in combined_df.columns if n != indexname]
#         # 移动数据
#         df = dfutil.shift(
#             df=combined_df,
#             names=names,
#             periods=-shift_length,
#             must_include_names=True,
#         )

#         # 删除移动后为null的尾部几行
#         df = df.iloc[:-shift_length]
#         return df


# def reset_predict_df(
#     df: pd.DataFrame,
#     loadf_dir_path: Path,
#     pair: str,
#     time: datetime,
#     timeindex: str,
#     timeframe: timedelta,
# ):
#     pre_df, _ = load_df(
#         dir=loadf_dir_path, pair=pair, pre_time=time, time_frame=timeframe
#     )
#     slice_df = merge_predict_df(
#         pair=pair,
#         time=time,
#         informative=df,
#         predict_df=pre_df,
#         indexname=timeindex,
#     )
#     return slice_df










def get_bband(
    pair: str,
    direction: SequenceType,
    upper: SequenceType,
    lowwer: SequenceType,
    middle: SequenceType,
    threshold=1e-9,
):

    length = len(direction)
    low_bound = np.zeros(length)
    high_bound = np.zeros(length)

    for i in range(0, length):

        cid = direction[i]

        if cid > threshold:
            # high long_peroid
            # low short_peroid fixed
            high_bound[i] = upper[i]
            low_bound[i] = middle[i]
        elif cid < threshold:
            # reverted cid big than threshold
            high_bound[i] = middle[i]
            low_bound[i] = upper[i]
        else:

            high_bound[i] = upper[i]
            low_bound[i] = lowwer[i]
    return upper, lowwer


def get_volumeband (h:SequenceType,l:SequenceType,v:SequenceType,timeperiod:int,pv:tuple):

    upper,lower = get_analyze_volumeprofiles(h,l,v,timeperiod,pv)

    
    
