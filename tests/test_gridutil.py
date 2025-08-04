import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tradeutils.strategies.gridutil import (
    get_grid_index,
    get_pricesgrid,
    get_position,
    getgrid,
 
    pricesgridnum,
 
)


@pytest.mark.parametrize(
    "cur_p, low_p, high_p, interval, side, expected",
    [
        (50, 40, 60, 10, "long", 1),
        (40, 40, 60, 10, "long", 0),
        (60, 40, 60, 10, "long", 2),
        (30, 40, 60, 10, "long", -1),
        (70, 40, 60, 10, "long", 3),
        (50, 40, 60, 10, "short", 1),
        (40, 40, 60, 10, "short", 2),
        (60, 40, 60, 10, "short", 0),
        (30, 40, 60, 10, "short", 3),
        (70, 40, 60, 10, "short", -1),
    ],
)
def test_getindex_long_side_curp_in_between(
    cur_p, low_p, high_p, interval, side, expected
):
    assert (
        get_grid_index(
            cur_p=cur_p, low_p=low_p, high_p=high_p, interval=interval, side=side
        )
        == expected
    )


def test_getindex_invalid_side():
    with pytest.raises(ValueError):
        get_grid_index(50, 40, 60, 10, "invalid")  # type: ignore


def test_gridnum_interval_zero():
    with pytest.raises(ZeroDivisionError):
        pricesgridnum(10, 0, 0)


def test_gridnum_non_number():
    with pytest.raises(TypeError):
        pricesgridnum("a", "b", "c")  # type: ignore


def test_gridnum_interval_zero_float():
    with pytest.raises(ZeroDivisionError):
        pricesgridnum(0.5, 0, 0)


@pytest.mark.parametrize(
    "up, low, interval, expected",
    [
        (-5, -10, 1, 5),
        (0, 0, 1, 0),
        (10, 0, 1, 10),
        (10.5, 0, 1, 11),
        (0.5, 0, 0.1, 5),
        (0, -0.5, 0.1, 5),
        (0, 0, 0.1, 0),
        (0.75, 0.25, 0.1, 5),
        (0.5, -0.5, 1, 1),
    ],
)
def test_gridnum_parametrized(up, low, interval, expected):
    assert pricesgridnum(up_p=up, low_p=low, interval=interval) == expected


def test_get_grid_with_list():
    prices = [5, 4, 3, 2]
    stablecoin = 10
    expected_stake_ary = np.array([2.611113, 1.944444, 1.111111, 0])
    stake_ary = getgrid(
        stablecoin=stablecoin,
        up_p=prices[0],
        low_p=prices[-1],
        interval=1,
        side="short",
    )
    assert np.allclose(
        stake_ary, expected_stake_ary, atol=1e-6
    ), "Stake array calculation is incorrect"


def test_get_grid_with_numpy_array():
    prices = np.array([5, 4, 3, 2], dtype=float)
    stablecoin = 10
    expected_stake_ary = np.array([2.611113, 1.944444, 1.111111, 0])
    stake_ary = getgrid(
        stablecoin=stablecoin,
        up_p=prices[0],
        low_p=prices[-1],
        interval=1,
        side="short",
    )
    assert np.allclose(
        stake_ary, expected_stake_ary, atol=1e-6
    ), "Stake array calculation is incorrect"


def test_get_grid_with_tuple():
    prices = (2, 3, 4, 5)
    stablecoin = 10
    expected_stake_ary = np.array([3.611111, 1.94444, 0.83333333, 0])
    stake_ary = getgrid(
        stablecoin=stablecoin,
        up_p=prices[-1],
        low_p=prices[0],
        interval=1,
        side="long",
    )
    assert np.allclose(
        stake_ary, expected_stake_ary, atol=1e-6
    ), "Stake array calculation is incorrect"


def test_get_grid_edge_case():
    """Tests edge case with only two prices."""
    prices = [2, 1]
    stablecoin = 10
    expected_stake_ary = np.array([5, 0])
    stake_ary = stake_ary = getgrid(
        stablecoin=stablecoin,
        up_p=prices[0],
        low_p=prices[-1],
        interval=1,
        side="short",
    )
    assert np.allclose(stake_ary, expected_stake_ary), "Edge case calculation failed"


def test_get_shortgrid_increasing_prices():
    # 测试是否在上升趋势中生成正确的价格网格
    up_p = 100.5
    low_p = 90.5
    interval = -10.0
    expected = [100.5, 90.5]
    prices = get_pricesgrid(low_p=low_p, up_p=up_p, interval=interval)
    assert (
        prices == expected
    ), "The prices should be in descending order with the given interval."


def test_get_shortgrid_decreasing_prices():
    # 测试是否在下降趋势中生成正确的价格网格
    up_p = 90.5
    low_p = 10.5
    interval = -10.0
    expected = [90.5, 80.5, 70.5, 60.5, 50.5, 40.5, 30.5, 20.5, 10.5]
    prices = get_pricesgrid(low_p=low_p, up_p=up_p, interval=interval)
    assert (
        prices == expected
    ), "The prices should be in descending order with the given interval."


def test_get_shortgrid_negative_interval():
    # 测试负间隔是否被转换为正间隔
    up_p = 100.5
    low_p = 90.5
    interval = -10.0
    expected = [100.5, 90.5]
    prices = get_pricesgrid(up_p=up_p, low_p=low_p, interval=interval)
    assert prices == expected, "The interval should be treated as positive."


def test_get_shortgrid_invalid_input_1():
    # 测试无效输入，如非数字或不合理的参数
    with pytest.raises(TypeError):
        get_pricesgrid("a", 10.5, 1.0)  # type: ignore

    with pytest.raises(ValueError):
        get_pricesgrid(up_p=100.5, low_p=10.5, interval=0.0)

    with pytest.raises(ValueError):
        get_pricesgrid(up_p=10.5, low_p=100.5, interval=10.0)


def test_get_shortgrid_invalid_input():
    # 测试无效输入，如非数字或不合理的参数
    with pytest.raises(TypeError):
        get_pricesgrid("a", 10.5, 1.0)  # type: ignore

    with pytest.raises(ValueError):
        get_pricesgrid(up_p=100.5, low_p=10.5, interval=0.0)

    with pytest.raises(ValueError):
        get_pricesgrid(up_p=10.5, low_p=100.5, interval=10.0)


def test_valid_parameters():
    # 测试有效的参数
    low = 10
    high = 20
    interval = 2
    expected = [10, 12, 14, 16, 18, 20]
    result = get_pricesgrid(low, high, interval)
    assert result == expected


def test_zero_interval():
    # 测试零间隔应该引发错误
    low = 10
    high = 20
    interval = 0
    with pytest.raises(ValueError):
        get_pricesgrid(low, high, interval)


def test_reversed_parameters():
    # 测试反转的参数（上限 < 下限）应该引发错误
    low = 10
    high = 5
    interval = 2
    with pytest.raises(ValueError):
        get_pricesgrid(low, high, interval)


def test_boundary_conditions():
    # 测试边界条件
    low = 0
    high = 1
    interval = 0.1
    expected = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    result = get_pricesgrid(low, high, interval)
    assert np.allclose(result, expected, atol=1e-6)


def test_get_grid_with_large_stablecoin():
    prices = [5, 4, 3, 2]
    stablecoin = 1e6
    stake_ary = getgrid(
        stablecoin=stablecoin,
        up_p=prices[0],
        low_p=prices[-1],
        interval=1,
        side="short",
    )
    assert len(stake_ary) == len(
        prices
    ), "Stake array length mismatch for large stablecoin"


@pytest.mark.parametrize(
    "stack_arr, cur_p, cur_amount, low_p, high_p, interval,  expected",
    [
        (  # 1
            [100, 90, 80, 70],
            # [90,95,100,105,110]
            95,  # 当前价格
            80,  # 当前持有的数量
            90,  # 最低价
            110,  # 最高价
            5,  # 间隔
            0,  # 期望的结果
        ),  # 当前为80, 卖出在sellindex=2（80），买入在buyindex=1（90）
        (  # 2
            [100, 90, 80, 70],
            # [90,95,100,105,110]
            94,
            99,
            90,
            110,
            5,
            0,
        ),  # 当前为100, 卖出在sellindex=2（80），买入在buyindex=1（90）
        (  # 3
            [100, 90, 80, 70],
            40,
            100.5,
            90,
            110,
            5,
            -0.5,
        ),  # 当前为40, 卖出在sellindex=0（100），买入在buyindex=0（100）
        (  # 4
            [100, 90, 80, 70],
            # 0,1
            40,
            10,
            90,
            110,
            5,
            80,
        ),  # 当前为40, 卖出在sellindex=0（100），买入在buyindex=0（100）
        (  # 5
            [100, 90, 80, 70, 60],
            # [90,95,100,105,110]
            113.5,  # 当前价格
            100,  # 当前持有的数量
            90,  # 最低价
            110,  # 最高价
            5,  # 间隔
            -40,  # 期望的结果
        ),  # 当前为100, 卖出在sellindex=2（80），买入在buyindex=1（90）
        # stack,90    95   100   105  110 price, stake, low_p,high_p, interval
        (  # 6
            [100.5, 90.2, 80.3, 70.1, 0],
            # [90,95 , 100,105,110]
            94.7,
            85.0,
            90.0,
            110.0,
            5.0,
            5.2,
        ),  # 当前为85.0, 卖出在sellindex=2（80.3），买入在buyindex=1（90.2）
        (
            [100.5, 90.2, 80.3, 70.1, 0],
            95.5,
            100.0,
            90.0,
            110.0,
            5.0,
            -9.8,
        ),  # 当前为100.0, 卖出在sellindex=2（80.3），买入在buyindex=1（90.2）
        (
            [100.5, 90.2, 80.3, 70.1, 0],
            101.5,
            105.0,
            90.0,
            110.0,
            5.0,
            -24.7,
        ),  # 当前为105.0, 卖出在sellindex=2（80.3），买入在buyindex=1（90.2）
    ],
)
def test_long_get_stakeamount(
    stack_arr, cur_p, cur_amount, low_p, high_p, interval, expected
):
    result = get_position(
        stack_grid=stack_arr,
        cur_p=cur_p,
        cur_amount=cur_amount,
        low_p=low_p,
        high_p=high_p,
        interval=interval,
        side="long",
    )
    assert math.isclose(result, expected)


@pytest.mark.parametrize(
    "stack_arr, cur_p, cur_amount, low_p, high_p, interval,  expected",
    [
        (
            [100, 90, 80, 70, 60],
            # [110,105,100,95,90]
            106,
            80,
            90,
            110,
            5,
            10,
        ),  # 当前为80, 卖出在sellindex=1（90），买入在buyindex=0（100）
        (
            [100, 90, 80, 70],
            105,
            100,
            90,
            110,
            5,
            -10,
        ),  # 当前为100, 卖出在sellindex=1（90），买入在buyindex=0（100）
        # stack,110    105   100   95  90 price, stake, low_p,high_p, interval
        (
            [100.5, 90.2, 80.3, 70.1, 0],
            106.7,
            80.0,
            90.0,
            110.0,
            5.0,
            10.2,
        ),  # 当前为80.0, 卖出在sellindex=1（90.2），买入在buyindex=0（100.5）
        (
            [100.5, 90.2, 80.3, 70.1, 0],
            106.7,
            95.0,
            90.0,
            110.0,
            5.0,
            0,
        ),  # 当前为95.0, 卖出在sellindex=1（90.2），买入在buyindex=0（100.5）
        (
            [100.5, 90.2, 80.3, 70.1, 0],
            100.0,
            100.0,
            90.0,
            110.0,
            5.0,
            -19.7,
        ),  # 当前为100.0, 卖出在sellindex=1（90.2），买入在buyindex=0（100.5）
        # stack_arr, cur_p, cur_amount, low_p, high_p, interval,  expected
        ([100.5, 90.2, 80.3, 70.1, 0], 119.0, 100.0, 90.0, 110.0, 5.0, 0),
    ],
)
def test_short_get_stakeamount(
    stack_arr, cur_p, cur_amount, low_p, high_p, interval, expected
):
    result = get_position(
        stack_grid=stack_arr,
        cur_p=cur_p,
        cur_amount=cur_amount,
        low_p=low_p,
        high_p=high_p,
        interval=interval,
        side="short",
    )
    assert math.isclose(result, expected)


def test_edge_cases():
    stack_arr = [100, 90, 80, 70]
    longprice = [90, 95, 100, 105, 110]
    shortprice = [110, 105, 100, 95, 90]
    # cur_amount == sellamount
    assert (
        get_position(
            stack_grid=stack_arr,
            cur_p=95,
            cur_amount=80,
            low_p=90,
            high_p=110,
            interval=5,
            side="long",
        )
        == 0
    )  # 无需变化

    # cur_amount == buyamount
    assert get_position(stack_arr, 95, 90, 90, 110, 5, "long") == 0  # 无需变化

    # cur_amount == sellamount && buyamount
    assert (
        get_position(stack_arr, 95.00000001, 80, 90, 110, 5, "short") == 0
    )  # 无需变化


def test_get_pricesgrid_valid_parameters():
    low = 10
    high = 20
    interval = 2
    expected = [10, 12, 14, 16, 18, 20]
    result = get_pricesgrid(low_p=low, up_p=high, interval=interval)
    assert result == expected


def test_get_pricesgrid_zero_interval():
    low = 10
    high = 20
    interval = 0
    with pytest.raises(ValueError):
        get_pricesgrid(low_p=low, up_p=high, interval=interval)


def test_get_pricesgrid_reversed_parameters():
    low = 10
    high = 5
    interval = 2
    with pytest.raises(ValueError):
        get_pricesgrid(low_p=low, up_p=high, interval=interval)


def test_get_pricesgrid_negative_interval():
    low = 10
    high = 20
    interval = -2
    expected = [20, 18, 16, 14, 12, 10]
    result = get_pricesgrid(low_p=low, up_p=high, interval=interval)
    assert result == expected


def test_get_pricesgrid_boundary_conditions():
    low = 0
    high = 1
    interval = 0.1
    expected = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    result = get_pricesgrid(low_p=low, up_p=high, interval=interval)
    assert np.allclose(result, expected, atol=1e-6)


def test_get_pricesgrid_large_interval():
    low = 10.1
    high = 50
    interval = 20
    expected = [10.1, 30.1, 50.1]
    result = get_pricesgrid(low_p=low, up_p=high, interval=interval)
    assert np.allclose(result, expected, atol=1e-6)


# @pytest.mark.parametrize(
#     "time,data,result",
#     [
#         (
#             "2023-01-02",
#             ["2023-01-01", "2023-01-02", "2023-01-03"],
#             1,
#         ),
#         (
#             "2023-01-02",
#             ["2023-01-01", "2023-01-02", "2023-01-03"],
#             1,
#         ),
#         (
#             "2023-01-02",
#             ["2023-01-01", "2023-01-03"],
#             1,
#         ),
#         ("2023-01-01", ["2023-01-02", "2023-01-03"], 0),
#         ("2023-01-03", ["2023-01-01", "2023-01-02"], 2),
#         ("2023-01-01", [], 0),
#         ("2023-01-01", ["2023-01-01"], 0),
#     ],
# )
# def test_clampfalse_search_dfindex(time, data, result):
#     df = pd.DataFrame({"date": pd.to_datetime(data)})
#     assert search_dfindex(time=time, df=df, isclamp=False) == result


# @pytest.mark.parametrize(
#     "time,data,result",
#     [
#         (
#             "2023-01-02",
#             ["2023-01-01", "2023-01-02", "2023-01-03"],
#             1,
#         ),
#         (
#             "2023-01-02",
#             ["2023-01-01", "2023-01-02", "2023-01-03"],
#             1,
#         ),
#         (
#             "2023-01-02",
#             ["2023-01-01", "2023-01-03"],
#             1,
#         ),
#         ("2023-01-01", ["2023-01-02", "2023-01-03"], 0),
#         ("2023-01-03", ["2023-01-01", "2023-01-02"], 1),
#         ("2023-01-01", [], 0),
#         ("2023-01-01", ["2023-01-01"], 0),
#     ],
# )
# def test_clamptrue_search_dfindex(time, data, result):
#     df = pd.DataFrame({"date": pd.to_datetime(data)})
#     assert search_dfindex(time=time, df=df, isclamp=True) == result


# def test_search_dfindex_with_custom_index_name():
#     df = pd.DataFrame(
#         {"timestamp": pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03"])}
#     )
#     time = "2023-01-02"
#     assert search_dfindex(time=pd.to_datetime(time), df=df, indexname="timestamp") == 1


# def test_adjust_datetime_midnight_should_add_one_minute():
#     dt = datetime(2023, 10, 1, 0, 0, 0)
#     expected = datetime(2023, 10, 1, 0, 1, 0, tzinfo=timezone.utc)
#     result = adjust_datetime(dt)
#     assert result == expected, f"Expected {expected}, but got {result}"


# def test_adjust_datetime_not_midnight_should_return_same():
#     dt = datetime(2023, 10, 1, tzinfo=timezone.utc)
#     expected = datetime(2023, 10, 1, 0, 1, 0, tzinfo=timezone.utc)
#     result = adjust_datetime(dt)
#     assert result == expected, f"Expected {expected}, but got {result}"


# def test_adjust_datetime_midnight_dif_timezone():
#     dt = datetime(2023, 10, 1, tzinfo=timezone(timedelta(hours=0)))
#     expected = datetime(2023, 10, 1, 0, 1, 0, tzinfo=timezone.utc)
#     result = adjust_datetime(dt)
#     assert result == expected, f"Expected {expected}, but got {result}"


# def test_adjust_datetime_midnight_diff_timezone():
#     dt = datetime(2023, 10, 1, tzinfo=timezone(timedelta(hours=1)))
#     expected = datetime(2023, 10, 1, 0, 1, 0, tzinfo=timezone.utc)
#     result = adjust_datetime(dt)
#     assert result != expected, f"Expected {expected}, but got {result}"


# def test_adjust_datetime_slightly_after_midnight_should_return_same():
#     dt = datetime(2023, 10, 1, 0, 0, 1, tzinfo=timezone.utc)
#     expected = datetime(2023, 10, 1, 0, 0, 1, tzinfo=timezone.utc)
#     result = adjust_datetime(dt)
#     assert result == expected, f"Expected {expected}, but got {result}"


# @pytest.fixture
# def resample_dataframe():
#     # 准备数据
#     data = {
#         "date": pd.date_range(start="2023-01-01", periods=10, freq="5min"),
#         "open": np.arange(10),
#         "high": np.arange(10) + 2,
#         "low": np.arange(10) - 2,
#         "close": np.arange(10) + 1,
#         "volume": np.arange(10) * 10,
#     }
#     return pd.DataFrame(data)


# def test_resample_DefaultTimeframe_10Min(resample_dataframe):
#     print(resample_dataframe)
#     # 重新采样
#     resampled_df = resample(resample_dataframe)
#     print(resampled_df)
#     # 验证结果
#     expected_data = {
#         "date": pd.date_range(start="2023-01-01", periods=5, freq="10min"),
#         "open": np.array([0, 2, 4, 6, 8]),
#         "high": np.array([3, 5, 7, 9, 11]),
#         "low": np.array([-2, 0, 2, 4, 6]),
#         "close": np.array([2, 4, 6, 8, 10]),
#         "volume": np.array([10, 50, 90, 130, 170]),
#     }

#     expected_df = pd.DataFrame(expected_data).set_index("date")
#     expected_df = expected_df.asfreq("10min")
#     pd.testing.assert_frame_equal(resampled_df, expected_df)


@pytest.fixture
def comb_data():
    """返回一个包含示例数据的 DataFrame"""
    dates = pd.date_range(start="2023-01-01", periods=10, freq="D")
    data = pd.DataFrame(data={"value": range(10), "date": dates})
    return data


@pytest.fixture
def predict_data():
    """返回一个包含示例数据的 DataFrame"""
    predict_data = pd.DataFrame(
        data={
            "date": pd.date_range(start="2023-01-10", periods=5, freq="D"),
            "value": range(10, 15),
        },
    )
    return predict_data





@pytest.fixture
def setup_data():

    informative = pd.DataFrame(
        {
            "date": pd.date_range(start="2023-01-01 10:00", periods=5, freq="h"),
            "value": ["1", "2", "3", "4", "5"],
        }
    ).set_index("date")
    predict_df = pd.DataFrame(
        {
            "date": pd.date_range(start="2023-01-01 14:00", periods=3, freq="h"),
            "value": ["a", "b", "c"],
        }
    ).set_index("date")
    return informative, predict_df


# def test_merge_predict_df_predict_df_none_or_empty_returns_informative(setup_data):
#     informative, _ = setup_data
#     t = datetime(2023, 1, 1, 14, 0)
#     # Test with predict_df=None
#     result_none = merge_predict_df("pair", t, informative, None)
#     pd.testing.assert_frame_equal(result_none, informative)

#     # Test with predict_df empty
#     result_empty = merge_predict_df("pair", t, informative, pd.DataFrame())
#     pd.testing.assert_frame_equal(result_empty, informative)


# def test_merge_predict_df_informative_empty_returns_empty(setup_data):
#     # informative is None or empty
#     _, predict_df = setup_data
#     t = datetime(2023, 1, 1, 14, 0)
#     with pytest.raises(ValueError):
#         result = merge_predict_df("pair", t, pd.DataFrame(), predict_df)


# def test_merge_predict_df_overlap(setup_data):
#     informative, predict_df = setup_data

#     t = datetime(2023, 1, 1, 15, 10)
#     # 可以通过time来延展数组，返回的df未必与informative的长度相同
#     result = merge_predict_df(
#         "pair", time=t, informative=informative, predict_df=predict_df
#     )
#     print(result)
#     df = pd.DataFrame(
#         {
#             "date": pd.date_range(start="2023-01-01 10:00", periods=5, freq="h"),
#             "value": ["3", "4", "5", "b", "c"],
#         }
#     )
#     dfutil.setindex(df, "date")
#     pd.testing.assert_frame_equal(result, df)


# def test_merge_predict_df_big(setup_data):
#     informative, predict_df = setup_data
#     t = datetime(2023, 1, 1, 15, 0)
#     result = merge_predict_df("pair", t, informative, predict_df)
#     print(result)
#     expected = pd.DataFrame(
#         {
#             "date": pd.date_range(start="2023-01-01 10:00", periods=5, freq="h"),
#             "value": ["3", "4", "5", "b", "c"],
#         }
#     )
#     dfutil.setindex(expected, "date")
#     pd.testing.assert_frame_equal(result, expected)





# -------------------------------
# 构造测试数据的辅助函数
# -------------------------------
def create_sample_df(n: int) -> pd.DataFrame:
    """
    构造一个包含n行数据的DataFrame，
    包含必需的字段：date, high, low, atr
    """
    dates = pd.date_range("2023-01-01", periods=n, freq="D")
    # 这里构造的high和low为线性序列，atr设为固定值1.0
    high = np.linspace(100, 200, n)
    low = np.linspace(90, 190, n)
    atr = np.full(n, 1.0)
    return pd.DataFrame({"date": dates, "high": high, "low": low, "atr": atr})


# -------------------------------
# 测试用例1：测试返回值的长度
# -------------------------------



# -------------------------------
# 测试用例2：测试Direction数组中的取值是否仅包含 0, 1, -1
# -------------------------------
# def test_get_indicators_direction_values(tmp_path):
#     df = create_sample_df(15)
#     # 这里采用较小的window方便观测
#     Direction, _, _ = get_direction(
#         tmp_path, "dummy_pair", df, timeindex="date", omega_n="atr", window=3
#     )
#     unique_values = np.unique(Direction)
#     for val in unique_values:
#         assert val in (0, 1, -1)


# # -------------------------------
# # 测试用例3：测试lowerband和upperband的计算结果
# # -------------------------------
# def test_lowerband_upperband_values(tmp_path):
#     n = 10
#     window = 4
#     df = create_sample_df(n)
#     _, lowerband, upperband = get_direction(
#         tmp_path, "dummy_pair", df, timeindex="date", omega_n="atr", window=window
#     )
#     # 根据代码逻辑，对于i从(window-1)开始，
#     # lowerband[i] 应该是取merged_df中最近window个元素的最低值，
#     # 由于dummy_merge_predict_df直接返回informative，即原始df，因此我们直接取df["low"]
#     for i in range(window - 1, n):
#         expected_lower = df["low"].iloc[i - window + 1 : i + 1].min()
#         expected_upper = df["high"].iloc[i - window + 1 : i + 1].max()
#         np.testing.assert_allclose(lowerband[i], expected_lower)
#         np.testing.assert_allclose(upperband[i], expected_upper)
#     # 对于i < window-1, 由于循环未赋值，默认值应为0
#     for i in range(window - 1):
#         assert lowerband[i] == 0
#         assert upperband[i] == 0


# # 测试数据
# @pytest.fixture
# def sample_data():
#     data = {
#         "date": pd.date_range(start="1/1/2023", periods=10),
#         "high": [100, 102, 101, 103, 104, 105, 106, 107, 108, 109],
#         "low": [98, 99, 97, 96, 95, 94, 93, 92, 91, 90],
#         "atr": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
#     }
#     return pd.DataFrame(data)


# def test_get_indicators_window_1(sample_data):

#     direction, lowerband, upperband = get_direction(
#         loadf_dir_path=Path("dummy_path"),
#         pair="dummy_pair",
#         df=sample_data,
#         timeindex="date",
#         omega_n="atr",
#         window=1,
#     )

#     expected_direction = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
#     expected_lowerband = np.array([98, 99, 97, 96, 95, 94, 93, 92, 91, 90])
#     expected_upperband = np.array([100, 102, 101, 103, 104, 105, 106, 107, 108, 109])
#     print(direction)
#     assert np.array_equal(direction, expected_direction)
#     assert np.array_equal(lowerband, expected_lowerband)
#     assert np.array_equal(upperband, expected_upperband)


# def test_get_indicators_window_3(tmp_path, sample_data):

#     direction, lowerband, upperband = get_direction(
#         loadf_dir_path=tmp_path,
#         pair="dummy_pair",
#         df=sample_data,
#         timeindex="date",
#         omega_n="atr",
#         window=3,
#     )
#     print(direction)
#     expected_direction = np.array([0, 0, 1, 1, 1, 1, 1, 1, 1, 1])
#     expected_lowerband = np.array([0, 0, 97, 96, 95, 94, 93, 92, 91, 90])
#     expected_upperband = np.array([0, 0, 102, 103, 104, 105, 106, 107, 108, 109])

#     assert np.array_equal(direction, expected_direction)
#     assert np.array_equal(lowerband, expected_lowerband)
#     assert np.array_equal(upperband, expected_upperband)


# def test_get_indicators_empty_df():
#     empty_df = pd.DataFrame(columns=["date", "high", "low", "atr"])

#     direction, lowerband, upperband = get_direction(
#         loadf_dir_path=Path("dummy_path"),
#         pair="dummy_pair",
#         df=empty_df,
#         timeindex="date",
#         omega_n="atr",
#         window=3,
#     )
#     N = len(empty_df)
#     expected = np.zeros(N)

#     assert np.array_equal(direction, expected)
#     assert np.array_equal(lowerband, expected)
#     assert np.array_equal(upperband, expected)


# # 测试数据框不足窗口大小的情况


# def test_get_indicators_insufficient_data(sample_data):
#     insufficient_df = sample_data.iloc[:2]

#     direction, lowerband, upperband = get_direction(
#         loadf_dir_path=Path("dummy_path"),
#         pair="pair",
#         df=insufficient_df,
#         timeindex="date",
#         omega_n="atr",
#         window=3,
#     )
#     N = len(insufficient_df)
#     expected = np.zeros(N)

#     assert np.array_equal(direction, expected)
#     assert np.array_equal(lowerband, expected)
#     assert np.array_equal(upperband, expected)


# @pytest.fixture
# def tmp_dirdata(tmp_path):
#     # 创建临时目录，并写入测试数据
#     df = pd.DataFrame(
#         {
#             "date": pd.date_range(start="2023-01-09", periods=4, freq="D"),
#             "high": [150, 155, 100, 165],
#             "low": [120, 125, 170, 135],
#             "close": [135, 140, 145, 150],
#             "atr": [5, 5.2, 5.1, 5],
#         }
#     )
#     pair = "EURUSD"
#     pre_time = datetime(2023, 1, 9)
#     time_frame = timedelta(days=1)
#     p = tmp_path / filename(pair, pre_time=pre_time, time_frame=time_frame)
#     df.to_csv(p, index=False)
#     return p


# def test_load_existing_file(tmp_path, tmp_dirdata):
#     path = tmp_dirdata
#     print(path)
#     pair = "EURUSD"
#     pre_time = datetime(2023, 1, 9)
#     time_frame = timedelta(days=1)
#     # 测试文件存在时，load 函数应该返回文件内容
#     df, _ = load_df(tmp_path, pair, pre_time=pre_time, time_frame=time_frame)
#     assert len(df) == 4  # 验证加载的行数是否为10


# def test_load_non_existing_file(tmp_path):
#     pair = "EURUSD"
#     pre_time = datetime(2023, 1, 1)
#     time_frame = timedelta(days=1)

#     # 测试文件不存在时，load 函数应该返回空 DataFrame
#     df, _ = load_df(tmp_path, pair, pre_time=pre_time, time_frame=time_frame)
#     assert df.empty  # 文件不存在时，返回的 DataFrame 应为空


# def test_get_indicators_no_exception(tmp_path, sample_data, tmp_dirdata):
#     pair = "EURUSD"
#     dirdata = tmp_dirdata
#     # 调用 get_indicators 并验证没有异常

#     direction, lowerband, upperband = get_direction(
#         loadf_dir_path=tmp_path,
#         pair=pair,
#         df=sample_data,
#         timeindex="date",
#         omega_n="atr",
#         window=5,
#     )
#     assert direction[9] == -1

#     assert upperband[8] == 165


# def test_get_indicators_invalid_window(tmp_path, sample_data):
#     pair = "EURUSD"
#     # 设置一个无效的窗口值（大于数据长度）
#     window = 20
#     direction, lowerband, upperband = get_direction(
#         loadf_dir_path=tmp_path,
#         pair=pair,
#         df=sample_data,
#         timeindex="date",
#         omega_n="atr",
#         window=window,
#     )

#     # 确保返回的 direction, lowerband, upperband 为空数组
#     assert np.all(direction == 0)
#     assert np.all(lowerband == 0)
#     assert np.all(upperband == 0)


# def test_get_indicators_invalid_atr_column(tmp_path):
#     # 测试没有提供正确的 ATR 列时的行为
#     pair = "EURUSD"
#     df = pd.DataFrame(
#         {
#             "date": pd.date_range(start="2023-01-01", periods=10, freq="D"),
#             "high": [150, 155, 160, 165, 170, 175, 180, 185, 190, 195],
#             "low": [120, 125, 130, 135, 140, 145, 150, 155, 160, 165],
#             "close": [135, 140, 145, 150, 155, 160, 165, 170, 175, 180],
#         }
#     )

#     # ATR 列缺失
#     with pytest.raises(KeyError):
#         get_direction(
#             loadf_dir_path=tmp_path,
#             pair=pair,
#             df=df,
#             timeindex="date",
#             omega_n="atr",  # 'atr' column is missing
#             window=5,
#         )


# def test_get_indicators_invalid_timeindex_column(tmp_path):
#     # 测试没有提供正确的 timeindex 列时的行为
#     tmp_dir = tmp_path
#     pair = "EURUSD"
#     df = pd.DataFrame(
#         {
#             "date": pd.date_range(start="2023-01-01", periods=10, freq="D"),
#             "high": [150, 155, 160, 165, 170, 175, 180, 185, 190, 195],
#             "low": [120, 125, 130, 135, 140, 145, 150, 155, 160, 165],
#             "close": [135, 140, 145, 150, 155, 160, 165, 170, 175, 180],
#             "atr": [5, 5.2, 5.1, 5, 4.8, 5.1, 5, 5.2, 5, 4.9],
#         }
#     )

#     # timeindex 列缺失
#     with pytest.raises(KeyError):
#         get_direction(
#             loadf_dir_path=tmp_dir,
#             pair=pair,
#             df=df,
#             timeindex="nonexistent_column",  # Column that does not exist
#             omega_n="atr",
#             window=5,
#         )


# def testget_direction_empty_window(tmp_path, sample_data):
#     # 测试窗口为0的情况
#     pair = "EURUSD"

#     # 设置一个窗口值为0
#     window = 0
#     direction, lowerband, upperband = get_direction(
#         loadf_dir_path=tmp_path,
#         pair=pair,
#         df=sample_data,
#         timeindex="date",
#         omega_n="atr",
#         window=window,
#     )

#     # 确保返回的 direction, lowerband, upperband 为空数组
#     assert np.all(direction == 0)
#     assert np.all(lowerband == 0)
#     assert np.all(upperband == 0)
