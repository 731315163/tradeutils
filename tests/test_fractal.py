import math
import numpy as np
import pytest

# 假设被测函数位于 fractal 模块中，请根据实际项目调整导入路径
from tradeutils.technical_analysis.fractal import OverlapFractal  # 替换 your_module 为实际模块名


# ------------------- 测试辅助函数 -------------------
def assert_fractal_result(result, expected):
    """断言两个数组相等，正确处理 NaN 和 inf 值"""
    # 将两者转为 float64 数组
    result = np.asarray(result, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    nan_mask = np.isnan(expected)
    # 期望为 NaN 的位置结果也应为 NaN
    assert np.all(np.isnan(result[nan_mask])), f"Expected NaN at {np.where(nan_mask)}"
    # 其余位置使用 isclose 比较，equal_nan=False 因为已处理 NaN
    valid_mask = ~nan_mask
    assert np.allclose(result[valid_mask], expected[valid_mask], equal_nan=False), \
        f"Values differ at valid indices: {np.where(valid_mask)}"


# ------------------- 多周期多数据长度参数化测试 -------------------
@pytest.mark.parametrize("period, low, high, expected", [
    # ---- 用例1: period=3, 基本数据，长度9 ----
    (
        3,
        [1, 2, 3, 4, 5, 6, 7, 8, 9],
        [2, 3, 4, 5, 6, 7, 8, 9, 10],
        [np.nan, np.nan, np.nan, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    ),
    # ---- 用例2: period=3, 出现 +inf ----
    (
        3,
        [3, 3, 3, 2],   # prev_low=3, low[3]=2 -> l=-1
        [5, 5, 5, 6],   # prev_high=5, high[3]=6 -> h=1
        [np.nan, np.nan, np.nan, math.inf]
    ),
    # ---- 用例3: period=3, 出现 -inf ----
    (
        3,
        [3, 3, 3, 4],   # prev_low=3, low[3]=4 -> l=1
        [5, 5, 5, 4],   # prev_high=5, high[3]=4 -> h=-1
        [np.nan, np.nan, np.nan, -math.inf]
    ),
    # ---- 用例4: period=3, 向下突破，h<=0且l<0 ----
    (
        3,
        [3, 3, 3, 2],   # prev_low=3, low[3]=2 -> l=-1
        [5, 5, 5, 4],   # prev_high=5, high[3]=4 -> h=-1
        # distance = (5-3 + 4-2)/2 = (2+2)/2=2, result = l/distance = -0.5
        [np.nan, np.nan, np.nan, -0.5]
    ),
    # ---- 用例5: period=3, 向上突破，h>0且l>=0 (此处l=0) ----
    (
        3,
        [3, 3, 3, 3],   # prev_low=3, low[3]=3 -> l=0
        [5, 5, 5, 6],   # prev_high=5, high[3]=6 -> h=1
        # distance = (5-3 + 6-3)/2 = (2+3)/2=2.5, result = h/distance = 0.4
        [np.nan, np.nan, np.nan, 0.4]
    ),
    # ---- 用例6: period=3, h=0, l=0 (完全等于前高前低) ----
    (
        3,
        [3, 3, 3, 3],
        [5, 5, 5, 5],
        # h=0, l=0, 进入else, sign(h)>0? No => result = l/distance = 0
        [np.nan, np.nan, np.nan, 0.0]
    ),
    # ---- 用例7: period=1, 长度为4 ----
    (
        1,
        [2, 3, 1, 4],
        [5, 4, 6, 5],
        [
            np.nan,     # i=0 前1根无
            # i=1: prev_high=5, prev_low=2; high=4, low=3 -> h=-1, l=1 => -inf
            -math.inf,
            # i=2: prev_high=4, prev_low=3; high=6, low=1 -> h=2, l=-2 => inf
            math.inf,
            # i=3: prev_high=6, prev_low=1; high=5, low=4 -> h=-1, l=3 => -inf
            -math.inf,
        ]
    ),
    # ---- 用例8: period=2, 长度为5 (混合值) ----
    (
        2,
        [2, 2, 3, 1, 2],
        [6, 5, 7, 4, 6],
        [
            np.nan, np.nan,
            # i=2: prev[0:2] low=2 high=6; curr low=3,high=7 -> h=1,l=1 => h>0,l>0
            # distance = (6-2 + 7-3)/2 = (4+4)/2=4, result=1/4=0.25
            0.25,
            # i=3: prev[1:3] low=2 high=7; curr low=1,high=4 -> h=-3,l=-1 => h<0,l<0
            # distance = (7-2 + 4-1)/2 = (5+3)/2=4, result = l/distance = -1/4 = -0.25
            -0.25,
            # i=4: prev[2:4] low=1 high=7; curr low=2,high=6 -> h=-1,l=1 => -inf
            -math.inf,
        ]
    ),
    # ---- 用例9: period=5, 长度为6（仅一个有效值） ----
    (
        5,
        [1, 2, 2, 3, 2, 1],
        [5, 6, 5, 7, 6, 8],
        [np.nan]*5 + [
            # i=5: prev_low=1, prev_high=7; high=8, low=1 -> h=1, l=0 => h>0, l=0
            # distance = (7-1 + 8-1)/2 = (6+7)/2=6.5, result = h/distance = 1/6.5 ≈ 0.15384615
            1/6.5
        ]
    ),
    # ---- 用例10: period 大于数据长度，全部 NaN ----
    (
        10,
        [1, 2, 3],
        [4, 5, 6],
        [np.nan, np.nan, np.nan]
    ),
])
def test_overlap_fractal_various(period, low, high, expected):
    """多周期多数据长度测试"""
    low_arr = np.array(low, dtype=float)
    high_arr = np.array(high, dtype=float)
    close_arr = np.zeros_like(low_arr)  # close 未被使用
    result = OverlapFractal(low_arr, high_arr, close_arr, period)
    assert len(result) == len(expected)
    assert_fractal_result(result, expected)


# ------------------- 单独测试输入长度不一致 -------------------
def test_length_mismatch_raises():
    low = np.array([1.0, 2.0, 3.0])
    high = np.array([4.0, 5.0])  # 长度不同
    close = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="low and high must be the same length"):
        OverlapFractal(low, high, close, period=3)


# ------------------- 测试 close 长度不一致是否应报错（当前实现未检查，此处可扩展） -------------------
def test_close_length_not_checked():
    """当前实现未检查close长度，仅验证不会崩溃且结果与预期一致"""
    low = np.array([1, 2, 3, 4])
    high = np.array([5, 6, 7, 8])
    close = np.array([0])  # 比 low 短
    # 不应抛出异常（因为未检查 close 长度）
    result = OverlapFractal(low, high, close, period=2)
    # 预期: 前2个 NaN, 后面根据数据计算
    expected = [np.nan, np.nan, 
                # i=2: prev_low=1, prev_high=6; curr low=3,high=7 -> h=1,l=2 => 0.2857...
                1 / ((6-1+7-3)/2),  
                # i=3: prev_low=2, prev_high=7; curr low=4,high=8 -> h=1,l=2 => 1/ ((5+4)/2) = 1/4.5 ≈0.2222
                1 / ((7-2+8-4)/2)]
    assert_fractal_result(result, expected)


# ------------------- 测试 period=0 的情况（边界） -------------------
def test_period_zero():
    """period=0 时 prev 区间为空？按代码逻辑 range(0, n) 从0开始，prev 切片为 high[0:0] 空数组，np.max 会报错。
       本测试仅记录预期行为：应抛出异常或返回 NaN。"""
    low = np.array([1.0, 2.0])
    high = np.array([3.0, 4.0])
    close = np.array([0.0, 0.0])
    with pytest.raises(ValueError):  # np.max 对空数组会抛出 ValueError
        OverlapFractal(low, high, close, period=0)