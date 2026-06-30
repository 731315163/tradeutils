import pytest
from tradeutils.strategies.stoploss import Move_Stoploss

@pytest.fixture
def move_stoploss():
    """创建一个Move_Stoploss实例用于测试"""
    return Move_Stoploss(max_profit=0.0,ratio=1)

def test_initial_state(move_stoploss):
    """测试初始状态"""
    assert move_stoploss.max_profit == 0
    assert move_stoploss._stoploss == 0
    assert move_stoploss.current_index == 1
    assert move_stoploss.step == 0
    assert len(move_stoploss.stoploss_sequece) == 10

def test_stoploss_sequence(move_stoploss):
    """测试止损序列的递增逻辑"""
    # 初始序列 [0.01, 0.01, 0.02, 0.03, 0.05, 0.08, 0.13, 0.21, 0.34, 0.55]
    
    # 第一次触发阈值
    move_stoploss.stoploss(0.02)
    assert move_stoploss.current_index == 2
    assert move_stoploss.step == 0.02/2
    
    # 第二次触发阈值
    move_stoploss.stoploss(0.03)
    assert move_stoploss.current_index == 3
    assert move_stoploss.step == 0.03/3
    
    # 测试最大索引限制
    for i in range(10):
        move_stoploss.stoploss(1.0)
    
    assert move_stoploss.current_index == len(move_stoploss.stoploss_sequece) - 1
    assert move_stoploss.step == pytest.approx(0.55/9)

def test_max_profit_update(move_stoploss):
    """测试最大利润更新逻辑"""
    # 测试利润增长
    profits = [0.01, 0.02, 0.015, 0.03]
    expected_max = [0.01, 0.02, 0.02, 0.03]
    
    for profit, expected in zip(profits, expected_max):
        move_stoploss.stoploss(profit)
        assert move_stoploss.max_profit == expected

def test_stoploss_value(move_stoploss):
    """测试止损值计算"""
    # 初始状态
    assert move_stoploss.stoploss() == 0
    
    # 设置初始利润
    move_stoploss.stoploss(0.05)
    assert move_stoploss._stoploss == pytest.approx(0.05 - (0.02/2))
    
    # 触发多次升级
    for profit in [0.08, 0.13, 0.21]:
        move_stoploss.stoploss(profit)
    
    assert move_stoploss._stoploss == pytest.approx(0.21 - (0.03/3))

def test_no_threshold_crossing(move_stoploss):
    """测试未触发阈值时状态不变"""
    initial_index = move_stoploss.current_index
    initial_step = move_stoploss.step
    
    # 输入低于当前阈值的利润
    move_stoploss.stoploss(0.005)
    assert move_stoploss.current_index == initial_index
    assert move_stoploss.step == initial_step



import pytest
from tradeutils.strategies.position import get_position_by_stoploss


class TestGetPositionByStoploss:
    """测试 get_position_by_stoploss 函数"""

    def test_basic_calculation(self):
        """测试基本计算逻辑"""
        # position = total_balance * max_stoploss / cur_stoploss
        result = get_position_by_stoploss(
            total_balance=10000.0,
            max_stoploss=0.02,
            cur_stoploss=0.01
        )
        expected = 10000.0 * 0.02 / 0.01
        assert result == pytest.approx(expected)

    def test_equal_stoploss(self):
        """测试当前止损等于最大止损的情况"""
        result = get_position_by_stoploss(
            total_balance=10000.0,
            max_stoploss=0.02,
            cur_stoploss=0.02
        )
        expected = 10000.0 * 0.02 / 0.02
        assert result == pytest.approx(expected)
        assert result == pytest.approx(10000.0)

    def test_small_cur_stoploss(self):
        """测试当前止损很小的情况（仓位会变大）"""
        result = get_position_by_stoploss(
            total_balance=10000.0,
            max_stoploss=0.02,
            cur_stoploss=0.001
        )
        expected = 10000.0 * 0.02 / 0.001
        assert result == pytest.approx(expected)
        assert result == pytest.approx(200000.0)

    def test_large_cur_stoploss(self):
        """测试当前止损很大的情况（仓位会变小）"""
        result = get_position_by_stoploss(
            total_balance=10000.0,
            max_stoploss=0.02,
            cur_stoploss=0.1
        )
        expected = 10000.0 * 0.02 / 0.1
        assert result == pytest.approx(expected)
        assert result == pytest.approx(2000.0)

    def test_zero_max_stoploss(self):
        """测试最大止损为0的情况"""
        result = get_position_by_stoploss(
            total_balance=10000.0,
            max_stoploss=0.0,
            cur_stoploss=0.01
        )
        assert result == pytest.approx(0.0)

    def test_zero_total_balance(self):
        """测试总余额为0的情况"""
        result = get_position_by_stoploss(
            total_balance=0.0,
            max_stoploss=0.02,
            cur_stoploss=0.01
        )
        assert result == pytest.approx(0.0)

    def test_division_by_zero(self):
        """测试当前止损为0的情况（会导致除零错误）"""
        with pytest.raises(ZeroDivisionError):
            get_position_by_stoploss(
                total_balance=10000.0,
                max_stoploss=0.02,
                cur_stoploss=0.0
            )

    def test_negative_values(self):
        """测试负值输入"""
        result = get_position_by_stoploss(
            total_balance=-10000.0,
            max_stoploss=0.02,
            cur_stoploss=0.01
        )
        expected = -10000.0 * 0.02 / 0.01
        assert result == pytest.approx(expected)

    def test_float_precision(self):
        """测试浮点数精度"""
        result = get_position_by_stoploss(
            total_balance=10000.123,
            max_stoploss=0.02345,
            cur_stoploss=0.01234
        )
        expected = 10000.123 * 0.02345 / 0.01234
        assert result == pytest.approx(expected)

    def test_various_balances(self):
        """测试不同的余额值"""
        test_cases = [
            (1000.0, 0.01, 0.005, 2000.0),
            (5000.0, 0.03, 0.01, 15000.0),
             (5000.0, 0.03, 0.03, 5000.0),
             (6000.0, 0.03, 0.06, 3000.0),
            (100000.0, 0.05, 0.02, 250000.0),
        ]
        
        for total_balance, max_stoploss, cur_stoploss, expected in test_cases:
            result = get_position_by_stoploss(
                total_balance=total_balance,
                max_stoploss=max_stoploss,
                cur_stoploss=cur_stoploss
            )
            assert result == pytest.approx(expected)

    def test_return_type(self):
        """测试返回值类型"""
        result = get_position_by_stoploss(
            total_balance=10000.0,
            max_stoploss=0.02,
            cur_stoploss=0.01
        )
        assert isinstance(result, float)