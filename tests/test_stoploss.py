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