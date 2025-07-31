import numpy as np
import pytest
from tradeutils.strategies.kelly_leverage import kelly_leverage


def test_kelly_leverage_positive_negative_profits():
    profits = np.array([0.1, -0.05, 0.2, -0.1, 0.15])
    result = kelly_leverage(profits)
    expected = (np.mean(profits) - 0.03) / np.var(profits)
    assert np.isclose(result, expected)


def test_kelly_leverage_all_negative_profits():
    profits = np.array([-0.1, -0.05, -0.2, -0.15])
    result = kelly_leverage(profits)
    expected = -33.04681295221511
    assert np.isclose(result, expected)


def test_kelly_leverage_mean_equals_r():
    profits = np.array([0.03, 0.03, 0.03])  # mean = r = 0.03
    result = kelly_leverage(profits)
    assert np.isnan(result)


def test_kelly_leverage_mixed_input_types():
    # Test with list of floats (positive and negative)
    list_profits = [-0.1, 0.2, 0.05]
    result1 = kelly_leverage(list_profits)
    expected1 = 3.0337061568196226
    assert np.isclose(result1, expected1)

def test_kelly_leverage_mixe_():
   
    # Test with numpy array (positive and negative)
    np_profits = np.array([0.1, -0.05, 0.1,0.2,0.3])
    result2 = kelly_leverage(np_profits)
    expected2 = 10.684268153246293
    assert np.isclose(result2, expected2)


def test_kelly_leverage_invalid_inputs():
    # Test with None
    with pytest.raises(TypeError):
        kelly_leverage(None)
    
    # Test with non-numeric string
    with pytest.raises(TypeError):
        kelly_leverage("invalid")
    
    # Test with single number
    with pytest.raises(TypeError):
        kelly_leverage(0.1)