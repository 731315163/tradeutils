from tradeutils.technical_analysis.Indicator import hurst_exponent
import numpy as np
import talib


def test_hurst_exponent():
    # 生成测试数据
    np.random.seed(42)
    
    # 1. 随机游走序列 (理论H=0.5)
    random_walk = np.cumsum(np.random.randn(1000))
    hurst_random = hurst_exponent(random_walk)
    
    # 2. 趋势性序列 (理论H>0.5)
    trend = np.cumsum(np.random.randn(1000) + 0.1)  # 加入正向漂移
    hurst_trend = hurst_exponent(trend)
    
    # 3. 反持续性序列 (理论H<0.5)
    anti_persistent = np.zeros(1000)
    for i in range(1, 1000):
        anti_persistent[i] = -0.8 * anti_persistent[i-1] + np.random.randn()
    hurst_anti = hurst_exponent(anti_persistent)
    
    print(f"随机游走序列赫斯特指数: {hurst_random:.4f}")
    print(f"趋势性序列赫斯特指数: {hurst_trend:.4f}")
    print(f"反持续性序列赫斯特指数: {hurst_anti:.4f}")



def test_tanh():
    a = 1000
    b=10
    c= 1
    d= 0.001
    tanh= talib.TANH( np.array([1000,10 ,1,d,np.pi]))
    print(f"tanh(1000) = {tanh[0]}")
    print(f"tanh(10) = {tanh[1]}")
    print(f"tanh(1) = {tanh[2]}")
    print(f"tanh(0.001) = {tanh[3]}")
    print(f"tanh(pi) = {tanh[4]}")
    assert True