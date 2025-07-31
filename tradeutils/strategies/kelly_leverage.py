
import numpy as np
from tradeutils.type import SequenceType


def kelly_leverage(returns:SequenceType):
    # 转换为对数收益率
    log_returns = np.log(1 + np.array(returns))
    
    # 计算均值和方差
    mu = np.mean(log_returns)
    sigma_sq = np.var(log_returns)
    
    # 处理方差为零的情况
    if sigma_sq == 0:
        if mu > 0:
            return np.nan  # 如果均值为正且方差为零，理论上可以无限杠杆
        else:
            return 0.0  # 如果均值非正，不建议投资
    
    # 计算凯利杠杆
    kelly_leverage = mu / sigma_sq
    
    return kelly_leverage
