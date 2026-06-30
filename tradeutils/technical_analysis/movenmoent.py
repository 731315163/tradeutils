from tradeutils.type import SequenceType
import numpy as np



def williams_force(high:SequenceType, low:SequenceType, close:SequenceType):



    """
    结合威廉力度与年线滤波的自定义技术指标实现函数。

    该函数先通过长期均线进行趋势过滤，再结合威廉力度的多空力量对比结果，
    以及价格变动的相对方向，将市场状态细分为五类可量化的条件。

    参数:
        high (np.ndarray): 对应交易周期内的最高价序列，数据类型为NumPy数组
        low (np.ndarray): 对应交易周期内的最低价序列，数据类型为NumPy数组
        close (np.ndarray): 对应交易周期内的收盘价序列，数据类型为NumPy数组
        willr_period (int): 威廉指标的计算周期，默认值为14个交易日
    
         'status': 与输入数据长度对应的市场状态标记序列，各数字编码对应的状态说明为：
                0: 价格在年线下方，或无有效计算数据，不做处理
                1: 力度值为正，且收盘价较前一日上涨
                2: 力度值为正，且收盘价较前一日下跌
                3: 力度值为负，且收盘价较前一日下跌
                4: 力度值为负，且收盘价较前一日上涨

    返回:
      
            'williams_force': 威廉力度计算结果序列
           
    """
    # 数据预处理与有效性校验
    # 将输入数据统一转换为NumPy数组格式，避免因输入数据类型不兼容导致的计算异常
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)

    # 检查输入数据的长度是否一致，避免因数据长度不匹配导致的计算异常
    if not (len(high) == len(low) == len(close)):
        raise ValueError("输入的最高价、最低价、收盘价序列长度必须一致")

   
    # 3. 计算衍生的威廉力度指标，公式为：2*收盘价 - 最高价 - 最低价
    williams_force =  2*close - low- high 
    return williams_force
  
  


