from typing import Literal
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint
import statsmodels.api as sm
from mathematics.normal import zscore
def check_cointegration(x, y,trend:Literal['c','ct']='c'):
    # 检验x与y的协整关系
    test_stat, pvalue, crit_values = coint(y, x, trend=trend, autolag='aic')
    print(f"协整检验统计量: {test_stat:.4f}")
    print(f"p值: {pvalue:.4f}")
    print(f"临界值: {crit_values}")
    return (1-pvalue)
    # 结果解读
    if pvalue < 0.05 and test_stat < crit_values[1]:
        print("拒绝原假设：存在协整关系")
        return True
    else:
        print("无法拒绝原假设：无协整关系")
        return False

 


def ols(x,y):
    result_ols = (sm.OLS(y, x)).fit()
    params = result_ols.params[1]
    dif_series = zscore(y - params * x)
    return dif_series   #两个品种经过回归系数调整后z-score的值的差价
    current_mean =np.mean( dif_series)
    current_std = np.std( dif_series)
    
    # # 当最新spread超过阈值时：
    # if spread > current_mean + 2.5 * current_std and spread != 0:
    #     if self.trade_side == 0:
    #         #short y，long x
    #         self.trade_side = -1
    #     elif self.trade_side == 1:
    #         #short y，long x
    #         #short y，long x
    #         self.trade_side = -1
    #     else:
    #         pass
    # # 当最新spread低于阈值时：
    # elif spread < current_mean - 2.5 * current_std and spread != 0:
    #     if self.trade_side == 0:
    #         #short x，long y
    #         self.trade_side = 1
    #     elif self.trade_side == -1:
    #         #short x，long y
    #         #short x，long y
    #         self.trade_side = 1
    #     else:
    #         pass
    # # 平仓逻辑
    # elif spread < current_mean and spread != 0:
    #     if self.trade_side == -1:
    #         #买入y，卖出x
    #         self.trade_side = 0
    #     else:
    #         pass
    # elif spread > current_mean and spread != 0:
    #     if self.trade_side == 1:
    #         #买入x，卖出y
    #         self.trade_side = 0
    #     else:
    #         pass
    



def check_zscore_with_df(df:pd.DataFrame,x:str,windows=1):
    length = len(df)
    coint_ary= np.zeros(length)
    zscore_df  = pd.DataFrame(np.nan, index=df.index, columns=df.columns)
    for idx in range(windows,len(df)):
        x_series = df[x][idx-windows:idx]
        for y_name in df.columns:
            if x == y_name:
                continue
            y_series = df[y_name][idx-windows:idx]
            p = check_cointegration(x_series, y_series)
            zscore_df[y_name][idx] = ols(x_series,y_series)*p


if __name__ == "__main__":
   # 生成模拟数据（I(1)序列）
    np.random.seed(42)
    n = 200
    x = np.cumsum(np.random.randn(n))          # 非平稳序列
    y = 0.8 * x + np.random.randn(n)          # 与x协整
    z = np.cumsum(np.random.randn(n))          # 独立非平稳序列

    check_cointegration(x, y)
    check_cointegration(x, z)