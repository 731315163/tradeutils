from dataclasses import dataclass
import numpy as np
import bisect

@dataclass
class Move_Stoploss():

    def __init__(self,max_profit:float=0.0,ratio:float=1):
        self.max_profit= max_profit
        self.ratio =ratio
        self.stoploss_sequece=self.set_stoploss_lever(ratio)
    def set_stoploss_lever(self,ratio:float =1):
        ratio = ratio/100
        return [ratio*n for n in [1,1,2,3,5,8,13,21,34,55]]

    def stoploss(self,profit:float=0):
        """
        更新止损价
        :param profit: 当前利润
        :return: None
        """
        index = max(1, bisect.bisect_left(self.stoploss_sequece, self.max_profit))
        if profit > self.max_profit:
            self.max_profit = profit
        _stoploss = self.max_profit - self.stoploss_sequece[index]/index
        return _stoploss   