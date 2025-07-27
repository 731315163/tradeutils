import numpy as np



class Move_Stoploss():


    def __init__(self):
        self.max_profit= 0
        self._stoploss = 0
        self.stoploss_sequece=np.array( [1,1,2,3,5,8,13,21,34,55])/100
        self.current_index = 1
        self.step = 0


    def stoploss(self,profit:float=0):
        """
        更新止损价
        :param profit: 当前利润
        :return: None
        """
        if profit > self.max_profit:
            self.max_profit = profit
        if profit > self.stoploss_sequece[self.current_index] :
            self.current_index = min(self.current_index+1, len(self.stoploss_sequece) -1)
            self.step = self.stoploss_sequece[self.current_index]/self.current_index
        self._stoploss = self.max_profit - self.step
        return self._stoploss         