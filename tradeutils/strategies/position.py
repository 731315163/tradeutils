


from collections.abc import Sequence,MutableSequence
import numpy as np








class Position:
    def __init__(self,  index: int, position_sizing: float,positions: Sequence | MutableSequence | np.ndarray):
        """
        side: long or short
        position_sizing: 仓位大小
        positions: 仓位列表
        """
        self.position_sizing = position_sizing
        self.index = index
        self.positions = positions
     
        
        
    def adjust_position_size(self,index:int,position_sizing=None):
        """
        + 加仓，- 减仓
        在positions中
        例如 [100,90,80,60]
        [100,90,80,_]为买入仓位
        [_,90,80,60]为卖出仓位
        当前持有仓位大于大的卖出 ,小于小的买入
        """
        positions = self.positions
        max_length = len(positions) - 1
        if max_length < 1:
            raise ValueError("positions must have at least 2 elements")
        # 向下舍入，
        index = np.clip(index, 0, max_length)
        # bigstake_idx = max(index, 0)
        # smallstake_idx = min(bigstake_idx + 1, max_length)
        # bigstake_idx = min(bigstake_idx, max_length)

        # # buy amount > sell amount
        # bigamount = positions[bigstake_idx]
        # smamount = positions[smallstake_idx]
        # [ 100 ,        90 ,         80 ,            0]
        # [      cur > sellindex, buyindex  > cur,  allsell]
        # 大于大的 ，小于小的,100不会被买入，0不会被卖出
        cur_amount = position_sizing if position_sizing is not None else self.position_sizing
        amount = positions[index] - cur_amount
        # if cur_amount > bigamount:
        #     amount = bigamount - cur_amount  # (-)
        # elif cur_amount < smamount:
        #     amount= smamount - cur_amount  # (+)
        
        if position_sizing is None:
            self.position_sizing =positions[index]
        self.index = index
        #调正仓位，+ 买入，- 卖出
        return amount


 