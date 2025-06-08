from attr import dataclass
from pandas import DataFrame
import numpy as np

class MartingaleRecoder:
    """
    Martingale recoder。
    """
    max_entry_times:int=2
    current_times:int = 0
    stake_amount:float = 0
    scaling_factor:list[float] =[] 

    def __init__(self,scaling_factor:list[float]):
        self.scaling_factor = scaling_factor
    

