
import numpy as np
from tradeutils.type import SequenceType

def kelly_leverage(array: SequenceType,r=0.03):
    return (np.mean(array)-r) /np.var(array)
