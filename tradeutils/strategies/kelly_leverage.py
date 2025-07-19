
import numpy as np



from collections.abc import Sequence,MutableSequence

def kelly_leverage(array: Sequence|MutableSequence|np.ndarray,r=0.03):
    return (np.mean(array)-r) /np.var(array)
