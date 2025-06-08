
import numpy as np



from collections.abc import Sequence

def Kelly_Leverage(array: Sequence,r=0.03):
    return (np.mean(array)-r) /np.var(array)
