import math
import sys
from copy import deepcopy
from typing import Any, cast, overload

import numpy as np
import pandas as pd

# 现在你可以导入共享文件夹中的模块了
from pandasutils import SequenceType


class SparseArray:
    def __init__(self, arr_range: range, init_value):
        """
        初始化自定义数组
        :param initial_data: 初始数据（可迭代对象，如列表、元组等）
        """
        self.default_value = init_value
        # self.range = arr_range
        # self.array = None
        self.sparse_array: dict[int, Any] = {}
        self._iter_init_index = 0
        self.max_index = -1
        self._min_index = sys.maxsize

    @property
    def min_index(self):
        return self._min_index
    @min_index.setter
    def min_index(self,value:int):
        if  self._min_index > value:
            self._min_index = value
        
            
    # def count(self):
    #     notnan = np.count_nonzero(~np.isnan(self.array)) if self.array else 0
    #     return notnan + len(self.sparse_array)

    def __len__(self) -> int:
        return len(self.sparse_array)

    def __iter__(self):
        
        index = cast(int,self.min_index)
        self._iter_init_index = index
        return self  # 返回自身作为迭代器

    def __next__(self):
        if self._iter_init_index > self.max_index:
            raise StopIteration
        value = self[self._iter_init_index]
        self._iter_init_index += 1
        return value

    def _slice_indicator(self, index: slice, length):
        start, stop, step = index.start, index.stop, index.step
        if step is None:
            step = 1
        if start is None or start < 0:
            start = 0
        if stop is None:
            stop = start + length
        if stop < 0:
            stop = length + stop + 1
        return range(start, stop, step)

    @overload
    def __getitem__(self, index: int) -> Any: ...
    @overload
    def __getitem__(self, index: range) -> list: ...
    @overload
    def __getitem__(self, index: slice) -> list: ...
    def __getitem__(self, index):
        """
        支持通过索引访问元素
        """
        if isinstance(index, slice):
            return [self[i] for i in self._slice_indicator(index, self.max_index + 1)]
        elif isinstance(index, range):
            return [self[i] for i in index]
        else:
            return self.sparse_array.get(index, self.default_value)

    @overload
    def __setitem__(self, index: int, value): ...
    @overload
    def __setitem__(self, index: range, value): ...
    @overload
    def __setitem__(self, index: slice, value): ...
    def __setitem__(self, index, value):
        """
        支持通过索引修改元素
        """

        if isinstance(index, int):
            index = int(index)
            self.min_index = index
            if index > self.max_index:
                self.max_index = index
                
            elif index == self.max_index:
                for i in range(self.max_index, -1, -1):
                    if pd.isna(self[i]) or self[i] == self.default_value:
                        self.max_index -= 1
                    else:
                        break
            self.sparse_array[index] = value
        elif isinstance(index, range):
            for idx in index:
                self[idx] = value[idx]
        elif isinstance(index, slice):
            self[self._slice_indicator(index=index, length=len(value))] = value
        else:
            raise TypeError("index must be int or range or slice")


class VolumeDistribution(SparseArray):

    def __init__(self, arr_range: range, init_value):
        super().__init__(arr_range, init_value)
        self.std_idx = -1
        self.std_price = 0

    def get_floor_idx(self, value):

        if self.std_idx == -1:
            return -1
        else:
            return cast(int, math.floor(value / self.std_price * self.std_idx))

    def get_floor_price(self, idx: int):
        if self.std_idx == -1:
            return np.nan
        else:
            return idx / self.std_idx * self.std_price

    def set_anchor(self, idx: int, *args):

        _v = sum(args) / len(args) if args else 0
        self.std_idx, self.std_price = idx, _v

    def _for_volume(self, idx: int, vol: SequenceType, *args):
        dism = len(args)
        for price in args:
            i = self.get_floor_idx(price[idx])
            yield i, (vol[idx] / dism)

    def add_volume(self, idx: int, vol: SequenceType, *args):

        for i, v in self._for_volume(idx, vol, *args):
            if pd.isna(self[i]):
                self[i] = v
            else:
                self[i] += v

    def sub_volume(self, idx: int, vol: SequenceType, *args):
        for i, v in self._for_volume(idx, vol, *args):
            if pd.isna(self[i]):
                self[i] = -v
            else:
                self[i] -= v

    def analyze_volumeprofile(self, PV_percent: tuple[float, float]):
        r = analyze_volumeprofile(self, PV_percent)

        if r:

            low = self.get_floor_price(r.start)
            high = self.get_floor_price(r.stop)
            return high, low
        else:
            return np.nan, np.nan


# todo
def analyze_volumeprofile(vd: VolumeDistribution, PV_percent: tuple[float, float]):
    price_percent, v_percent = PV_percent
    if price_percent < 0 or price_percent > 1 or v_percent < 0 or v_percent > 1:
        raise ValueError("p,v must be 0<=p,v<=1")
    price_sum=len(vd)
    volume_sum = sum(vd)
    # for _volume in vd:
    #     if not pd.isna(_volume) and not math.isclose(_volume,0.0):
    #         volume_sum += _volume
    price_num = math.floor(price_sum * price_percent)
    start = price_num + vd.min_index
    max_r = range(vd.min_index, start)
    cur_volume = np.sum(vd[max_r])
    max_volume = cur_volume
    for i in range(start, vd.max_index +1):
        cur_volume += vd[i]
        cur_volume -= vd[i - price_num]
        if cur_volume > max_volume:
            max_r = range(i - price_num + 1, i + 1)
            max_volume = cur_volume

    volume_num = volume_sum * v_percent
    if max_volume > volume_num:
        return max_r


def get_volumedistributions(
    high: SequenceType, low: SequenceType, vol: SequenceType, timeperiod: int
):
    if high is None or low is None or vol is None:
        raise ValueError("high,low,vol must be not empty")
    length = len(high)
    if length == 0 or length != len(low) or length != len(vol):
        raise ValueError("length is must equl and big than 0")

    volume_distributions = np.empty(shape=length, dtype=VolumeDistribution)

    start = timeperiod - 1
    max_limit_price = 0
    for i in range(start, length):

        if high[i] > max_limit_price:
            r = range(max(0, i - start), i + 1)
            max_limit_price = np.max(high[r.start : r.stop])
            _volumedistribution = VolumeDistribution(
                arr_range=range(50, 150), init_value=0.0
            )
            _volumedistribution.set_anchor(
                100, _volumedistribution.std_price, max_limit_price
            )
            for k in r:
                _volumedistribution.add_volume(k, vol, high, low)
        else:
            _volumedistribution = deepcopy(volume_distributions[i - 1])
            _volumedistribution.add_volume(i, vol, high, low)
            _volumedistribution.sub_volume(i - timeperiod, vol, high, low)

        volume_distributions[i] = _volumedistribution
    return volume_distributions


def get_analyze_volumeprofiles(
    high: SequenceType,
    low: SequenceType,
    vol: SequenceType,
    timeperiod: int,
    PV_percent: tuple[float, float],
):
    volumes = get_volumedistributions(high, low, vol, timeperiod)

    high = np.full(len(volumes), np.nan, dtype=float)
    low = np.full(len(volumes), np.nan, dtype=float)
    for i in range(timeperiod, len(volumes)):
        v = cast(VolumeDistribution, volumes[i])
        high[i], low[i] = v.analyze_volumeprofile(PV_percent)
    return high, low
