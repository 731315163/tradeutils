from pandasutils import TimedeltaType,DatetimeType,dfutil,pathutil,TimeFrameStr,TimeFormat
from pathlib import Path
import pandas as pd


def filename(
    fn: str = "",
    pair: str = "",
    pre_time: DatetimeType | None = None,
    time_frame: TimedeltaType | str = "",
    suffix: str = ".csv",
):

    name_array = []
    if fn:
        name_array.append(fn)
    if pair:
        name_array.append(pair)
    if time_frame:
        time_frame = (
            time_frame
            if isinstance(time_frame, str)
            else str(TimeFrameStr(freq=time_frame))
        )
        name_array.append(time_frame)
    if pre_time:
        name_array.append(TimeFormat(datetime=pre_time).yymmdd())

    _filename = "_".join(name_array)
    fn = pathutil.sanitize_filename(_filename) + suffix
    return fn

def load_df(
    dir,
    pair: str,
    time_frame: TimedeltaType,
    pre_time: DatetimeType | None = None,
    part_filename: str = "",
):
    fname = filename(
        fn=part_filename, pair=pair, pre_time=pre_time, time_frame=time_frame
    )

    filepath = Path(dir) / fname
    if not filepath.exists():
        return pd.DataFrame(), filepath
    else:
        return dfutil.readpd(filepath), filepath