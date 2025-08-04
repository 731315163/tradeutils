
class IndicatorName:
    high = "high"
    low = "low"
    open = "open"
    close = "close"
    volume = "volume"
    pair = "pair"
    side="side"
    enter_long = "enter_long"
    enter_short = "enter_short"
    exit_long = "exit_long"
    exit_short = "exit_short"
    stake_amount = "stake_amount"
    direction = "direction"
    profit = "profit"
    #indicator
    vwap = "vwap"
    std = "std"

    def __init__(self, timeframe: str = ""):
        #base
        self.high = "high"
        self.low = "low"
        self.open = "open"
        self.close = "close"
        self.volume = "volume"
        self.pair = "pair"
        self.enter_long = "enter_long"
        self.enter_short = "enter_short"
        self.exit_long = "exit_long"
        self.exit_short = "exit_short"
        #indicator
        self.avgprice = "avgprice"
        self.lsma = "lsma"
        self.atr = "atr"
        self.obv= "obv"
        self.tr = "tr"
        
        
        # if timeframe:
        #     self.atr_longtime = f"{self.atr}_{timeframe}"
        #     self.direction_longtime = f"{self.direction}_{timeframe}"

        #     self.upperband_longtime = f"{self.upperband}_{timeframe}"
        #     self.middleband_longtime = f"{self.middleband}_{timeframe}"
        #     self.lowerband_longtime = f"{self.lowerband}_{timeframe}"
        # else:
        #     self.atr_longtime = self.atr
        #     self.direction_longtime = self.direction
        #     self.lowerband_longtime = self.lowerband
        #     self.upperband_longtime = self.upperband
        #     self.middleband_longtime = self.middleband