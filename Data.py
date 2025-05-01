import numpy as np
import pandas as pd

def get_close_price_np(symbol):
    data = pd.read_csv("./data/stocks/{0}.csv".format(symbol))
    data = pd.DataFrame(data)
    close_price = data["Close"]
    return close_price.to_numpy()

def get_view(data, num_days=5):
    shape_5days = (len(data) - num_days, num_days)
    strides_5days = (data.strides[0], data.strides[0])
    view_5days = np.lib.stride_tricks.as_strided(data, shape=shape_5days, strides=strides_5days)
    return view_5days
