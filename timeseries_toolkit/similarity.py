import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def scale_zero_one(xvals, yinterp):
    scaler = MinMaxScaler()
    scaled = scaler.fit(list(zip(xvals, yinterp)))
    scaled_xy = scaled.transform(list(zip(xvals, yinterp)))
    return scaled_xy


def DTWDistance(s1, s2):
    DTW = {}

    for i in range(len(s1)):
        DTW[(i, -1)] = float('inf')
    for i in range(len(s2)):
        DTW[(-1, i)] = float('inf')
    DTW[(-1, -1)] = 0

    for i in range(len(s1)):
        for j in range(len(s2)):
            dist = (s1[i] - s2[j]) ** 2
            DTW[(i, j)] = dist + min(DTW[(i - 1, j)], DTW[(i, j - 1)],
                                     DTW[(i - 1, j - 1)])

    return np.sqrt(DTW[len(s1) - 1, len(s2) - 1])


num = 100
c = 0
m = 1
n = int(0.7 * num)

x1 = np.linspace(0, 26, num=num)

y_sustainedRiser = np.exp(x1 * 0.1)
y_sustainedFaller = 1 / (x1 + 1)  # +1, cannot divide by 0
y_seasonalRisers = x1 * np.sin(x1 / 2) ** 2 + 0.5 * x1
y_seasonalFallers = np.sin(x1 / 2) ** 2 / (
            x1 + 1) - x1 / 200  # +1, cannot divide by 0
y_risingStar = np.exp(x1)
y_fallenStar = [np.exp(i * 0.5) for i in x1[:n]] + [1 / i for i in x1[n:]]

bucket = {'Sustained Riser': scale_zero_one(x1, y_sustainedRiser)[:, 1],
          'Sustained Faller': scale_zero_one(x1, y_sustainedFaller)[:, 1],
          'Rising Star': scale_zero_one(x1, y_risingStar)[:, 1],
          'Fallen Star': scale_zero_one(x1, y_fallenStar)[:, 1],
          'Seasonal Riser': scale_zero_one(x1, y_seasonalRisers)[:, 1],
          'Seasonal Faller': scale_zero_one(x1, y_seasonalFallers)[:, 1]}


def get_dtw(df_raw):
    df_category = pd.DataFrame()
    for p in range(len(df_raw)):
        sublen = df_raw.iloc[p]
        x = range(len(df_raw.loc[sublen.name]))
        y = df_raw.loc[sublen.name].values

        xvals = np.linspace(min(list(x)), max(list(x)), num)
        yinterp = np.interp(xvals, x, y)
        ys_interp = scale_zero_one(xvals, yinterp)[:, 1]

        for i in bucket:
            DTW_Distance = DTWDistance(ys_interp, bucket[i])
            df_category.loc[sublen.name, i] = DTW_Distance
    return df_category