from timeseries_toolkit.base import BaseTS
import pandas as pd
df = pd.read_csv('~/Desktop/personal/repos/timeseries_toolkit/data/ts_data.csv')

tsobj = BaseTS(df, id_col='id', target_col = 'target',
                            datetime_col = 'datetime',
                            dt_format='%Y/%m/%d', forecast_horizon=5)

tsobj