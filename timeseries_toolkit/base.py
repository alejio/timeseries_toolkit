import pandas as pd
from timeseries_toolkit.diagnostics import omega

class BaseTS:
    """Base class for ts objects"""

    def __init__(self, df: pd.DataFrame,
                 id_col: str,
                 target_col: str,
                 datetime_col: str,
                 dt_format: str,
                 forecast_horizon: int):


        self.df = df
        self.id_col = id_col
        self.target_col = target_col
        self.datetime_col = datetime_col
        self.dt_format = dt_format
        self.forecast_horizon = forecast_horizon
        self.n_ids = self._unique_ids(df, id_col)
        self.n_timeids = self._map_timeid(df, datetime_col).nunique()
        self.max_feature_window = self.n_timeids - self.forecast_horizon

    @staticmethod
    def _unique_ids(df: pd.DataFrame, id_col: str):
        return df[id_col].nunique()

    @staticmethod
    def _map_timeid(df: pd.DataFrame, datetime_col: str):
        """
        Maps datetime to an integer time_id for internal purposes
        """
        datetime = df[datetime_col]
        dates = [str(i) for i in sorted(datetime.unique())]
        ids = list(range(1, len(dates) + 1))
        dict_map = {dates[key]: ids[key] for key in range(len(dates))}
        return datetime.map(lambda x: dict_map[str(x)])

    @property
    def forecastability(self):
        df = self.df
        id_col = self.id_col
        target_col = self.target_col
        return df.groupby(id_col)[target_col].apply(omega)

    # def preprocess(self):
    #     return self.df
    #
    # def predict(self, df_pred, model):
    #     df_pred = df_pred.preprocess()
    #     df_pred['predictions'] = model(df_pred)
    #     return df_pred