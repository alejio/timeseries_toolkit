import pandas as pd
from tsfresh.utilities.dataframe_functions import roll_time_series
import numpy as np


class RollingWindow:

    def __init__(self, df: pd.DataFrame, feature_window, forecast_horizon,
                 target_col: str='target', datetime_col: str='datetime'):
        # self.valid = False
        self.df = df
        self.target_col = target_col
        self.datetime_col = datetime_col
        self.feature_window = feature_window
        self.forecast_horizon = forecast_horizon

        # def validate():
        #     # TODO: forecast window
        #     return

    def map_timeId(datetime: pd.Series) -> pd.Series:
        """
        Maps datetime to an integer time_id
        :param datetime: pandas Series with input datetime
        :return: pandas Series integer index of increasing time
        """
        dates = [str(i) for i in sorted(datetime.unique())]
        ids = list(range(1, len(dates) + 1))
        dict_map = {dates[key]: ids[key] for key in range(len(dates))}
        return datetime.map(lambda x: dict_map[str(x)])



    def basic_prep(self) -> pd.DataFrame:
        """

        :return:
        """
        df = self.df.copy()
        datetime_col = self.datetime_col

        df[datetime_col] = pd.to_datetime(df[datetime_col],
                                        format='%d/%m/%Y').dt.date
        df['timeID'] = RollingWindow.map_timeId(df[datetime_col])
        df['kind'] = df['id']

        return df

    def get_rolled(self) -> pd.DataFrame:
        """

        :return:
        """

        feature_window = self.feature_window
        forecast_horizon = self.forecast_horizon
        target_col = self.target_col
        datetime_col = self.datetime_col


        df = RollingWindow.basic_prep(self)

        df_target = df[['id', datetime_col, target_col]]
        df_target['target_shift'] = df_target.groupby('id')[target_col].shift(
            -forecast_horizon)
        df_target = df_target.rename(columns={datetime_col: 'ref_date'})
        df_target.drop(target_col, 1, inplace=True)

        df_rolled = roll_time_series(df, column_id='id', column_sort='timeID',
                                     column_kind='kind', rolling_direction=1,
                                     max_timeshift=feature_window - 1)

        df_rolled = df_rolled.rename(columns={'id': 'winID', 'kind': 'id'})

        cols = list(df_rolled.columns.values)
        first_cols = ['id', 'winID', 'timeID', datetime_col]
        remaining_cols = sorted(list(set(cols) - set(first_cols)))
        cols = first_cols + remaining_cols

        df_rolled = df_rolled[cols].sort_values(
            by=['id', 'winID', 'timeID']).reset_index(drop=True)
        df_rolled['ref_date'] = df_rolled.groupby(['id', 'winID'])[
            datetime_col].transform('last')

        df_rolled_full = pd.merge(df_rolled, df_target, how='left',
                                  on=['id', 'ref_date'])
        df_rolled_full = df_rolled_full[
            df_rolled_full.groupby(['id', 'winID'])['timeID'].transform(
                len) == feature_window]
        df_rolled_full.dropna(subset=['target_shift'], inplace=True)
        cols = list(df_rolled_full.columns)
        first_cols = ['id', 'ref_date', 'winID', datetime_col, 'timeID',
                      'target_shift']
        remaining_cols = list(set(cols) - set(first_cols))
        cols = first_cols + sorted(remaining_cols)
        df_rolled_full = df_rolled_full[cols]

        return df_rolled_full