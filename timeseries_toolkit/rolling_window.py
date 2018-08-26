from timeseries_toolkit.base import BaseTS
import pandas as pd
from tsfresh.utilities.dataframe_functions import roll_time_series
from copy import deepcopy


class RollWin(BaseTS):

    def __init__(self, df: pd.DataFrame,
                 id_col: str,
                 target_col: str,
                 datetime_col: str,
                 dt_format: str,
                 forecast_horizon: int):
        super().__init__(df, id_col, target_col, datetime_col, dt_format,
                         forecast_horizon)

    def preprocess(self, feature_window: int,
                   aggregations: dict,
                   strict_feature_window: bool = True,
                   include_target: bool = True,
                   drop_na_target: bool = True
                   ) -> tuple:

        assert feature_window <= self.max_feature_window, \
            "Try smaller integer feature window!"
        df = self.df.copy()
        id_col = self.id_col
        datetime_col = self.datetime_col
        target_col = self.target_col
        forecast_horizon = self.forecast_horizon
        df[datetime_col] = pd.to_datetime(df[datetime_col],
                                          format=self.dt_format).dt.date
        df['timeID'] = super(RollWin, RollWin)._map_timeid(df, datetime_col)
        df['kind'] = df[id_col]

        # Process target variable outside rolling window implementation
        df_target = df[[id_col, datetime_col, target_col]]
        df_target['target_shift'] = df_target.groupby(id_col)[
            target_col].shift(
            -forecast_horizon)
        df_target = df_target.rename(columns={datetime_col: 'ref_date'})
        df_target.drop(target_col, 1, inplace=True)

        # Apply rolling and do some processing
        df_rolled = roll_time_series(df, column_id=id_col,
                                     column_sort='timeID',
                                     column_kind='kind',
                                     rolling_direction=1,
                                     max_timeshift=feature_window - 1)
        df_rolled = df_rolled.rename(
            columns={id_col: 'winID', 'kind': id_col})
        cols = list(df_rolled.columns.values)
        first_cols = [id_col, 'winID', 'timeID', datetime_col]
        remaining_cols = sorted(list(set(cols) - set(first_cols)))
        cols = first_cols + remaining_cols
        df_rolled = df_rolled[cols].sort_values(by=[id_col, 'winID',
                                                    'timeID']). \
            reset_index(drop=True)
        df_rolled['ref_date'] = df_rolled.groupby([id_col, 'winID'])[
            datetime_col].transform('last')

        df_rolled = pd.merge(df_rolled, df_target, how='left',
                             on=[id_col, 'ref_date'])
        cols = list(df_rolled.columns)
        first_cols = [id_col, 'ref_date', 'winID', datetime_col, 'timeID',
                      'target_shift']
        remaining_cols = list(set(cols) - set(first_cols))
        cols = first_cols + sorted(remaining_cols)
        df_rolled = df_rolled[cols]

        self.n_strict_rolling_win = df_rolled[
                df_rolled.groupby([id_col, 'winID'])['timeID'].transform(
                    len) == feature_window].dropna(subset=['target_shift'])['timeID'].nunique()

        if strict_feature_window:
            df_rolled = df_rolled[
                df_rolled.groupby([id_col, 'winID'])['timeID'].transform(
                    len) == feature_window]
        else:
            pass

        if drop_na_target:
            df_rolled.dropna(subset=['target_shift'], inplace=True)
        else:
            pass

        self.n_rolling_win = df_rolled['timeID'].nunique()

        # TODO: set default aggregations as mean and last

        aggregations_local = deepcopy(aggregations)

        if include_target:
            aggregations_local[target_col] = 'last'
        else:
            pass

        df_aggregated = df_rolled.groupby([id_col,
                                           'ref_date']).agg(
            aggregations_local)
        df_aggregated.reset_index(inplace=True)
        # Rename columns
        df_aggregated.columns = [i[0] + '_' + i[1] if len(i) == 2 else i for i
                                 in
                                 df_aggregated.columns]
        df_aggregated.columns = [i[:-1] if i[-1] == '_' else i for i in
                                 df_aggregated.columns]
        df_aggregated['month'] = df_aggregated['ref_date'].map(
            lambda x: x.month)
        # TODO: get dummies for all categoricals
        df_aggregated = pd.concat(
            [df_aggregated, pd.get_dummies(list(df_aggregated.id))], axis=1)
        if include_target:
            df_aggregated.rename(columns={target_col + '_last': target_col},
                                 inplace=True)
        else:
            pass

        cols = list(df_aggregated.columns)
        first_cols = [id_col, 'ref_date', 'target', 'month']
        remaining_cols = list(set(cols) - set(first_cols))
        cols = first_cols + sorted(remaining_cols)
        df_aggregated = df_aggregated[cols]

        return df_aggregated, df_rolled
