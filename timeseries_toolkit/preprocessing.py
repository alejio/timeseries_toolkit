import pandas as pd
from tsfresh.utilities.dataframe_functions import roll_time_series
from copy import deepcopy







def get_aggregated_df(df_raw: pd.DataFrame,
                      feature_window: int,
                      forecast_horizon: int,
                      aggregations: dict,
                      strict_feature_window: int or None,
                      id_col: str='id',
                      target_col: str='target_shift',
                      datetime_col: str='ref_date',
                      dt_format: str='%d/%m/%Y',
                      include_target: bool=True,
                      drop_na_target: bool=True
                      ) -> pd.DataFrame:

    def get_rollwin_df() -> pd.DataFrame:
        """
        Implements tsfresh utilites to preprocess data for ML
        :param df_raw:
        :param feature_window:
        :param forecast_horizon:
        :param id_col:
        :param target_col:
        :param datetime_col:
        :param dtformat:
        :return:
        """

        def map_timeid(datetime: pd.Series) -> pd.Series:
            """
            Maps datetime to an integer time_id for internal purposes
            :param datetime: pandas Series with input datetime
            :return: pandas Series integer index of increasing time
            """
            dates = [str(i) for i in sorted(datetime.unique())]
            ids = list(range(1, len(dates) + 1))
            dict_map = {dates[key]: ids[key] for key in range(len(dates))}
            return datetime.map(lambda x: dict_map[str(x)])

        df = df_raw.copy()
        df[datetime_col] = pd.to_datetime(df[datetime_col],
                                          format=dtformat).dt.date
        df['timeID'] = map_timeid(df[datetime_col])
        df['kind'] = df[id_col]

        # Process target variable outside rolling window implementation
        df_target = df[[id_col, datetime_col, target_col]]
        df_target['target_shift'] = df_target.groupby(id_col)[target_col].shift(
            -forecast_horizon)
        df_target = df_target.rename(columns={datetime_col: 'ref_date'})
        df_target.drop(target_col, 1, inplace=True)

        # Apply rolling and do some processing
        df_rolled = roll_time_series(df, column_id=id_col, column_sort='timeID',
                                     column_kind='kind', rolling_direction=1,
                                     max_timeshift=feature_window - 1)
        df_rolled = df_rolled.rename(columns={id_col: 'winID', 'kind': id_col})
        cols = list(df_rolled.columns.values)
        first_cols = [id_col, 'winID', 'timeID', datetime_col]
        remaining_cols = sorted(list(set(cols) - set(first_cols)))
        cols = first_cols + remaining_cols
        df_rolled = df_rolled[cols].sort_values(by=[id_col, 'winID',
                                                    'timeID']). \
            reset_index(drop=True)
        df_rolled['ref_date'] = df_rolled.groupby([id_col, 'winID'])[
            datetime_col].transform('last')

        df_rolled_full = pd.merge(df_rolled, df_target, how='left',
                                  on=[id_col, 'ref_date'])
        cols = list(df_rolled_full.columns)
        first_cols = [id_col, 'ref_date', 'winID', datetime_col, 'timeID',
                      'target_shift']
        remaining_cols = list(set(cols) - set(first_cols))
        cols = first_cols + sorted(remaining_cols)
        df_rolled_full = df_rolled_full[cols]

        return df_rolled_full



    if strict_feature_window:
        df_rolled = df_rolled[df_rolled.groupby([id_col, 'winID'])['timeID'].transform(len) == strict_feature_window]
    else:
        pass

    if drop_na_target:
        df_rolled.dropna(subset=['target_shift'], inplace=True)
    else:
        pass

    aggregations_local = deepcopy(aggregations)

    if include_target:
        aggregations_local[target_col] = 'last'
    else:
        pass

    df_aggregated = df_rolled.groupby([id_col,
                                       datetime_col]).agg(aggregations_local)
    df_aggregated.reset_index(inplace=True)
    # Rename columns
    df_aggregated.columns = [i[0] + '_' + i[1] if len(i) == 2 else i for i in
                             df_aggregated.columns]
    df_aggregated.columns = [i[:-1] if i[-1] == '_' else i for i in
                             df_aggregated.columns]
    df_aggregated['month'] = df_aggregated['ref_date'].map(lambda x: x.month)
    df_aggregated = pd.concat(
        [df_aggregated, pd.get_dummies(list(df_aggregated.id))], axis=1)
    if include_target:
        df_aggregated.rename(columns={target_col + '_last': target_col},
                             inplace=True)
    else:
        pass

    return df_aggregated
