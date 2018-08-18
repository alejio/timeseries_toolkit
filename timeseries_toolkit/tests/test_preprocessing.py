from timeseries_toolkit import preprocessing
import pandas as pd


def test_map_timeID():
    datetimes = pd.Series(pd.date_range(start='1/1/2018', end='1/08/2018'))
    datetimes = datetimes.apply(lambda x: x.strftime('%Y-%m-%d'))
    datetimes = datetimes.sample(frac=1, random_state=42).reset_index(drop=True)
    ids = preprocessing.map_timeid(datetimes)
    assert ids[0] == datetimes.rank()[0], 'Something went wrong'


def test_get_rolled_df():
    df = pd.read_csv('timeseries_toolkit/tests/fixtures/test_ts_data.csv')
    feature_window = 5
    forecast_horizon = 3
    df_rolled = preprocessing.get_rollwin_df(df, feature_window, forecast_horizon)
    assert len(df_rolled) == df['id'].nunique() * feature_window * (len(df[df.id =='AXP']) - feature_window - forecast_horizon + 1)
    # TODO: add tests :(
    # assert str(df_rolled.loc[df_rolled['id'] =='AXP', 'ref_date'].min()) == \
    #        str(pd.to_datetime(df.loc[df['id'] == 'AXP', 'datetime']).dt.date.sort_values().unique()[feature_window-1])
    # assert df_rolled.loc[(df_rolled.id == 'AXP') & (df_rolled['ref_date'] == pd.to_datetime('2017-01-09')), 'target_roll'] == 76.62

# def test_get_aggregated_df():
    # TODO