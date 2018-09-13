def crosscorr(datax, datay, lag=0):
    """ Lag-N cross correlation.
    Parameters
    ----------
    lag : int, default 0
    datax, datay : pandas.Series objects of equal length

    Returns
    ----------
    crosscorr : float
    """
    if lag >= 0:
        return datax.shift(lag).corr(datay)
    else:
        return datax.corr(datay.shift(-lag))

def sublens_volfilt(df1, df2, minvol):
    return df1.index[(df1.sum(1) > minvol) & (df2.sum(1) > minvol)]

def get_crosscor_df(df_gnip, df_sg, lags, sublens_list):
    corr_dict = {}
    for subLens in sublens_list:
        corr_list = []
        for lag in lags:
            corr_list.append(crosscorr(df_gnip.loc[subLens], df_sg.loc[subLens], lag))
        corr_dict[subLens] = corr_list
    df_crosscor = pd.DataFrame.from_dict(corr_dict, orient='index')
    df_crosscor.columns = lags
    return df_crosscor

def crosscorr_plot(df_crosscor, only_positive, top_n):
    if only_positive:
        df_crosscor_adj = df_crosscor.applymap(lambda x: 0 if x<=0 else x)
    else:
        df_crosscor_adj = df_crosscor
    if top_n:
        sns.heatmap(df_crosscor_adj.head(top_n), annot=False, yticklabels=True, cmap="YlGnBu")
    else:
        sns.heatmap(df_crosscor_adj, annot=False, yticklabels=True, cmap="YlGnBu")

def lag_plot(df_gnip, df_sg, subLens, lag, norm):
    df = pd.DataFrame([df_gnip.loc[subLens], df_sg.loc[subLens]]).T
    colnames = ['Gnip_' + subLens, 'SG_' + subLens]
    df.columns = colnames
    if lag >= 0:
        df[colnames[0]] = df[colnames[0]].shift(lag)
    elif lag < 0:
        df[colnames[1]] = df[colnames[1]].shift(-lag)
    if norm:
        df -= df.min()
        df /= df.max()
    df.plot(figsize=(15,10), style='.-', grid=True)




def granger_causation(x_cause, y_target, maxlag, crit_pval, verbose=False):
    # Get lags where timeseries x Granger causes timeseries y
    # Make stationary
    comb_stat = np.column_stack((y_target.diff(1).dropna(),
                                 x_cause.diff(1).dropna()))
    # Assume co-integration
    # Apply statsmodels granger causation to get dict of lag and significance
    granger = grangercausalitytests(comb_stat, maxlag, crit_pval, verbose)
    # Return lag_p and significance where significance is lowest
    sig_lags = []
    for lag in granger:
        pval = granger[lag][0]['ssr_ftest'][1]
        if pval <= crit_pval:
            sig_lags.append((lag, pval))
    return sig_lags


def get_granger_df(df1, df2, vol_thresh, make_stat, maxlag, crit_pval, verbose=False):
    x_vols = []
    y_vols = []
    lags = []
    subLens_list = []
    tot_lags = []
    for subLens in df1.index:
        x = df1.loc[subLens]
        y = df2.loc[subLens]
        if (x.sum() >= vol_thresh) & (y.sum() >= vol_thresh):
            sig_lags = granger_causation(x, y, make_stat, maxlag, crit_pval, verbose)
        else:
            sig_lags = []
        if len(sig_lags) > 0:
            lag = min([i[0] for i in sig_lags])
        else:
            lag = np.nan
        lags.append(lag)
        x_vols.append(x.sum())
        y_vols.append(y.sum())
        tot_lags.append(sig_lags)
        subLens_list.append(subLens)

    df_granger = pd.DataFrame([lags, x_vols, y_vols, subLens_list, tot_lags])
    df_granger = df_granger.T
    df_granger.columns = ['lag', 'source_volume', 'target_volume', 'subLens', 'total_lags']
    df_granger.set_index('subLens', inplace=True)
    return df_granger