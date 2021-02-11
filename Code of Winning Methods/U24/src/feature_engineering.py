import src.preprocessing as prep
import src.constants as constants
import src.ml_utils as ml_utils

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import lightgbm as lgb


def get_df_groups(sales):
    # All infos about ['graynualrity', 'groups'] and side variables
    cols_id = ['total_id', 'state_id', 'store_id', 'cat_id', 'dept_id']
    df_groups = sales[['granularity', 'group'] + cols_id].drop_duplicates()
    return df_groups


def get_df_loss_fe(sales):
    """
    We only need sales from historical data to compute df_loss_fe

    Returns : A dataframe with followings columns. In order to compute scaling and
    weighting in local loss.
    [date, group, sales_diff_cummean, revenu_sum_28]
    """
    # sales_diff_cummean - average sales diff between two days PRIOR forecast after first non 0 sale day
    # revenu_sum_28 - cumulative actual dollar sales on the last 28 observations of the training sample
    df_loss_fe = (
        sales[sales['sales'] > 0]
        .groupby('group', observed=True)
        ['date'].min().rename('first_sale_date')
        .reset_index().copy())
    df_loss_fe = pd.merge(sales[['date', 'group', 'granularity', 'sales', 'revenu']], df_loss_fe)
    df_loss_fe = df_loss_fe[df_loss_fe['date'] >= df_loss_fe['first_sale_date']].copy()
    # Due to the fact we replaced Christmas with NA, it creates side effects here :(
    # Avoid this by filling it back with 0
    grouped = df_loss_fe.pipe(lambda x: x.assign(sales=x['sales'].fillna(0))).groupby('group', observed=True)
    df_loss_fe['revenu_sum_28'] = (
        grouped['revenu']
        .transform(lambda x: x.rolling(28).sum())
        # .round(10) because rolling mean of positive floats produces small negative numbers
        .round(10)
    )

    # Weighted daatset according to the metric
    # Max equals to sales amount of total_X
    df_loss_fe['revenu_sum_28_max'] = df_loss_fe.groupby('date')['revenu_sum_28'].transform('max')

    df_loss_fe['sales_diff'] = (df_loss_fe['sales'].fillna(0) - grouped['sales'].shift(1)).abs()
    df_loss_fe['size'] = 1
    grouped = df_loss_fe.groupby('group', observed=True)
    df_loss_fe['sales_diff_cummean'] = grouped['sales_diff'].cumsum() / (grouped['size'].cumsum() - 1)

    # If fill NA after shift, remove -1 from grouped['size'].cumsum()
    # Depends if we want to take into account the 1st day into account or not
    # df_loss_fe['sales_diff'] = (df_loss_fe['sales'].fillna(0) - grouped['sales'].shift(1).fillna(0)).abs()
    # df_loss_fe['size'] = 1
    # grouped = df_loss_fe.groupby('group')
    # df_loss_fe['sales_diff_cummean'] = grouped['sales_diff'].cumsum() / (grouped['size'].cumsum())

    df_loss_fe['weight_revenu'] = df_loss_fe['revenu_sum_28'] / df_loss_fe['revenu_sum_28_max']
    df_loss_fe['weight'] = df_loss_fe['weight_revenu'] / df_loss_fe['sales_diff_cummean']

    assert df_loss_fe['weight'].max() != np.inf

    df_loss_fe = df_loss_fe[['date', 'group', 'weight']]
    df_loss_fe = df_loss_fe.rename(columns={'date': 'date_from'})
    return df_loss_fe


def fe_first_sale_date(sales):
    """
    Date of first sales for specific item's granularities.
    Used later on for contextual FE in fe_learning to compute item's age
    NOT USED for filtering data
    """
    l_granularity = [
        'item_id',
        'state_id_&_item_id',
        'item_id_&_store_id',
    ]

    first_sale_date = (
        sales
        .pipe(lambda x: x[x['granularity'].isin(l_granularity)])
        .groupby('group', observed=True)
        ['date'].min().rename('first_sale_date')
        .reset_index().copy()
    )

    # Data is censored, assume that if we have a sales during the first K days, it was probably sold before
    # Plot first sales distribution to have a better idea
    K = 56
    not_censored = first_sale_date['first_sale_date'] >= first_sale_date['first_sale_date'].min() + pd.Timedelta(days=K)
    first_sale_date = first_sale_date[not_censored]

    return first_sale_date


def fe_sales(sales):
    sales['date_dayofweek'] = sales['date'].dt.dayofweek
    sales['date_dayofyear'] = sales['date'].dt.dayofyear
    sales['date_year'] = sales['date'].dt.year.astype(str)
    # sales['date_month'] = sales['date'].dt.month
    sales['date_day'] = sales['date'].dt.day
    return sales


def get_fe_sales_historical(sales):
    """
    Returns :
    - the average sales from 1, 3, 7, 28 days
    - quantiles from 7, 28 days (TODO check this seems overkill)

    """
    fe_sales_historical = sales[['date', 'group', 'granularity']].copy()

    sales['sales_lop1p'] = np.log1p(sales['sales'])

    # Rolling mean
    grouped = sales.groupby('group', observed=True)
    list_window = [1, 3, 7, 14, 28, 30, 60, 56, 112]
    cols = ['sales', 'sell_price']
    for window in list_window:
        cols_rolling = [f'{col}_mean_{window}' for col in cols]
        # .round(10) because rolling mean of positive floats produces small negative numbers
        fe_sales_historical[cols_rolling] = grouped[cols].transform(
            lambda x: x.rolling(window, min_periods=1).mean()).round(10)

    # Rolling mean
    grouped = sales.groupby('group', observed=True)
    list_window = [28]
    cols = ['active_item']
    for window in list_window:
        cols_rolling = [f'{col}_mean_{window}' for col in cols]
        # .round(10) because rolling mean of positive floats produces small negative numbers
        fe_sales_historical[cols_rolling] = grouped[cols].transform(
            lambda x: x.rolling(window, min_periods=1).mean()).round(10)

    # Rolling log_mean
    list_window = [7, 14, 28]
    cols = ['sales_lop1p']
    for window in list_window:
        cols_rolling = [f'{col}_mean_{window}' for col in cols]
        # .round(10) because rolling mean of positive floats produces small negative numbers
        fe_sales_historical[cols_rolling] = grouped[cols].transform(
            lambda x: x.rolling(window, min_periods=1).mean()).round(10)

    # Rolling std
    # Ugly, make it cleaner later
    list_window = [28, 56, 112]
    cols = ['sales']
    for window in list_window:
        cols_rolling = [f'{col}_std_{window}' for col in cols]
        # .round(10) because rolling mean of positive floats produces small negative numbers
        fe_sales_historical[cols_rolling] = grouped[cols].transform(
            lambda x: x.rolling(window, min_periods=1).std()).round(10)

    # rolling quantiles
    list_window = [28, 56, 112]
    for window in list_window:
        # All quantiles might be overkill, espcially 0.005 ~= 0.025 when window is small
        # TODO remplace 0.005 & 0.025 by 0, and opposite by 1 (min, max)
        for quantile in constants.LIST_QUANTILE:
            fe_sales_historical[f'sales_q{quantile}_{window}'] = (
                grouped['sales'].transform(lambda x: x.rolling(window, min_periods=1).quantile(quantile))).round(10)

    # Rolling trend (based on rolling mean)
    grouped = fe_sales_historical.groupby('group', observed=True)

    cols = ['sales']
    list_window = [7]
    for window in list_window:
        cols_rolling = [f'{col}_mean_{window}' for col in cols]
        cols_trend = [f'{col}_mean_{window}_on_{window}' for col in cols]
        fe_sales_historical[cols_trend] = (
            fe_sales_historical[cols_rolling] / grouped[cols_rolling].shift(window).replace(0, 1))

    fe_sales_historical = fe_sales_historical.rename(columns={'date': 'date_from'})
    return fe_sales_historical


def get_fe_sales_historical_hierarchical(fe_sales_historical):
    """
    Copy sales_mean_7_on_7 from fe_sales_historical for certain granularity and pass it to lower hierarchies.
    Gives big picture context to lower hierarchies.
    Eg : if an item_id has a growth in sales, it can be luck (noise) but if the whole store_id has a growth,
    it is signal.

    Returns a dict of dataframe ready to be merged.
    dataframe example : date_from, granularity, store_id, store_id_sales_mean_7_on_7
    """
    tmp = fe_sales_historical[['date_from', 'group', 'granularity', 'sales_mean_7_on_7']].copy()
    fe_sales_historical_hierarchical = {}
    for granularity in ['state_id', 'store_id']:
        fe_sales_historical_hierarchical_tmp = tmp[tmp['granularity'] == granularity]
        fe_sales_historical_hierarchical_tmp['group'] = fe_sales_historical_hierarchical_tmp['group'].str.replace(
            '_X$', '')
        fe_sales_historical_hierarchical_tmp = (
            fe_sales_historical_hierarchical_tmp
            .rename(columns={
                'sales_mean_7_on_7': f'{granularity}_sales_mean_7_on_7',
                'group': granularity,
                'granularity': 'initial_granularity',
            })
        )

        # Remove yourself from joining
        fe_sales_historical_hierarchical_tmp = pd.merge(
            fe_sales_historical_hierarchical_tmp.eval('key = 1'),
            fe_sales_historical[['granularity']].drop_duplicates().eval('key = 1')
        ).drop(columns='key')
        cond = (
            fe_sales_historical_hierarchical_tmp['initial_granularity'] !=
            fe_sales_historical_hierarchical_tmp['granularity'])
        fe_sales_historical_hierarchical_tmp = (
            fe_sales_historical_hierarchical_tmp[cond])
        fe_sales_historical_hierarchical_tmp = fe_sales_historical_hierarchical_tmp.drop(columns=[
                                                                                         'initial_granularity'])

        fe_sales_historical_hierarchical[f'hierarchical_sales_{granularity}'] = fe_sales_historical_hierarchical_tmp

    return fe_sales_historical_hierarchical


def get_fe_sales_historical_dow(sales):
    """
    Returns the average value from same day of week
    """
    fe_sales_historical_dow = sales[['date', 'group', 'date_dayofweek']].copy()
    grouped = sales.groupby('group', observed=True)
    list_window = [1, 2, 3, 4]
    for window in list_window:
        # .round(10) because rolling mean of positive floats produces small negative numbers
        fe_sales_historical_dow['sales_dow_mean_{}'.format(window)] = (
            grouped['sales'].transform(lambda x: x.rolling(window, min_periods=1).mean())).round(10)

    fe_sales_historical_dow = fe_sales_historical_dow.rename(columns={'date': 'date_from'})
    return fe_sales_historical_dow


def get_fe_sales_historical_last_date(sales):
    """
    Returns the number of days since there were no purchase
    """
    fe_sales_historical_last_date = sales.query('sales > 0')[['date', 'group']].copy()
    grouped = fe_sales_historical_last_date.groupby('group', observed=True)
    fe_sales_historical_last_date['days_since_last_sale'] = (
        fe_sales_historical_last_date['date'] - grouped['date'].shift(1)).dt.days
    fe_sales_historical_last_date = fe_sales_historical_last_date.rename(columns={'date': 'date_from'})
    return fe_sales_historical_last_date


def fe_dayofyear_trend(fe_sales_historical):
    """
    Based on fe_sales_historical, returns out-of-fold yearly trend given horizon
    and date_dayofyear.

    TODO improve description.
    TODO still improvable. Eg : 2016 is a bisextile year, might fuck up a little bit.
    Idea : use group [month, day of month]?

    It is extremely difficult for the algo to detect YEARLY (more sales in June)
    and MONTHLY (begin vs end of month) SEASONNALITY.
    compared to the last 28 days of sales (our strongest feature).

    Predicting day t depends of :
    1) a - on which days month there were computed (eg based on may and predict june)
    b - on which 28 days of the month there were computed (if it includes the first saturday/sunday of the month)
    2) the horizon of the forecast

    It is a 2-D variable problem which is extremely difficult to deal for trees.
    This trick makes it 1-D.

    For 1 - a and 1 - b we also target encode impact of this days in a day.
    """
    dates = pd.read_csv('data/calendar.csv')[['date']].drop_duplicates()  # Yes ugly
    dates['date'] = pd.to_datetime(dates['date'])

    l_granularity = [
        'total_id',
        'state_id', 'store_id',
        'cat_id', 'dept_id',
        'store_id_&_cat_id', 'state_id_&_cat_id',
        'state_id_&_dept_id', 'store_id_&_dept_id',
        # Granularities below are not computable doable
        # 'item_id',
        # 'state_id_&_item_id',
        # 'item_id_&_store_id',
    ]

    trend_from = fe_sales_historical[fe_sales_historical['granularity'].isin(l_granularity)]
    col_target_period = 'sales_mean_7'
    # Strongest features, we want to unbias
    cols_to_unbias = ['sales_mean_30', 'sales_mean_28', 'sales_mean_14', 'sales_mean_7']
    trend = trend_from[['date_from', 'group', 'granularity', col_target_period]].copy()
    trend_from = trend_from[['date_from', 'group', 'granularity'] + cols_to_unbias].copy()

    trend = trend.rename(columns={
        'date_from': 'date',
        col_target_period: 'target_period',
    })
    # Center it (previously aligned to left) - depends of the window choosen in col_target_period
    trend['date'] = trend['date'] + pd.DateOffset(days=4)

    # Data augmentation - handles all combinaison
    horizon = pd.DataFrame({'horizon': range(1, 29)})
    trend = pd.merge(
        trend.eval('key=1'),
        horizon.eval('key=1'),
    ).drop(columns='key')

    trend['date_from'] = trend['date'] - pd.to_timedelta(trend['horizon'], unit='d')
    trend_from = trend_from.replace(0, np.nan)
    trend = pd.merge(trend, trend_from)
    del trend_from

    for col in cols_to_unbias:
        trend[f'{col}_dayofyear_trend'] = trend['target_period'] / trend[col]

    cols_dayofyear_trend = [f'{col}_dayofyear_trend' for col in cols_to_unbias]

    # Complete with dates from test/prod
    trend = pd.merge(dates, trend, how='left')

    trend['date_dayofyear'] = trend['date'].dt.dayofyear
    trend['date_year'] = trend['date'].dt.year.astype(str)

    def avg_dayofyear_trend(trend):
        res = (
            trend
            .groupby(['date_dayofyear', 'horizon', 'granularity', 'group'], observed=True)
            [cols_dayofyear_trend].mean().reset_index()
        )
        return res

    # Out of fold to avoid any leak
    yearly_trend = ml_utils.oof_dataframe(data=trend, col_fold='date_year', function=avg_dayofyear_trend)
    return yearly_trend


def get_df_hierarchical(df, df_name, df_groups, cols_group, cols_fe):
    """
    Propagate cols_fe to lower granularities.
    Gives big picture context to lower hierarchies

    Returns a dict of dataframe ready to be merged.

    TODO improve this to do give it to higher hierarchies but only to lower
    because we are creating lines which can't exists (eg state into total granularity)
    Slowest part of get_learning
    """
    cols_id = ['total_id', 'state_id', 'store_id', 'cat_id', 'dept_id']

    # We needs cols_id in df, problem is we don't
    # I think we should enforce df to have theses columns
    df_enriched = pd.merge(df, df_groups)
    df_hierarchical = {}
    for granularity in df_enriched['granularity'].unique():
        # print(granularity)
        df_hierarchical_tmp = df_enriched[df_enriched['granularity'] == granularity]

        # cols_id for this granularity
        cols_id_tmp = df_hierarchical_tmp[cols_id].isnull().sum() == 0
        cols_id_tmp = cols_id_tmp[cols_id_tmp == 1].index.tolist()
        # print(granularity, 'associated id columns : ',cols_id_tmp)

        df_hierarchical_tmp = df_hierarchical_tmp[cols_group + cols_id_tmp + cols_fe]

        for col in cols_fe:
            df_hierarchical_tmp = df_hierarchical_tmp.rename(
                columns={col: f'{granularity}_{col}'})

        df_hierarchical[f'hierarchical_{df_name}_{granularity}'] = df_hierarchical_tmp

    return df_hierarchical


def fe_dayofyear_trend_hierarchical(yearly_trend, df_groups):
    """
    Propagate yearly_trend to lower granularities.
    Gives big picture context to lower hierarchies.
    Eg : if an item_id has a growth in sales, it can be luck (noise) but if the whole store_id has a growth,
    it is signal.

    Returns a dict of dataframe ready to be merged.
    """

    yearly_trend_hierarchical = get_df_hierarchical(
        df=yearly_trend,
        df_name='yearly_trend',
        df_groups=df_groups,
        cols_group=['date_year', 'date_dayofyear', 'horizon'],
        cols_fe=['sales_mean_28_dayofyear_trend'])

    return yearly_trend_hierarchical


def fe_yearly_trend_day_hierarchical(yearly_trend_day, df_groups):
    """
    Propagate yearly_trend_day to lower granularities.
    Gives big picture context to lower hierarchies.
    Eg : if an item_id has a growth in sales, it can be luck (noise) but if the whole store_id has a growth,
    it is signal.

    Returns a dict of dataframe ready to be merged.
    """

    yearly_trend_day_hierarchical = get_df_hierarchical(
        df=yearly_trend_day,
        df_name='yearly_trend_day',
        df_groups=df_groups,
        cols_group=[
            'date_year',
            'date_from_month',
            'date_from_day',
            'date_month',
            'date_dayofweek',
            'date_month_week',
        ],
        cols_fe=['sales_mean_28_yearly_trend_day'])

    return yearly_trend_day_hierarchical


def fe_monthly_trend_day_hierarchical(monthly_trend_day, df_groups):
    """
    Propagate yearly_trend_day to lower granularities.
    Gives big picture context to lower hierarchies.
    Eg : if an item_id has a growth in sales, it can be luck (noise) but if the whole store_id has
    a growth, it is signal.

    Returns a dict of dataframe ready to be merged.
    """

    monthly_trend_day_hierarchical = get_df_hierarchical(
        df=monthly_trend_day,
        df_name='monthly_trend_day',
        df_groups=df_groups,
        cols_group=[
            'date_year',
            'date_from_day',
            'date_dayofweek',
            'date_month_week',
        ],
        cols_fe=['sales_mean_28_monthly_trend_day'])

    return monthly_trend_day_hierarchical


def fe_dates_for_yearly_trend_day(df):
    df['date_month_week'] = df['date'].dt.day // 7
    # We have 4 full weeks and 2/3 days, rattach them to 4th week, avoid NA and stuff
    df['date_month_week'] = df['date_month_week'].clip(0, 3)
    df['date_dayofweek'] = df['date'].dt.dayofweek
    df['date_month'] = df['date'].dt.month
    df['date_year'] = df['date'].dt.year.astype(str)
    df['date_from_day'] = df['date_from'].dt.day
    df['date_from_month'] = df['date_from'].dt.month

    # Replace date_from 29 February by 28, avoid NA values
    is_february_29 = (df['date_from_month'] == 2) & (df['date_from_day'] == 29)
    df.loc[is_february_29, 'date_from_day'] = 28
    return df


def fe_yearly_trend_day(fe_sales_historical, sales):
    """
    Same as get_yearly_trend but instead of day of year
    based on [date_from_month, date_from_day, date_month, date_dayofweek, date_month_week]
    which is expected to be more robust

    Intuition : we want to encode trend for the Nth day of week of the month
    Eg : First (can be 1 to 7) Sunday of december 201X has the same "trend"
    than other years even if it is not the same dayofmonth
    """
    dates = pd.read_csv('data/calendar.csv')[['date']].drop_duplicates()  # Yes ugly
    dates['date'] = pd.to_datetime(dates['date'])

    l_granularity = [
        'total_id',
        'state_id', 'store_id',
        'cat_id', 'dept_id',
        'store_id_&_cat_id', 'state_id_&_cat_id',
        'state_id_&_dept_id', 'store_id_&_dept_id',
        # 'item_id',
        # 'state_id_&_item_id', 'item_id_&_store_id',
    ]

    # Strongest features, we want to unbias
    cols_to_unbias = ['sales_mean_30', 'sales_mean_28', 'sales_mean_14', 'sales_mean_7']

    trend_from = (
        fe_sales_historical[fe_sales_historical['granularity'].isin(l_granularity)]
        [['date_from', 'group', 'granularity'] + cols_to_unbias]
        .copy())
    col_target_period = 'sales'
    trend = (
        sales[sales['granularity']
              .isin(l_granularity)]
        [['date', 'group', 'granularity', col_target_period]]
        .drop_duplicates()
        .copy())

    # Data augmentation - handles all combinaison
    # yes this code is tricky, you have NA values otherwise
    horizon = pd.DataFrame({'horizon': range(-2, 32)})
    trend = pd.merge(
        trend.eval('key=1'),
        horizon.eval('key=1'),
    ).drop(columns='key')

    trend['date_from'] = trend['date'] - pd.to_timedelta(trend['horizon'], unit='d')
    trend = pd.merge(trend, trend_from)

    for col in cols_to_unbias:
        trend[f'{col}_yearly_trend_day'] = trend['sales'] / trend[col]

    cols_yearly_trend = [f'{col}_yearly_trend_day' for col in cols_to_unbias]

    trend = fe_dates_for_yearly_trend_day(trend)

    def avg_yearly_trend(trend):
        res = (
            trend
            .groupby(['date_from_month', 'date_from_day', 'date_month', 'date_dayofweek',
                      'date_month_week', 'granularity', 'group'], observed=True)
            [cols_yearly_trend].mean().reset_index()
        )
        return res

    # Out of fold to avoid any leak
    yearly_trend_day = ml_utils.oof_dataframe(data=trend, col_fold='date_year', function=avg_yearly_trend)
    return yearly_trend_day


def fe_monthly_trend_day(yearly_trend_day):
    cols = [
        #  'date_from_month',
        'date_from_day',
        #  'date_month',
        'date_dayofweek',
        'date_month_week',
        'granularity',
        'group',
        #  'sales_mean_28_yearly_trend_day',
        'date_year'
    ]

    cols_to_unbias = ['sales_mean_30', 'sales_mean_28', 'sales_mean_14', 'sales_mean_7']
    cols_fe = [f'{col}_yearly_trend_day' for col in cols_to_unbias]
    cols_fe_rename = [f'{col}_monthly_trend_day' for col in cols_to_unbias]

    monthly_trend_day = (
        yearly_trend_day
        .groupby(cols, observed=True)
        [cols_fe].mean()
        # .reset_index()
    )
    monthly_trend_day.columns = cols_fe_rename
    monthly_trend_day = monthly_trend_day.reset_index()
    return monthly_trend_day


def get_learning(sales, dict_df_fe_merge, dict_df_fe_merge_asof):
    print('get_learning')
    df_learning = sales

    # Specific FE on dates to allow later merging with yearly_trend_day
    df_learning = fe_dates_for_yearly_trend_day(df_learning)

    nrow_begin = df_learning.shape[0]

    for fe_name, df_fe in dict_df_fe_merge.items():
        print('  Merge', fe_name)
        df_learning = pd.merge(df_learning, df_fe, how='left')
        assert nrow_begin == df_learning.shape[0]

    # Why isn't it already done before?
    df_learning = df_learning.sort_values('date_from')

    for fe_name, df_fe in dict_df_fe_merge_asof.items():
        print('  Merge', fe_name)
        col_time = 'date_from'
        df_fe = df_fe.sort_values(col_time)
        cols_by = list(df_learning.columns.intersection(df_fe.columns))
        cols_by.remove(col_time)
        df_learning = pd.merge_asof(
            df_learning,
            df_fe,
            on=col_time,
            by=cols_by)
        assert nrow_begin == df_learning.shape[0]

    return df_learning


def fe_learning(df_learning):
    # Contextual feature engineering, current price wiht previous average price (computed in get_fe_sales_historical())
    # % of "promo"
    # Maybe list_windows and col_rolling should be shared from

    col_trend_tuple = [
        ('sell_price', [3, 7, 14, 28, 56, 112]),
        ('active_item', [28]),
    ]

    for col, list_window in col_trend_tuple:
        for window in list_window:
            col_rolling = '{col}_mean_{window}'.format(col=col, window=window)
            # I'm not convinced by this naming convention
            col_contextual = '{col}_{col_rolling}_div'.format(col=col, col_rolling=col_rolling)
            df_learning[col_contextual] = df_learning[col] / df_learning[col_rolling]
            # The raw feature is useless, just drop it
            df_learning = df_learning.drop(columns=col_rolling)

    df_learning['item_age_days'] = (df_learning['date_from'] - df_learning['first_sale_date']) / pd.Timedelta(days=1)
    # Knowing when is was introduced might help (there is a survival biais, old product are only top sellers products )
    df_learning['first_sale_date_num'] = (df_learning['first_sale_date'] -
                                          pd.Timestamp('2011-01-01 00:00:00')).dt.total_seconds()
    df_learning = df_learning.drop(columns='first_sale_date')

    return df_learning


def strongest_event(df_learning):
    # Find the event_power for each event_name
    calendar_event = prep.read_calendar_event()
    calendar_event['date'] = pd.to_datetime(calendar_event['date'])
    event_power = pd.merge(df_learning[['date', 'granularity', 'group',
                                        'event_trend', 'pred', 'sales']], calendar_event)

    # res = df_learning_full.groupby(['granularity', 'group', 'event_name'], observed=True)[
    #     'event_trend'].agg(['mean', 'median']).reset_index()

    event_power = event_power.groupby('event_name')['event_trend'].agg(['mean', 'median']).reset_index()
    # Assign minimal absolute effect, acts an normalization
    min_effect = (event_power[['mean', 'median']] - 1).abs().idxmin(axis=1)
    event_power['event_trend'] = np.where(min_effect == 'median', event_power['median'], event_power['mean'])
    event_power['event_trend'] = event_power[['mean', 'median']].min(axis=1)
    event_power = event_power.drop(columns=['mean', 'median'])

    # event_power = event_power.groupby('event_name', observed=True)['event_trend'].median().reset_index()
    # event_power = event_power.groupby('event_name', observed=True)['pred', 'sales'].sum().reset_index()
    # event_power['event_trend'] = event_power['pred'] / event_power['sales']

    event_power['event_power'] = (1 - event_power['event_trend']).abs()
    event_power = event_power[['event_name', 'event_power']]
    event_power = pd.merge(calendar_event, event_power)
    event_power = event_power.sort_values('event_power', ascending=False)
    # event_power = event_power.drop_duplicates(['date'], keep='first')
    # event_power = event_power.drop(columns='event_power')
    event_power = event_power.sort_values('date')
    return event_power


def add_date_from_prod(df_learning):
    # Add date_from_test - we need them for out-of-fold
    # Otherwise they won't exists
    calendar = pd.read_csv('data/calendar.csv')
    calendar['date'] = pd.to_datetime(calendar['date'])
    calendar['date_year'] = calendar['date'].dt.year.astype(str)
    calendar = calendar[['date', 'date_year']]

    # Right join because we also needs dates from test/prod
    df_learning = pd.merge(df_learning, calendar, how='right')
    # Fill missing group/granularity which is unique

    # Clumsy but keep Categorical dtype!
    df_learning['granularity'] = df_learning['granularity'].fillna(df_learning['granularity'].dropna().unique()[0])
    df_learning['group'] = df_learning['group'].fillna(df_learning['group'].dropna().unique()[0])
    return df_learning


def _fe_event_trend(df_learning):
    """
    Compute average event trend (effect) for calenar_events.
    To do so, we train a model based on our features. Then we consider error ratio (pred / real) as the explained trend.
    We compute the out-of-fold trend before assigning it in order to avoid any data leakage.
    """

    # We are convinced that normalizing is key and make our model work better.
    # Therefore also apply this normalization step here to modelize event's impact/trend
    normalize_by = 'sales_mean_28'
    assert (df_learning['sales_mean_28'].fillna(0) != 0).all()
    df_learning = prep.preprocessing_learning(df_learning, normalize_by=normalize_by, is_train=True)

    to_drop = [
        'item_id',
        'dept_id',
        'cat_id',
        'store_id',
        'state_id',
        'total_id',
        'granularity',
        'group',
        'sales',
        'date',
        'date_from',
        'revenu',
        # 'sell_price',
        'weight',
        'date_dayofyear',
        'date_year',
        'event_name'
    ]

    y = 'sales'
    y_learning = df_learning[y]
    assert y_learning.isnull().sum() == 0
    X_cols = list(df_learning.head().drop(columns=to_drop + [y], errors='ignore'))

    param = {
        'objective': 'regression',
        'learning_rate': 0.05,
        'min_sum_hessian_in_leaf': 0.0,
        'verbose': -1}
    num_boost_round = 500
    early_stopping_rounds = 100

    # Perform 2 CV in order to make event_trend more accurate/stable (stacking)
    df_learning['pred'] = 0
    iterator = range(1, 3)
    for i in iterator:
        # Modeling
        folds = KFold(n_splits=5, shuffle=True, random_state=i)
        oof_preds = np.zeros(df_learning.shape[0])
        for train_index, val_index in folds.split(df_learning):
            X_train, y_train = df_learning[X_cols].iloc[train_index], y_learning.iloc[train_index]
            X_valid, y_valid = df_learning[X_cols].iloc[val_index], y_learning.iloc[val_index]

            lgb_train = lgb.Dataset(
                X_train,
                label=y_train,
                free_raw_data=False,
            )
            lgb_valid = lgb.Dataset(
                X_valid,
                label=y_valid,
                free_raw_data=False,
            )

            model_lgb = lgb.train(
                params=param,
                train_set=lgb_train,
                num_boost_round=num_boost_round,
                valid_sets=[lgb_train, lgb_valid],
                early_stopping_rounds=early_stopping_rounds,
                verbose_eval=False,
            )

            oof_preds[val_index] = model_lgb.predict(X_valid)

        df_learning['pred'] = df_learning['pred'] + (oof_preds / len(iterator))

    df_learning['event_trend'] = df_learning['pred'] / df_learning['sales']

    event_power = strongest_event(df_learning)

    df_learning_full = add_date_from_prod(df_learning)
    df_learning_full = pd.merge(df_learning_full, event_power, how='left')
    df_learning_full['event_name'] = df_learning_full['event_name'].fillna('')
    df_learning_full['event_name'] = pd.Categorical(df_learning_full['event_name'])

    df_learning_full['event_power'] = df_learning_full['event_power'].fillna(0)
    # Deduplicate events
    df_learning_full = df_learning_full.sort_values(
        'event_power', ascending=False).drop_duplicates('date', keep='first')

    # Out of fold to avoid any leak
    def avg_event_trend(trend):
        res = trend.groupby(['granularity', 'group', 'event_name'], observed=True)[
            'event_trend'].agg(['mean', 'median']).reset_index()

        # Assign minimal absolute effect, acts as normalization
        min_effect = (res[['mean', 'median']] - 1).abs().idxmin(axis=1)
        res['event_trend'] = np.where(min_effect == 'median', res['median'], res['mean'])
        res = res.drop(columns=['mean', 'median'])
        return res

    event_trend = ml_utils.oof_dataframe(data=df_learning_full, col_fold='date_year', function=avg_event_trend)
    event_trend = pd.merge(
        df_learning_full[['date', 'date_year', 'granularity', 'group', 'event_name', 'event_power']].drop_duplicates(),
        event_trend)

    event_trend = event_trend.sort_values('date')
    event_trend = event_trend.drop(columns='event_power')

    return event_trend


def fe_event_trend_group(df_learning):
    """
    Same as fe_event_trend but for each group.
    Hypothesis : Trend is different for each state, dept (eg : foods is not impacted the say way fo thanksgiving
    than hoobies.)
    """
    l_granularity = [
        'total_id',
        'state_id', 'store_id',
        'cat_id',
        'dept_id',
        'store_id_&_cat_id',
        'state_id_&_cat_id',
        'state_id_&_dept_id',
        'store_id_&_dept_id',
        # 'item_id',
        # 'state_id_&_item_id',
        # 'item_id_&_store_id',
    ]

    df_learning = df_learning[df_learning['granularity'].isin(l_granularity)]
    df_learning = df_learning[df_learning['horizon'] == 1]

    event_trend_group = []

    for group in df_learning['group'].unique().tolist():
        event_trend_tmp = _fe_event_trend(df_learning.pipe(lambda df: df[df['group'] == group]))
        event_trend_group.append(event_trend_tmp)

    event_trend_group = pd.concat(event_trend_group)

    return event_trend_group


def get_event_trend_hierarchical(event_trend_group, df_groups):
    """
    Propagate yearly_trend to lower granularities.
    Gives big picture context to lower hierarchies.
    Eg : if an item_id has a growth in sales, it can be luck (noise) but if the whole store_id has a growth,
    it is signal.

    Returns a dict of dataframe ready to be merged.
    """
    event_trend_hierarchical = get_df_hierarchical(
        df=event_trend_group, df_name='event_trend', df_groups=df_groups,
        cols_group=['date', 'date_year'], cols_fe=['event_trend'])
    return event_trend_hierarchical


def merge_event_trend_hierarchical(df_learning, event_trend_hierarchical):
    nrow_begin = df_learning.shape[0]
    for fe_name, df_fe in event_trend_hierarchical.items():
        print('  Merge', fe_name)
        df_learning = pd.merge(df_learning, df_fe, how='left')
        assert nrow_begin == df_learning.shape[0]
    return df_learning


def get_weather():
    weather = []
    files_weather = (
        ('CA', 'data/weather/californiaw.csv'),
        ('TX', 'data/weather/texasw.csv'),
        ('WI', 'data/weather/wisconsinw.csv'),
    )
    for state_id, fname in files_weather:
        weather_tmp = pd.read_csv(fname)
        weather_tmp['state_id'] = state_id
        if state_id == 'WI':
            weather_tmp['date'] = pd.to_datetime(weather_tmp['date_time'], format='%d-%m-%y')
        else:
            weather_tmp['date'] = pd.to_datetime(weather_tmp['date_time'])

        weather.append(weather_tmp)

    weather = pd.concat(weather)
    # weather['date'] = pd.to_datetime(weather['date_time'])
    weather['temperature'] = weather['FeelsLikeC']
    weather = weather[['date', 'state_id', 'temperature']]

    # Rolling mean
    grouped = weather.groupby('state_id')
    list_window = [7, 14, 28]
    col = 'temperature'
    for window in list_window:
        col_rolling = f'{col}_mean_{window}'
        # .round(10) because rolling mean of positive floats produces small negative numbers
        weather[col_rolling] = grouped[col].transform(
            lambda x: x.rolling(window, min_periods=1).mean()).round(10)

        weather[f'temperature_{col_rolling}_diff'] = weather['temperature'] - weather[col_rolling]
        weather = weather.drop(columns=col_rolling)

    weather = weather.rename(columns={'date': 'date_from'})
    return weather


def get_fe_weather():
    """
    Returns the temperature difference between date_from (date of the predition) and the average
    temperature of the last 7, 14, 28. Used to unbias previous sales.
    """
    weather = get_weather()

    cols_fe = ['temperature',
               'temperature_temperature_mean_7_diff',
               'temperature_temperature_mean_14_diff',
               'temperature_temperature_mean_28_diff']
    rename_cols_fe = {}
    for col in cols_fe:
        rename_cols_fe[col] = f'state_id_{col}'

    weather_state = weather.rename(columns=rename_cols_fe)

    weather_total = weather.groupby('date_from').mean().reset_index()
    weather_total['total_id'] = 'Total'
    rename_cols_fe = {}
    for col in cols_fe:
        rename_cols_fe[col] = f'total_id_{col}'

    weather_total = weather_total.rename(columns=rename_cols_fe)
    return weather_state, weather_total
