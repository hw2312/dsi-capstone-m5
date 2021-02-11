import functools
import numpy as np
import pandas as pd
import src.constants as constants
import src.path as path
import src.ml_utils as ml_utils

from workalendar.usa.texas import Texas
from workalendar.usa.california import California
from workalendar.usa.wisconsin import Wisconsin
import hashlib


def aggregate_sales_group(sales, is_train):
    sales['active_item'] = 1

    agg_func = {}
    agg_func['sell_price'] = 'mean'
    agg_func['snap'] = 'mean'
    agg_func['active_item'] = 'sum'
    if is_train:
        agg_func['sales'] = 'sum'
        agg_func['revenu'] = 'sum'

    # Preprocessing
    sales['total_id'] = 'Total'

    l_cols_group = [
        ['total_id'],
        ['state_id'],
        ['store_id'],
        ['cat_id'],
        ['dept_id'],
        ['state_id', 'cat_id'],
        ['state_id', 'dept_id'],
        ['store_id', 'cat_id'],
        ['store_id', 'dept_id'],
        ['item_id'],
        ['state_id', 'item_id'],
        ['item_id', 'store_id'],
    ]

    cols_id = ['item_id', 'cat_id', 'store_id', 'state_id', 'dept_id', 'total_id']

    sales_group = []
    for cols_group in l_cols_group:
        print(cols_group)

        # If we have a many-to-one mapping between cols_group and an id columns, add it as side variable
        tmp_cols_id = [col for col in cols_id if col not in cols_group]

        tmp = sales.groupby(cols_group, observed=True)[tmp_cols_id].transform('first')
        cols_unique = (sales[tmp_cols_id] == tmp).all()
        cols_unique = list(cols_unique[cols_unique].index)
        if len(cols_unique) > 0:
            print('Have a many-to-one mapping with', ','.join(cols_unique))

        sales_group_tmp = sales.groupby(['date'] + cols_group + cols_unique).agg(agg_func).reset_index()
        sales_group_tmp['group'] = sales_group_tmp[cols_group].apply(lambda x: '_'.join(x), axis=1)
        if len(cols_group) == 1:
            sales_group_tmp['group'] = sales_group_tmp['group'] + '_X'

        sales_group_tmp['granularity'] = '_&_'.join(cols_group)
        print('Number of series', sales_group_tmp['group'].nunique())
        sales_group.append(sales_group_tmp)
        print('\n')

    sales_group = pd.concat(sales_group, ignore_index=True)
    return sales_group


def read_calendar_event():
    """
    Create a calendar with event_name for all dates key (date, event)
    """
    calendar = pd.read_csv('data/calendar.csv')

    pentecost_dates = [
        "2011-06-12", "2012-05-27", "2013-05-19", "2014-06-08",
        "2015-05-24", "2016-05-15",
    ]
    pentecost_dates = pd.DataFrame({'date': pentecost_dates})
    pentecost_dates['event_name_3'] = 'pentecost_dates'

    orthodox_pentecost_dates = [
        "2011-06-12", "2012-06-03", "2013-06-23", "2014-06-08",
        "2015-05-31", "2016-06-19",
    ]
    orthodox_pentecost_dates = pd.DataFrame({'date': orthodox_pentecost_dates})
    orthodox_pentecost_dates['event_name_4'] = 'orthodox_pentecost_dates'

    # Always at least 4 games - we comment >4 because we don't know at the time we predict
    nba_finals_dates = [
        "2011-05-31", "2011-06-02", "2011-06-05", "2011-06-07",
        "2011-06-09", "2011-06-12",
        "2012-06-12", "2012-06-14", "2012-06-17", "2012-06-19",
        "2012-06-21",
        "2013-06-06", "2013-06-09", "2013-06-11", "2013-06-13",
        "2013-06-16", "2013-06-18", "2013-06-20",
        "2014-06-05", "2014-06-08", "2014-06-10", "2014-06-12",
        "2014-06-15",
        "2015-06-04", "2015-06-07", "2015-06-09", "2015-06-11",
        "2015-06-14", "2015-06-16",
        "2016-06-02", "2016-06-05", "2016-06-08", "2016-06-10",
        # "2016-06-13", "2016-06-16", "2016-06-19",
    ]
    nba_finals_dates = pd.DataFrame({'date': nba_finals_dates})
    nba_finals_dates['event_name_5'] = 'NBA_Finals'

    calendar_additionnal = [
        pentecost_dates,
        orthodox_pentecost_dates,
        # nba_finals_dates,
    ]

    calendar_additionnal = functools.reduce(lambda x, y: pd.merge(x, y, how='outer'), calendar_additionnal)

    nrow_begin = calendar.shape[0]
    calendar = pd.merge(calendar, calendar_additionnal, how='left')
    assert calendar.shape[0] == nrow_begin

    # Created a shifted calendar (eg : day before/after Christmas)
    default_calendar = False
    if default_calendar:
        cols_event = [f'event_name_{i}' for i in range(1, 2)]
    else:
        # use calendar_additionnal too
        max_event_name = 4  # => Use only event_name_1, 2, sort NBA finals end and Father's day
        cols_event = [f'event_name_{i}' for i in range(1, max_event_name + 1)]

    calendar_event = calendar[['date'] + cols_event].copy()
    for col in cols_event:
        SHIFT_SIZE = 2  # When testing 2 is better than only 1, and 3
        for i in range(-SHIFT_SIZE, SHIFT_SIZE + 1):
            calendar_event[f'{col}_{i}'] = (calendar_event[col] + f'_{i}').shift(i)

    if default_calendar:
        calendar_event = calendar_event.fillna(method='bfill', axis=1)
        calendar_event['event_name'] = calendar_event['event_name_1']
        calendar_event = calendar_event[['date', 'event_name']]
    else:
        calendar_event = calendar_event.drop(columns=cols_event)
        calendar_event = pd.melt(calendar_event, id_vars='date', value_name='event_name')
        calendar_event['event_name'] = calendar_event['event_name'].fillna('')
        calendar_event = calendar_event.drop(columns='variable')
        calendar_event = calendar_event.drop_duplicates()
        calendar_event = calendar_event.sort_values('date')
    return calendar_event


def read_calendar_state():
    """
    Read calendar, make it tidy then return it.

    event_name are not returned. Due to event's prioritirizing strategy is subjective, we separate this.
    It will be treated in another part.
    """
    calendar = pd.read_csv('data/calendar.csv')
    value_vars = ['snap_CA', 'snap_TX', 'snap_WI']
    id_vars = list(calendar.drop(columns=value_vars))
    calendar_state = pd.melt(
        calendar,
        id_vars=id_vars,
        value_vars=value_vars,
        var_name='state_id',
        value_name='snap',
    )
    calendar_state['state_id'] = calendar_state['state_id'].str.split('_').str[1]
    calendar_state = calendar_state[['date', 'd', 'wm_yr_wk', 'state_id', 'snap']]

    workalendar_state = {'CA': California(), 'TX': Texas(), 'WI': Wisconsin()}

    def state_is_working_day(state, date):
        return workalendar_state[state].is_holiday(date)

    state_is_working_day = np.vectorize(state_is_working_day)
    calendar_state['is_working_holiday'] = state_is_working_day(
        calendar_state['state_id'], pd.to_datetime(calendar_state['date']).dt.date).astype(int)

    return calendar_state


def extract_sales():
    # Old dataset
    # sales_raw = pd.read_csv('data/sales_train_validation.csv')
    # New dataset
    sales_raw = pd.read_csv('data/sales_train_evaluation.csv')
    sell_prices = pd.read_csv('data/sell_prices.csv')
    calendar_state = read_calendar_state()

    # Tidy format, to be able to merge with sell_prices
    cols_id = ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
    sales = pd.melt(
        sales_raw.drop(columns='id'),
        id_vars=cols_id,
        var_name='d',
        value_name='sales'
    )

    # Merge with calendar and sell_price slightly different between sales & sales_pred
    n_begin = sales.shape[0]
    sales = pd.merge(sales, calendar_state)
    assert n_begin == sales.shape[0]
    sales = pd.merge(sales, sell_prices, how='left')
    assert n_begin == sales.shape[0]

    sales['revenu'] = sales['sell_price'] * sales['sales']

    # If missing sell_price means there were no sales during training
    # But at predict there there no missing sell_price, asymetrie between train/prod
    assert sales[sales['sell_price'].isnull() & (sales['sales'] > 0)].shape[0] == 0

    # If sell_price is null, we assume it simply means that product was not for sale yet
    # sales['sell_price'].isnull().mean()  # ~20%
    sales = sales[sales['sell_price'].notnull()]

    sales_group = aggregate_sales_group(sales, is_train=True)

    sales_group.to_parquet('data/sales.parquet')
    return None


def extract_sales_pred():
    sales_raw = pd.read_csv('data/sales_train_validation.csv')
    sell_prices = pd.read_csv('data/sell_prices.csv')
    calendar_state = read_calendar_state()

    # Tidy format, to be able to merge with sell_prices
    cols_id = ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
    sales_pred = sales_raw[cols_id].drop_duplicates()

    calendar_pred = calendar_state[calendar_state['date'] > '2016-04-24'].copy()

    # Merge with calendar and sell_price slightly different between sales & sales_pred
    n_begin = sales_pred.shape[0]
    sales_pred = pd.merge(
        sales_pred.eval('dummy=1'),
        calendar_pred.eval('dummy=1'),
    ).drop(columns='dummy')
    assert n_begin * 56 == sales_pred.shape[0]
    sales_pred = pd.merge(sales_pred, sell_prices, how='left')
    assert n_begin * 56 == sales_pred.shape[0]
    assert sales_pred['sell_price'].isnull().mean() == 0

    sales_pred = aggregate_sales_group(sales_pred, is_train=False)

    period = calendar_pred[['date']].drop_duplicates().sort_values('date').eval('horizon = 1')
    period['horizon'] = period['horizon'].cumsum()
    period['period'] = np.where(period['horizon'] <= 28, 'validation', 'evaluation')
    period['horizon'] = period['horizon'] - (period.groupby('period')['horizon'].transform('min') - 1)
    checks = period.groupby('period').size()
    assert checks.shape[0] == 2
    assert checks.min() == 28
    assert checks.max() == 28

    sales_pred = pd.merge(sales_pred, period)
    sales_pred.to_parquet('data/sales_pred.parquet')
    return


def str_cols_to_category(sales):
    "Reduce memory size of sales datasets approximatively ~50 times"
    for col in ['total_id', 'group', 'granularity', 'state_id', 'store_id', 'cat_id', 'dept_id', 'item_id']:
        sales[col] = sales[col].astype('category')
    return sales


def read_sales():
    sales = pd.read_parquet('data/sales.parquet')
    sales = str_cols_to_category(sales)
    sales['sales'].astype(pd.Int64Dtype())

    if constants.IS_PROTOTYPING:
        sales = sales[sales['date'] >= '2015-01-01'].copy()

    np.random.seed(1)
    sales['horizon'] = np.random.randint(1, 29, sales.shape[0])
    return sales


def read_sales_pred():
    sales_pred = pd.read_parquet('data/sales_pred.parquet')
    sales_pred = str_cols_to_category(sales_pred)
    return sales_pred


def data_augmentation(sales):
    """
    Data augmentation for all granularities except ones including item_id.
    """
    sales_begin = sales[~sales['granularity'].str.contains('item_id')].copy()
    # sanity check our code is working
    # assert ((sales_begin['horizon']).mod(29).replace(0, 1) == sales_begin['horizon']).all()

    sales_augmented = []
    for i in range(1, constants.AUGMENTATION_SIZE):
        sales_begin['horizon'] = (sales_begin['horizon'] + 1).mod(29).replace(0, 1)
        sales_augmented.append(sales_begin.copy())

    sales_augmented = pd.concat([sales] + sales_augmented)

    return sales_augmented


def data_downsample(sales_augmented):
    """
    Downsample big granularities such as 'state_id_&_item_id', 'item_id_&_store_id'
    Reduces memory consumption to allow more data for other granularities.
    """
    sales_augmented['hash'] = sales_augmented['group'].astype(str).map(
        lambda x: int(hashlib.sha256(x.encode('utf-8')).hexdigest(), 16))
    # sales_augmented['hash'] = sales_augmented['group'].astype(str).apply(hash)
    is_big_granularity = sales_augmented['granularity'].isin(['state_id_&_item_id', 'item_id_&_store_id'])
    is_downsample = sales_augmented['hash'].mod(constants.DOWNSAMPLING_SIZE) == 0
    cond_keep = ~(is_big_granularity & is_downsample)
    sales_augmented = sales_augmented[cond_keep]
    sales_augmented = sales_augmented.drop(columns='hash')
    return sales_augmented


def prep_sales(sales):
    # Should be done before dumping to parquet
    sales['date'] = pd.to_datetime(sales['date'])
    # The date from which the forecast is done
    sales['date_from'] = sales['date'] - pd.to_timedelta(sales['horizon'], unit='d')

    if 'sales' in sales.columns:
        # Set christmas to NaN instead of 0. Because it adds a lot of noise to the FE otherwise
        cond = ((sales['date'].dt.month == 12) & (sales['date'].dt.day == 25))
        sales['sales'] = np.where(cond, np.nan, sales['sales'])
    return sales


def normalize_df(df, normalize_by, is_train, inverse=False):
    """
    Apply normalization or de-normalization

    normalize_by : column normalize by
    """
    cols_sales = [col for col in list(df.columns) if 'sales' in col]
    cols_quantile = list(df.columns[df.columns.str.startswith('quantile_')])
    # column we will normalize back
    for col in cols_sales + cols_quantile:
        if col != normalize_by:
            if inverse is False:
                df[col] = df[col] / df[normalize_by]
            else:
                df[col] = df[col] * df[normalize_by]

    # # We can normalize but keep the same "relative" weight to each series
    if is_train:
        if inverse is False:
            df['weight'] = df['weight'] / df[normalize_by]
        else:
            df['weight'] = df['weight'] * df[normalize_by]

    return df


def preprocessing_learning(df_learning, normalize_by, is_train):
    if normalize_by is not None:
        df_learning = normalize_df(df_learning, normalize_by, is_train)
    return df_learning


def compute_pinball(df):
    # Should probably be in another script
    y = 'sales'
    # Compute pinball loss
    df['pinball'] = 0
    for quantile in constants.LIST_QUANTILE:
        cond = df[y] >= df['quantile_{:.3f}'.format(quantile)]
        delta = (df[y] - df['quantile_{:.3f}'.format(quantile)]).abs()
        pinball_loss = delta * quantile * cond + delta * (1 - quantile) * ~cond
        df['pinball'] += pinball_loss / len(constants.LIST_QUANTILE)

    return df


def postprocessing_learning(df_learning, is_train, normalize_by):
    if normalize_by is not None:
        df_learning = normalize_df(df_learning, normalize_by, is_train, inverse=True)

        if is_train:
            # Compute pinball loss AFTER having denormalized
            df_learning = compute_pinball(df_learning)

    return df_learning


def get_loss_summary(df_valid_predicted):
    """
    Returns a summary of loss by granularity with all data sampling, augmentation cautions.
    """
    loss_summary = df_valid_predicted.groupby('granularity', observed=True).apply(
        ml_utils.wavg, 'pinball', 'weight').rename('pinball').reset_index()
    loss_summary['weight'] = df_valid_predicted.groupby('granularity', observed=True)['weight'].sum().values

    weight_correction = loss_summary[['granularity']]
    weight_correction['weight_correction'] = 1
    # Correct for data augmentation
    weight_correction['weight_correction'] = np.where(
        weight_correction['granularity'].str.contains('item_id'),
        weight_correction['weight_correction'],
        weight_correction['weight_correction'] / constants.AUGMENTATION_SIZE,
    )

    # Correct for data sampling
    weight_correction['weight_correction'] = np.where(
        weight_correction['granularity'].isin(['state_id_&_item_id', 'item_id_&_store_id']),
        weight_correction['weight_correction'] * constants.DOWNSAMPLING_SIZE,
        weight_correction['weight_correction'],
    )

    loss_summary = pd.merge(loss_summary, weight_correction)
    loss_summary['weight'] = loss_summary['weight'] * loss_summary['weight_correction']
    loss_summary = loss_summary.drop(columns='weight_correction')

    loss_summary['pinball_rt'] = (loss_summary['pinball'] * loss_summary['weight']) / \
        ((loss_summary['pinball'] * loss_summary['weight'])).sum()
    loss_summary = loss_summary.sort_values('pinball_rt')
    return loss_summary


def rename_and_dump_df_pred(df_valid_predicted, df_prod_predicted, strategy, normalize_by):
    """
    There are different strategies to train/predict a series. By granularity, by serie, by an higher hierarchy.
    In order to compare each strategy's performance and later use stacking.

    This function performs :
    - rename predictions columns
    - dumps dataframes to parquet files

    """
    str_normalized = '_normed_by_' + normalize_by if normalize_by is not None else ''
    cols_id = ['date', 'group', 'granularity', 'horizon']
    cols_train = ['sales', 'weight', 'pinball']
    cols_prod = ['period']
    cols_pred = ['quantile_{:.3f}'.format(quantile) for quantile in constants.LIST_QUANTILE]
    df_valid_pred = df_valid_predicted[cols_id + cols_train + cols_pred]
    df_prod_pred = df_prod_predicted[cols_id + cols_prod + cols_pred]
    rename_dict = {}
    strategy_normed = strategy + str_normalized  # Ugly :(
    for col in cols_pred + ['pinball']:
        rename_dict[col] = '_'.join([strategy_normed, col])

    df_valid_pred = df_valid_pred.rename(columns=rename_dict)
    df_prod_pred = df_prod_pred.rename(columns=rename_dict)

    df_valid_pred[strategy_normed + '_RETRAIN_FULL_DATA'] = constants.RETRAIN_FULL_DATA
    df_prod_pred[strategy_normed + '_RETRAIN_FULL_DATA'] = constants.RETRAIN_FULL_DATA

    # Dump df_valid_pred & df_prod for later stacking

    df_valid_pred.to_parquet(path.path_df_pred('valid', strategy, normalize_by))
    df_prod_pred.to_parquet(path.path_df_pred('prod', strategy, normalize_by))
    return df_valid_pred, df_prod_pred


def df_pred_to_submission(df_pred):
    """
    Transform a df_pred the correct submission format
    TODO : This func should be in another files
    """
    sample_submission = pd.read_csv('data/sample_submission.csv')

    # if constants.IS_PROTOTYPING:
    #     sales_pred = sales_pred[sales_pred['horizon'] <= 2].copy()

    cols_quantile = list(df_pred.columns[df_pred.columns.str.startswith('quantile_')])
    id_vars = list(df_pred.drop(columns=cols_quantile))
    submission = pd.melt(
        df_pred,
        id_vars=id_vars,
        var_name='quantile',
        value_name='sales_pred'
    )

    submission['F'] = 'F' + submission['horizon'].astype(str)
    submission_wide = submission.pivot_table(
        index=['group', 'quantile', 'period'], columns='F', values='sales_pred').reset_index()
    submission_wide['quantile'] = submission_wide['quantile'].str.split('_').str[1]
    submission_wide['id'] = submission_wide[['group', 'quantile', 'period']].apply(lambda x: '_'.join(x), axis=1)

    cols = ['id'] + ['F' + str(i) for i in range(1, 29)]
    submission_wide = submission_wide[cols]

    # Sanity check
    assert submission_wide.shape[0] == sample_submission.shape[0]

    left = submission_wide['id'].str.split('_').str[:-2].str.join('_').to_frame().drop_duplicates()
    right = sample_submission['id'].str.split('_').str[:-2].str.join('_').to_frame().drop_duplicates()
    check = pd.merge(left, right, how='outer', indicator=True)
    check = check[check['_merge'] != 'both'].copy()
    assert check.shape[0] == 0

    return submission_wide
