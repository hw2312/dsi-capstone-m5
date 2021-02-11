import numpy as np
import gc

import src.preprocessing as prep
import src.feature_engineering as fe
import src.ml_utils as ml_utils
import src.constants as constants
import src.path as path


def create_learning():
    print('IS_PROTOTYPING', constants.IS_PROTOTYPING)
    sales = prep.read_sales()
    sales_pred = prep.read_sales_pred()

    sales_augmented = prep.data_augmentation(sales)
    sales_augmented = prep.data_downsample(sales_augmented)

    sales = prep.prep_sales(sales)
    sales_augmented = prep.prep_sales(sales_augmented)
    sales_pred = prep.prep_sales(sales_pred)

    sales = fe.fe_sales(sales)
    sales_augmented = fe.fe_sales(sales_augmented)
    sales_pred = fe.fe_sales(sales_pred)

    df_groups = fe.get_df_groups(sales)

    df_loss_fe = fe.get_df_loss_fe(sales)

    first_sale_date = fe.fe_first_sale_date(sales)

    weather_state, weather_total = fe.get_fe_weather()

    print('get_fe_sales_historical')
    fe_sales_historical = ml_utils.parallelize_function_dataframe(
        dataframe=sales[['date', 'group', 'granularity', 'sales', 'sell_price', 'active_item']],
        function=fe.get_fe_sales_historical,
        partition_by=['group'],
        chunk_num=16*3)

    print('get_fe_sales_historical_dow')
    fe_sales_historical_dow = ml_utils.parallelize_function_dataframe(
        dataframe=sales[['date', 'group', 'sales', 'date_dayofweek']],
        function=fe.get_fe_sales_historical_dow,
        partition_by=['group'],
        chunk_num=16*3)

    print('get_fe_sales_historical_last_date')
    fe_sales_historical_last_date = fe.get_fe_sales_historical_last_date(sales)

    print('yearly_trend')
    print('yearly_trend is probably to be deprecated, expected to be less robust than yearly_trend_day')

    dayofyear_trend = fe.fe_dayofyear_trend(fe_sales_historical)

    print('yearly_trend_day')
    yearly_trend_day = fe.fe_yearly_trend_day(fe_sales_historical, sales)

    print('monthly_trend_day')
    monthly_trend_day = fe.fe_monthly_trend_day(yearly_trend_day)

    print('yearly_trend_hierarchical')
    dayofyear_trend_hierarchical = fe.fe_dayofyear_trend_hierarchical(dayofyear_trend, df_groups)

    print('yearly_trend_day_hierarchical')
    yearly_trend_day_hierarchical = fe.fe_yearly_trend_day_hierarchical(yearly_trend_day, df_groups)

    print('monthly_trend_day_hierarchical')
    monthly_trend_day_hierarchical = fe.fe_monthly_trend_day_hierarchical(monthly_trend_day, df_groups)

    # Dict for left join
    dict_df_fe_merge = {
        'df_loss_fe': df_loss_fe,
        'first_sale_date': first_sale_date,
        'weather_state': weather_state,
        'weather_total': weather_total,
        'fe_historical_sales': fe_sales_historical,
    }

    # Add fe_sales_historical_hierarchical which is already a dict
    # dict_df_fe_merge.update(fe_sales_historical_hierarchical)
    dict_df_fe_merge.update(dayofyear_trend_hierarchical)
    dict_df_fe_merge.update(yearly_trend_day_hierarchical)
    dict_df_fe_merge.update(monthly_trend_day_hierarchical)
    del dayofyear_trend, yearly_trend_day, monthly_trend_day
    del dayofyear_trend_hierarchical, yearly_trend_day_hierarchical, monthly_trend_day_hierarchical

    # Dict for rolling join
    dict_df_fe_merge_asof = {
        'fe_sales_historical_last_date': fe_sales_historical_last_date,
        'fe_sales_historical_dow': fe_sales_historical_dow,
    }

    # Pray here
    gc.collect()
    df_learning = fe.get_learning(sales_augmented, dict_df_fe_merge, dict_df_fe_merge_asof)
    gc.collect()
    df_prod = fe.get_learning(sales_pred, dict_df_fe_merge, dict_df_fe_merge_asof)
    gc.collect()

    # Reduce memory usage :(
    del sales_augmented, sales_pred, dict_df_fe_merge, dict_df_fe_merge_asof
    del fe_sales_historical_last_date, fe_sales_historical_dow
    gc.collect()

    # If a product had no sales during >= 28 days, its weight is 0, remove theses lines.
    # Should be done after fe_learning but reduce memory spike
    # Also removes serie with less than 28 days
    is_positive_weight = (df_learning['weight'] > 0)
    df_learning = df_learning[is_positive_weight].copy()

    print('fe_learning')
    # Dangerous memory peaks here, but i don't understand why
    df_learning = fe.fe_learning(df_learning)
    df_prod = fe.fe_learning(df_prod)

    # Remove Christmas
    df_learning = df_learning[df_learning['sales'].notnull()]

    print('event_trend_group')
    event_trend_group = fe.fe_event_trend_group(df_learning)
    event_trend_group.sort_values('event_trend')
    event_trend_hierarchical = fe.get_event_trend_hierarchical(event_trend_group, df_groups)
    del event_trend_group

    print('merge_event_trend_hierarchical')
    df_learning = fe.merge_event_trend_hierarchical(df_learning, event_trend_hierarchical)
    df_prod = fe.merge_event_trend_hierarchical(df_prod, event_trend_hierarchical)

    # assert df_learning['total_id_event_trend'].notnull().all()

    assert df_learning['weight'].isnull().sum() == 0
    assert df_learning['weight'].max() != np.inf
    assert df_learning['weight'].min() > 0

    if constants.IS_PROTOTYPING is False:
        df_learning.to_parquet(path.path_learning(is_train=True))
        df_prod.to_parquet(path.path_learning(is_train=False))
