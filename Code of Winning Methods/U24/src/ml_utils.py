import pandas as pd
import numpy as np
from multiprocessing import Pool
from functools import partial
import math


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def parallelize_function_dataframe(dataframe, function, partition_by=None, chunk_num=16, **kwargs):
    """ 
    Apply a function to a dataframe in a parallelized way. 
    Best practice : It takes time to copy chunk of dataframe to each core. So a best practice is 
    to feed this function with only the columns needed. It also reduces memory peak.
    """
    if partition_by is None:
        df_split = np.array_split(dataframe, chunk_num)
    else:
        # Not very effective to split then concat back
        # TODO improve this
        chunk = dataframe[partition_by].drop_duplicates()
        chunk['no_other_variable_name'] = 1
        chunk['no_other_variable_name'] = chunk['no_other_variable_name'].cumsum().mod(chunk_num)
        dataframe = pd.merge(dataframe, chunk, how='left')
        df_split = [dataframe[dataframe['no_other_variable_name'] == n]
                    for n in chunk['no_other_variable_name'].unique()]

    pool = Pool(chunk_num)
    dataframe = pd.concat(pool.map(partial(function, **kwargs), df_split))
    pool.close()
    pool.join()

    dataframe = dataframe.drop(columns='no_other_variable_name', errors='ignore')

    return dataframe


def get_importance_lgb(model_gbm, X_cols=None):
    importance = pd.DataFrame()
    if X_cols is None:
        importance["feature"] = model_gbm.feature_name()
    else:
        importance["feature"] = X_cols
    importance["importance"] = model_gbm.feature_importance(importance_type='gain')
    importance['importance'] = importance['importance'] / importance['importance'].replace(np.inf, 0).sum()
    importance['importance'] = importance['importance'] * 100
    importance['importance_rank'] = importance['importance'].rank(ascending=False)  # .astype(int)
    importance = importance.sort_values('importance_rank').round(2)
    return importance


def plot_importance_lgb(importance):
    # Ugly but pip install problem on Airflow otherwise
    import plotnine as pn
    from plotnine import ggplot, aes  # noqa
    # from plotnine.geoms import *  # noqa
    coef = 1.5
    pn.options.figure_size = (6.4*coef, 4.8*coef)
    from mizani.formatters import percent_format  # noqa
    # from mizani.breaks import date_breaks  # noqa
    # from mizani.formatters import date_format  # noqa

    importance['importance'] = importance['importance'] / 100
    importance['feature'] = pd.Categorical(
        importance['feature'],
        importance['feature'][::-1], ordered=True)
    plot = (ggplot(importance, aes('feature', 'importance')) +
            pn.geom_bar(stat='identity') +
            pn.coords.coord_flip() +
            pn.scales.scale_y_continuous(labels=percent_format()) +
            pn.labs(title='Feature importance', x='Feature', y='Gain'))
    return plot


def oof_dataframe(data, col_fold, function, **kwargs):
    folders = data[col_fold].unique()
    res = pd.DataFrame()
    for f in folders:
        res_tmp = function(data[data[col_fold] != f].copy(), **kwargs)
        res_tmp[col_fold] = f
        res = res.append(res_tmp)

    return res


def wavg(group, avg_name, weight_name):
    """ http://stackoverflow.com/questions/10951341/pandas-dataframe-aggregate-function-using-multiple-columns
    In rare instance, we may not have weights, so just return the mean. Customize this if your business case
    should return otherwise.
    """
    d = group[avg_name]
    w = group[weight_name]
    return (d * w).sum() / w.sum()


def adversarial_validation(X_train, X_test):
    """
    Train a lightgbm model to try identifying differences (leak) betwen a
    training and test set
    """
    from sklearn.model_selection import train_test_split
    import lightgbm as lgb
    import numpy as np
    import scipy
    from scipy.sparse import vstack
    if isinstance(X_train, pd.DataFrame):
        X_full = X_train.append(X_test)
    if isinstance(X_train, scipy.sparse.csr.csr_matrix):
        X_full = vstack([X_train, X_test], format='csr')

    y_full = np.concatenate((np.ones(X_train.shape[0]), np.zeros(X_test.shape[0])))

    X_train, X_valid, y_train, y_valid = train_test_split(X_full, y_full)

    param = {'objective': 'binary',
             "metric": 'auc',
             'learning_rate': 0.01}

    lgb_train = lgb.Dataset(X_train, label=y_train)
    lgb_valid = lgb.Dataset(X_valid, label=y_valid)

    model_gbm = lgb.train(param, lgb_train, 200,
                          valid_sets=[lgb_train, lgb_valid],
                          early_stopping_rounds=10,
                          verbose_eval=10)

    return model_gbm


def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df
