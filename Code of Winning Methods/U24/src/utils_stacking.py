import pandas as pd
import numpy as np
from statsmodels.regression.quantile_regression import QuantReg

import src.constants as constants
import src.path as path
import src.preprocessing as prep
import src.ml_utils as ml_utils

from functools import reduce

# from statsmodels.regression.quantile_regression import QuantReg


def read_each_df_pred(set_df, l_tuple_strategy_normalised, is_train):
    # Algorithms to stack
    df_list = [pd.read_parquet(path.path_df_pred(set_df, strategy, normalize_by))
               for strategy, normalize_by in l_tuple_strategy_normalised]
    # df_pred_stacking = reduce(lambda x, y: pd.merge(x, y.drop(columns=['weight', 'sales']), how='outer'), df_list)

    # weight and sales are equals on each dataset but merge fails. Don't understand why.
    # Code clumsy here because of this.
    df_pred_stacking = None
    for df in df_list:
        if is_train:
            df = df.drop(columns=['weight', 'sales'])
        if df_pred_stacking is None:
            df_pred_stacking = df
        else:
            df_pred_stacking = pd.merge(df_pred_stacking, df, how='outer')

    if is_train:
        keys_cols = pd.read_parquet(path.path_learning(is_train=True), columns=[
                                    'date', 'horizon', 'group', 'granularity', 'weight', 'sales'])
        keys_cols = keys_cols[keys_cols['date'] >= '2016-02-15']
        keys_cols = keys_cols.drop_duplicates()
        nrow_begin = df_pred_stacking.shape[0]
        df_pred_stacking = pd.merge(keys_cols, df_pred_stacking)
        assert nrow_begin == df_pred_stacking.shape[0]

    return df_pred_stacking


def train_predict_stacking_dummy_argmin(df_learning, df_prod, l_tuple_strategy_normalised):
    """
    Dummy algo which takes the argmin strategy as predictor.

    Ugly, refactor later
    """
    cols_pinball = []
    for strategy, normalize_by in l_tuple_strategy_normalised:
        str_normalized = '_normed_by_' + normalize_by if normalize_by is not None else ''
        cols_pinball.append(f'{strategy}{str_normalized}_pinball')

    loss_comparison = get_loss_comparison(df_learning, l_tuple_strategy_normalised).round(2)
    coeff = loss_comparison[cols_pinball].rank(method='first', axis=1) == 1

    print('Best strategy is:', coeff.idxmax(axis=1).values[0])

    for strategy, normalize_by in l_tuple_strategy_normalised:
        str_normalized = '_normed_by_' + normalize_by if normalize_by is not None else ''
        col = f'{strategy}{str_normalized}_pinball'
        df_learning[f'{strategy}{str_normalized}_coef'] = coeff[col].values[0]
        df_prod[f'{strategy}{str_normalized}_coef'] = coeff[col].values[0]

    return df_learning, df_prod


def get_loss_comparison(df_valid_pred, l_tuple_strategy_normalised):
    grouped = df_valid_pred.groupby('granularity', observed=True)
    loss_comparison = []
    for strategy, normalize_by in l_tuple_strategy_normalised:
        str_normalized = '_normed_by_' + normalize_by if normalize_by is not None else ''
        col_loss = f'{strategy}{str_normalized}_pinball'
        loss_comparison.append(grouped.apply(ml_utils.wavg, col_loss, 'weight').rename(col_loss).reset_index())

    loss_comparison = reduce(lambda x, y: pd.merge(x, y, how='outer'), loss_comparison)
    loss_comparison = loss_comparison.replace(0, np.nan)
    return loss_comparison


def train_predict_stacking_linear_regression(df_learning, df_prod, l_tuple_strategy_normalised):
    for quantile in constants.LIST_QUANTILE:
        to_keep = []
        for strategy, normalize_by in l_tuple_strategy_normalised:
            str_normalized = '_normed_by_' + normalize_by if normalize_by is not None else ''
            to_keep.append('{}{}_quantile_{:.3f}'.format(strategy, str_normalized, quantile))

        # Remove NA columns
        to_keep = df_learning[to_keep].notnull().all()
        to_keep = to_keep[to_keep].index.tolist()

        # We need to remove constants columns from the sampled data
        df_learning_weighted = df_learning.sample(10000, weights='weight', replace=True, random_state=1)

        # Remove constants columns
        cols_constants = df_learning_weighted[to_keep].std() == 0
        cols_constants = cols_constants[cols_constants].index.tolist()
        for col in cols_constants:
            to_keep.remove(col)

        # # Remove correlated features
        # # Create correlation matrix
        # corr_matrix = df_learning[to_keep].corr().abs().fillna(1)

        # # Select upper triangle of correlation matrix
        # upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

        # # Find index of feature columns with correlation greater than 0.95
        # to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
        # to_keep.remove(to_drop)

        # Drop duplicates columns
        def getDuplicateColumns(df):
            '''
            Get a list of duplicate columns.
            It will iterate over all the columns in dataframe and find the columns whose contents are duplicate.
            :param df: Dataframe object
            :return: List of columns whose contents are duplicates.
            '''
            duplicateColumnNames = set()
            # Iterate over all the columns in dataframe
            for x in range(df.shape[1]):
                # Select column at xth index.
                col = df.iloc[:, x]
                # Iterate over all the columns in DataFrame from (x+1)th index till end
                for y in range(x + 1, df.shape[1]):
                    # Select column at yth index.
                    otherCol = df.iloc[:, y]
                    # Check if two columns at x 7 y index are equal
                    if col.equals(otherCol):
                        duplicateColumnNames.add(df.columns.values[y])

            return list(duplicateColumnNames)

        cols_duplicate = getDuplicateColumns(df_learning_weighted[to_keep])
        for cols in cols_duplicate:
            to_keep.remove(cols)

        # to_keep = df_learning_weighted[to_keep].T.drop_duplicates().T.columns  # Not efficient but ok

        X_learning_weighted = df_learning_weighted[to_keep].fillna(0)
        X_learning = df_learning[to_keep].fillna(0)
        X_prod = df_prod[to_keep].fillna(0)

        y_learning_weighted = df_learning_weighted['sales']
        # weight_learning = df_learning['weight']
        if X_learning_weighted.nunique().max() != 1:
            linear_model = QuantReg(y_learning_weighted, X_learning_weighted)
            linear_model = linear_model.fit(q=quantile)
            # print(linear_model.summary())
            df_learning['quantile_{:.3f}'.format(quantile)] = linear_model.predict(X_learning)
            df_prod['quantile_{:.3f}'.format(quantile)] = linear_model.predict(X_prod)
        else:
            df_learning['quantile_{:.3f}'.format(quantile)] = 0
            df_prod['quantile_{:.3f}'.format(quantile)] = 0

    return df_learning, df_prod


def stacking_granularity_regression(df_learning, df_prod, l_tuple_strategy_normalised):
    """
    Apply stacking of learning one model for each granularity
    """
    df_valid_pred = []
    df_prod_pred = []
    for granularity in df_learning['granularity'].unique():
        print(granularity)
        df_learning_tmp = df_learning[(df_learning['granularity'] == granularity)]
        df_prod_tmp = df_prod[(df_prod['granularity'] == granularity)]
        df_valid_predicted_tmp, df_prod_predicted_tmp = (
            train_predict_stacking_linear_regression(
                df_learning=df_learning_tmp,
                df_prod=df_prod_tmp,
                l_tuple_strategy_normalised=l_tuple_strategy_normalised,
            )
        )
        df_valid_pred.append(df_valid_predicted_tmp)
        df_prod_pred.append(df_prod_predicted_tmp)

    df_valid_pred = pd.concat(df_valid_pred)
    df_prod_pred = pd.concat(df_prod_pred)
    df_valid_pred = prep.compute_pinball(df_valid_pred)

    return df_valid_pred, df_prod_pred
