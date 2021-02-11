import pandas as pd
import gc

import src.preprocessing as prep
import src.constants as constants
import src.path as path
import src.modeling as modeling


print('IS_PROTOTYPING', constants.IS_PROTOTYPING)
print('RETRAIN_FULL_DATA', constants.RETRAIN_FULL_DATA)


def remove_granularities(df, normalize_by):
    """
    From experimental results (and intuition), normalizing is :
    * good when the problem is treated as time-series one
    * bad when the problem is treated as point one

    In order to speed-up training, remove item's granularities if a normalization is used.

    Could be implemented at strategy level. It is subjective.
    """
    if normalize_by is not None:
        df = df[~df['granularity'].str.contains('item_id')]

    return df


def model_strategy():

    for strategy, normalize_by in constants.l_tuple_strategy_normalised:
        print('\nCurrent strategy is', strategy)
        print('normalize_by', normalize_by)

        # Read df_learning, df_pred
        df_learning = pd.read_parquet(path.path_learning(is_train=True))
        df_prod = pd.read_parquet(path.path_learning(is_train=False))

        df_learning = remove_granularities(df_learning, normalize_by)
        df_prod = remove_granularities(df_prod, normalize_by)

        if normalize_by is not None:
            df_learning = df_learning[~df_learning['granularity'].str.contains('item_id')]

        df_learning = prep.preprocessing_learning(df_learning, normalize_by=normalize_by, is_train=True)
        df_prod = prep.preprocessing_learning(df_prod, normalize_by=normalize_by, is_train=False)

        if strategy == 'granularity':
            df_valid_predicted, df_prod_predicted = modeling.strategy_granularity(
                df_learning, df_prod)

        if strategy == 'serie':
            df_valid_predicted, df_prod_predicted = modeling.strategy_serie(
                df_learning, df_prod)

        if strategy == 'hierarchical':
            df_valid_predicted, df_prod_predicted = modeling.strategy_hierarchical(
                df_learning, df_prod)

        if strategy == 'horizon_serie':
            df_valid_predicted, df_prod_predicted = modeling.strategy_horizon_serie(
                df_learning, df_prod)

        if strategy == 'horizon_granularity':
            df_valid_predicted, df_prod_predicted = modeling.strategy_horizon_granularity(
                df_learning, df_prod)

        if strategy == 'horizon_hierarchical':
            df_valid_predicted, df_prod_predicted = modeling.strategy_horizon_hierarchical(
                df_learning, df_prod)

        if strategy == 'granularity_ngboost':
            df_valid_predicted, df_prod_predicted = modeling.strategy_granularity_ngboost(
                df_learning, df_prod)

        if strategy == 'granularity_tweedie':
            df_valid_predicted, df_prod_predicted = modeling.strategy_granularity_point_to_uncertainity_tweedie(
                df_learning, df_prod)

        if strategy == 'granularity_point_to_uncertainity':
            df_valid_predicted, df_prod_predicted = modeling.strategy_granularity_point_to_uncertainity(
                df_learning, df_prod)

        if strategy == 'serie_tweedie':
            df_valid_predicted, df_prod_predicted = modeling.strategy_serie_tweedie(
                df_learning, df_prod)

        if strategy == 'serie_point_to_uncertainity':
            df_valid_predicted, df_prod_predicted = modeling.strategy_serie_point_to_uncertainity(
                df_learning, df_prod)

        if strategy == 'serie_tensorflow':
            df_valid_predicted, df_prod_predicted = modeling.strategy_serie_tensorflow(
                df_learning, df_prod)

        if strategy == 'granularity_tensorflow':
            df_valid_predicted, df_prod_predicted = modeling.strategy_granularity_tensorflow(
                df_learning, df_prod)

        df_valid_predicted = prep.postprocessing_learning(df_valid_predicted, is_train=True, normalize_by=normalize_by)
        df_prod_predicted = prep.postprocessing_learning(df_prod_predicted, is_train=False, normalize_by=normalize_by)

        loss_summary = prep.get_loss_summary(df_valid_predicted)
        print(loss_summary)

        df_valid_pred, df_prod_pred = prep.rename_and_dump_df_pred(
            df_valid_predicted, df_prod_predicted, strategy=strategy, normalize_by=normalize_by)
        del df_valid_pred, df_prod_pred
        gc.collect()
