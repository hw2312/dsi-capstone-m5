import src.constants as constants


def path_learning(is_train):
    if is_train:
        df_name = 'df_learning'
    else:
        df_name = 'df_pred'

    path = f'data/{df_name}.parquet'
    return path


def path_df_pred(set_df, strategy, normalize_by):
    str_normalized = '_normed_by_' + normalize_by if normalize_by is not None else ''

    path = f'stacking/df_{set_df}_pred_{strategy}{constants.str_prototyping}{str_normalized}.parquet'
    return path
