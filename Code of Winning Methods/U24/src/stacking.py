import src.preprocessing as prep
import src.ml_utils as ml_utils
import src.constants as constants
import src.utils_stacking as utils_stacking


def stacking():
    print('IS_PROTOTYPING', constants.IS_PROTOTYPING)

    df_valid_pred = utils_stacking.read_each_df_pred(
        set_df='valid', l_tuple_strategy_normalised=constants.l_tuple_strategy_normalised, is_train=True)
    df_prod_pred = utils_stacking.read_each_df_pred(
        set_df='prod', l_tuple_strategy_normalised=constants.l_tuple_strategy_normalised, is_train=False)

    df_valid_pred, df_prod_pred = utils_stacking.stacking_granularity_regression(
        df_learning=df_valid_pred,
        df_prod=df_prod_pred,
        l_tuple_strategy_normalised=constants.l_tuple_strategy_normalised)

    loss_comparison = utils_stacking.get_loss_comparison(
        df_valid_pred, constants.l_tuple_strategy_normalised).round(3)

    loss_summary = prep.get_loss_summary(df_valid_pred)

    loss_local = ml_utils.wavg(loss_summary, 'pinball', 'weight').round(3)
    print('loss_local:', loss_local)

    submission_wide = prep.df_pred_to_submission(df_prod_pred)
    fname = 'output/submission_stacking_{loss_local}{is_prototyping}.csv'.format(
        loss_local=loss_local,
        is_prototyping=constants.str_prototyping,
    )
    submission_wide.to_csv(fname, index=False)
    print('kaggle competitions submit -f {} -m ""'.format(fname))

    fname_loss_summary = 'output/loss_summary_{loss_local}{is_prototyping}.csv'.format(
        loss_local=loss_local,
        is_prototyping=constants.str_prototyping,
    )
    loss_summary.to_csv(fname_loss_summary, index=False)
