import src.constants as constants
import src.preprocessing as prep
import src.ml_utils as ml_utils

import lightgbm as lgb
import pandas as pd
import numpy as np
from statsmodels.regression.quantile_regression import QuantReg

import scipy.stats
from ngboost import NGBRegressor
from ngboost.distns import Normal


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
    'weight',
    'date_dayofyear',
    'date_year',
    # 'event_name',
    'sell_price',
]


def prepare_data(df_learning, df_prod):
    """
    Prepare different df_datasets, X_datasets, y_datasets, weight_datasets
    """
    df_train = df_learning[df_learning['date'] < '2016-02-15']
    df_valid = df_learning[df_learning['date'] >= '2016-02-15']
    # df_valid for a clean performance evaluation in order to perform stacking
    # df_valid_oof will be used for determine early stopping

    # Split must be deterministic and should not depend size of df_valid, otherwise different split for each strategy
    cond_split = (df_valid['date'].dt.day + df_valid['horizon']).mod(2) == 0
    df_valid_oof = df_valid[cond_split]
    df_valid = df_valid[~cond_split]

    weight = 'weight'
    y = 'sales'
    X_cols = list(df_learning.head().drop(columns=to_drop + [y]))

    X_learning = df_learning[X_cols]
    X_train = df_train[X_cols]
    X_valid = df_valid[X_cols]
    X_valid_oof = df_valid_oof[X_cols]
    X_prod = df_prod[X_cols]

    y_learning = df_learning[y]
    y_train = df_train[y]
    y_valid_oof = df_valid_oof[y]
    y_valid = df_valid[y]

    weight_learning = df_learning[weight]
    weight_train = df_train[weight]
    weight_valid = df_valid[weight]
    weight_valid_oof = df_valid_oof[weight]

    lgb_learning = lgb.Dataset(
        X_learning,
        label=y_learning,
        weight=weight_learning,
        free_raw_data=False,
    )
    lgb_train = lgb.Dataset(
        X_train,
        label=y_train,
        weight=weight_train,
        free_raw_data=False,
    )
    lgb_valid = lgb.Dataset(
        X_valid,
        label=y_valid,
        weight=weight_valid,
        free_raw_data=False,
    )

    tuple_res = (
        df_learning, df_train, df_valid, df_valid_oof,
        X_learning, X_train, X_valid, X_valid_oof, X_prod,
        y_learning, y_train, y_valid, y_valid_oof,
        weight_learning, weight_train, weight_valid, weight_valid_oof,
        lgb_learning, lgb_train, lgb_valid,
    )

    return tuple_res


def get_lgb_params(objective, dataset_nrows):
    """
    Returns set of parameters for LightGBM

    Totally subjective. Default parameters simply learning_rate and bagging_fraction changed
    for speed tradeoff :(
    """

    if objective == 'quantile':
        learning_rate = 0.2
    elif objective in ['regression', 'tweedie']:
        learning_rate = 0.05
    else:
        raise ValueError('objective learning rate not defined')

    param = {
        'objective': objective,
        'learning_rate': learning_rate,
        'max_cat_to_onehot': 15,
        'bagging_freq': 1,
        'min_data_in_leaf': 100,
        'min_sum_hessian_in_leaf': 0.0,
        'verbose': -1,
        'num_threads': 16,
    }
    num_boost_round = 5000
    # We need a rather high early stopping, because train != valid set (different season) therefore we face local minima
    early_stopping_rounds = 500
    param['bagging_freq'] = 1
    param['bagging_fraction'] = 1

    if objective == 'quantile':
        print('dataset_nrows :', dataset_nrows)
        if dataset_nrows > 100_000:
            # Increase learning_rate
            param['learning_rate'] = 0.2
            param['bagging_fraction'] = 0.7
            num_boost_round = 500
            early_stopping_rounds = 50

        if dataset_nrows > 500_000:
            param['bagging_fraction'] = 0.5

        if dataset_nrows > 1_000_000:
            param['learning_rate'] = 0.2
            param['bagging_fraction'] = 0.1

    if constants.IS_PROTOTYPING:
        num_boost_round = 10
        param['bagging_fraction'] = param['bagging_fraction'] / 4

    return param, num_boost_round, early_stopping_rounds


def train_predict_lgb(df_learning, df_valid, X_learning, X_valid, df_valid_oof, df_prod, X_valid_oof, X_prod, lgb_train, lgb_valid, lgb_learning, param, num_boost_round, early_stopping_rounds, verbose_eval, col_predict):
    model_lgb = lgb.train(
        params=param,
        train_set=lgb_train,
        num_boost_round=num_boost_round,
        valid_sets=[lgb_train, lgb_valid],
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=verbose_eval,
    )

    if verbose_eval > 0:
        importance = ml_utils.get_importance_lgb(model_lgb)
        print(importance.head(8))

    df_valid_oof[col_predict] = model_lgb.predict(X_valid_oof)
    df_prod[col_predict] = model_lgb.predict(X_prod)
    df_learning[col_predict] = model_lgb.predict(X_learning)
    df_valid[col_predict] = model_lgb.predict(X_valid)
    df_learning_pred = df_learning
    df_valid_pred = df_valid

    if constants.RETRAIN_FULL_DATA:
        # Retrain on full data, particularly usefull when datasize is small or changing trend (eg fit by serie)
        model_lgb = lgb.train(
            params=param,
            train_set=lgb_learning,
            num_boost_round=model_lgb.best_iteration,
        )
        df_prod[col_predict] = model_lgb.predict(X_prod)

    return df_learning_pred, df_valid_pred, df_valid_oof, df_prod


def train_predict_lgb_point_to_uncertainity(df_learning, df_prod, verbose_eval):
    """
    Args :
    - df_learning
    - df_prod

    Returns:
    - df_valid with quantile prediction and pinball loss
    - df_prod with quantile prediction
    """
    (
        df_learning, df_train, df_valid, df_valid_oof,
        X_learning, X_train, X_valid, X_valid_oof, X_prod,
        y_learning, y_train, y_valid, y_valid_oof,
        weight_learning, weight_train, weight_valid, weight_valid_oof,
        lgb_learning, lgb_train, lgb_valid,
    ) = prepare_data(df_learning, df_prod)

    param, num_boost_round, early_stopping_rounds = get_lgb_params(
        objective='regression', dataset_nrows=df_learning.shape[0])
    col_predict = 'pred'

    df_learning_pred, df_valid_pred, df_valid_oof, df_prod = train_predict_lgb(
        df_learning, df_valid,
        X_learning, X_valid,
        df_valid_oof, df_prod,
        X_valid_oof, X_prod, lgb_train, lgb_valid, lgb_learning,
        param, num_boost_round, early_stopping_rounds, verbose_eval, col_predict)

    df_learning_weighted = pd.concat([df_valid_oof, df_valid_pred]).sample(
        100000, weights='weight', replace=True, random_state=1)
    # If we fit QuantReg on overfitted prediction, QuantReg underestimate the security  needed
    # df_learning_weighted = df_learning.sample(100000, weights='weight', replace=True, random_state=1)

    to_keep = ['pred', 'horizon']
    X_learning_weighted = df_learning_weighted[to_keep]
    X_learning = df_learning[to_keep]
    X_valid_oof = df_valid_oof[to_keep]
    X_prod = df_prod[to_keep]
    # y_learning = df_learning['sales']
    y_learning_weighted = df_learning_weighted['sales']

    for quantile in constants.LIST_QUANTILE:
        # QuantReg do not have weight parameter, so we mannualy reweight our datasets
        linear_model = QuantReg(y_learning_weighted, X_learning_weighted)
        linear_model = linear_model.fit(q=quantile)
        # print(linear_model.summary())
        df_learning['quantile_{:.3f}'.format(
            quantile)] = linear_model.predict(X_learning)
        df_valid_oof['quantile_{:.3f}'.format(
            quantile)] = linear_model.predict(X_valid_oof)
        df_prod['quantile_{:.3f}'.format(
            quantile)] = linear_model.predict(X_prod)

    df_valid_oof = prep.compute_pinball(df_valid_oof)

    return df_valid_oof, df_prod


def train_predict_lgb_tweedie(df_learning, df_prod, verbose_eval=75):
    """
    Args :
    - df_learning
    - df_prod

    Returns:
    - df_valid with quantile prediction and pinball loss
    - df_prod with quantile prediction
    """
    (
        df_learning, df_train, df_valid, df_valid_oof,
        X_learning, X_train, X_valid, X_valid_oof, X_prod,
        y_learning, y_train, y_valid, y_valid_oof,
        weight_learning, weight_train, weight_valid, weight_valid_oof,
        lgb_learning, lgb_train, lgb_valid,
    ) = prepare_data(df_learning, df_prod)

    param, num_boost_round, early_stopping_rounds = get_lgb_params(
        objective='tweedie', dataset_nrows=df_learning.shape[0])
    col_predict = 'pred'

    df_learning_pred, df_valid_pred, df_valid_oof, df_prod = train_predict_lgb(
        df_learning, df_valid,
        X_learning, X_valid,
        df_valid_oof, df_prod,
        X_valid_oof, X_prod, lgb_train, lgb_valid, lgb_learning,
        param, num_boost_round, early_stopping_rounds, verbose_eval, col_predict)

    from statsmodels.regression.quantile_regression import QuantReg

    df_learning_weighted = df_learning.sample(
        100000, weights='weight', replace=True)

    to_keep = ['pred', 'horizon']
    X_learning_weighted = df_learning_weighted[to_keep]
    X_learning = df_learning[to_keep]
    X_valid_oof = df_valid_oof[to_keep]
    X_prod = df_prod[to_keep]
    # y_learning = df_learning['sales']
    y_learning_weighted = df_learning_weighted['sales']

    for quantile in constants.LIST_QUANTILE:
        # QuantReg do not have weight parameter, so we mannualy reweight our datasets
        linear_model = QuantReg(y_learning_weighted, X_learning_weighted)
        linear_model = linear_model.fit(q=quantile)
        # print(linear_model.summary())
        df_learning['quantile_{:.3f}'.format(
            quantile)] = linear_model.predict(X_learning)
        df_valid_oof['quantile_{:.3f}'.format(
            quantile)] = linear_model.predict(X_valid_oof)
        df_prod['quantile_{:.3f}'.format(
            quantile)] = linear_model.predict(X_prod)

    df_valid_oof = prep.compute_pinball(df_valid_oof)

    return df_valid_oof, df_prod


def train_predict_lgb_quantile(df_learning, df_prod, verbose_eval=75):
    """
    Args :
    - df_learning
    - df_prod

    Returns:
    - df_valid with quantile prediction and pinball loss
    - df_prod with quantile prediction
    """
    (
        df_learning, df_train, df_valid, df_valid_oof,
        X_learning, X_train, X_valid, X_valid_oof, X_prod,
        y_learning, y_train, y_valid, y_valid_oof,
        weight_learning, weight_train, weight_valid, weight_valid_oof,
        lgb_learning, lgb_train, lgb_valid,
    ) = prepare_data(df_learning, df_prod)

    param, num_boost_round, early_stopping_rounds = get_lgb_params(
        objective='quantile', dataset_nrows=df_learning.shape[0])

    for quantile in constants.LIST_QUANTILE:
        print('quantile:', quantile)
        param["alpha"] = quantile
        col_predict = 'quantile_{:.3f}'.format(quantile)
        df_learning_pred, df_valid_pred, df_valid_oof, df_prod = train_predict_lgb(
            df_learning, df_valid,
            X_learning, X_valid,
            df_valid_oof, df_prod,
            X_valid_oof, X_prod, lgb_train, lgb_valid, lgb_learning,
            param, num_boost_round, early_stopping_rounds, verbose_eval, col_predict)

    df_valid_oof = prep.compute_pinball(df_valid_oof)

    return df_valid_oof, df_prod


def train_predict_ngboost(df_learning, df_prod, verbose_eval=75):
    """
    NOT USED - poor performance

    Args :
    - df_learning
    - df_prod

    Returns:
    - df_valid with quantile prediction and pinball loss
    - df_prod with quantile prediction
    """
    (
        df_learning, df_train, df_valid, df_valid_oof,
        X_learning, X_train, X_valid, X_valid_oof, X_prod,
        y_learning, y_train, y_valid, y_valid_oof,
        weight_learning, weight_train, weight_valid, weight_valid_oof,
        lgb_learning, lgb_train, lgb_valid,
    ) = prepare_data(df_learning, df_prod)

    print('df_learning.shape[0]', df_learning.shape[0])

    ngb = NGBRegressor(Dist=Normal, verbose=True, learning_rate=0.1)
    ngb = ngb.fit(
        X=X_train.replace([np.nan, np.inf, -np.inf], 0).values,
        Y=y_train.values,
        sample_weight=weight_train.values,
        X_val=X_valid.replace([np.nan, np.inf, -np.inf], 0).values,
        Y_val=y_valid,
        val_sample_weight=weight_valid.values,
        early_stopping_rounds=20,
    )

    def ngboost_predict(df, X):
        y_pred = ngb.pred_dist(X.replace([np.nan, np.inf, -np.inf], 0).values).params
        y_pred = pd.DataFrame(y_pred)

        y_pred = scipy.stats.norm(y_pred['loc'], y_pred['scale'])

        for quantile in constants.LIST_QUANTILE:
            df['quantile_{:.3f}'.format(quantile)] = y_pred.ppf(quantile)

        return df

    df_valid_oof = ngboost_predict(df_valid_oof, X_valid_oof)
    df_prod = ngboost_predict(df_prod, X_prod)

    # if RETRAIN_FULL_DATA:
    #     print('TODO')

    df_valid_oof = prep.compute_pinball(df_valid_oof)

    return df_valid_oof, df_prod


def train_predict_tensorflow_quantile(df_learning, df_prod, verbose_eval=75):
    import tensorflow as tf
    import tensorflow.keras.layers as L
    import tensorflow.keras.models as M
    import tensorflow.keras.backend as K

    (
        df_learning, df_train, df_valid, df_valid_oof,
        X_learning, X_train, X_valid, X_valid_oof, X_prod,
        y_learning, y_train, y_valid, y_valid_oof,
        weight_learning, weight_train, weight_valid, weight_valid_oof,
        lgb_learning, lgb_train, lgb_valid,
    ) = prepare_data(df_learning, df_prod)

    if df_learning.shape[0] > 10_000_000:
        # assert (df_learning.sample(5, random_state=1).index == y_learning.sample(5, random_state=1).index).all()
        SAMPLING_SIZE = 4_000_000

        df_learning = df_learning.sample(SAMPLING_SIZE, replace=True, random_state=1)
        X_learning = X_learning.sample(SAMPLING_SIZE, replace=True, random_state=1)
        weight_learning = weight_learning.sample(SAMPLING_SIZE, replace=True, random_state=1)
        y_learning = y_learning.sample(SAMPLING_SIZE, replace=True, random_state=1)

        df_train = df_train.sample(SAMPLING_SIZE, replace=True, random_state=1)
        X_train = X_train.sample(SAMPLING_SIZE, replace=True, random_state=1)
        weight_train = weight_train.sample(SAMPLING_SIZE, replace=True, random_state=1)
        y_train = y_train.sample(SAMPLING_SIZE, replace=True, random_state=1)

    def make_X_tf(X, cols_numericals, cols_categoricals, scaler=None, dict_label_encoder=None):
        from sklearn.preprocessing import StandardScaler
        from sklearn.preprocessing import LabelEncoder

        X = X.replace([np.nan, np.inf, -np.inf], 0)

        # print('TODO securize this')
        # X[cols_categoricals] = X[cols_categoricals] - X[cols_categoricals].min()

        if scaler is None:
            scaler = StandardScaler()
            scaler.fit(X[cols_numericals])
            dict_label_encoder = {}
            for col in cols_categoricals:
                label_encoder = LabelEncoder()
                label_encoder.fit(X[col])
                dict_label_encoder[col] = label_encoder

        X[cols_numericals] = scaler.transform(X[cols_numericals])

        for col in cols_categoricals:
            label_encoder = dict_label_encoder[col]
            X[col] = label_encoder.transform(X[col])

        X_tf = {}
        X_tf["num"] = X[cols_numericals].values
        for cat in cols_categoricals:
            X_tf[cat] = X[cat].values

        return X_tf, scaler, dict_label_encoder

    cols_categoricals = [
        'horizon',
        'date_dayofweek',
        'date_day',
        'date_month_week',
        'date_month',
        'date_from_day',
        'date_from_month',
    ]

    cols_numericals = list(X_train.head().drop(columns=cols_categoricals))

    # Embedding rules. Max dim is 50, otherwise half of the uniques.
    categoricals_info = {}
    for c in cols_categoricals:
        total_unique = X_train[c].nunique()
        categoricals_info[c] = (total_unique, min(50, (total_unique + 1) // 2))

    def qloss(y_true, y_pred):
        # Pinball loss for multiple quantiles
        qs = [0.005, 0.025, 0.165, 0.250, 0.500, 0.750, 0.835, 0.975, 0.995]
        q = tf.constant(np.array([qs]), dtype=tf.float32)
        e = y_true - y_pred
        v = tf.maximum(q*e, (q-1)*e)
        return K.mean(v)

    def make_model(numeric_input, categoricals_info):
        layer_numeric = L.Input((numeric_input,), name="layer_numeric")
        inputs = {"layer_numeric": layer_numeric}
        list_layer_categorical = []
        for col, value in categoricals_info.items():
            tmp_input = L.Input((1,), name=col)
            inputs[col] = tmp_input
            list_layer_categorical.append(L.Embedding(value[0], value[1], name="%s_3d" % col)(tmp_input))

        emb = L.Concatenate(name="embds")(list_layer_categorical)
        context = L.Flatten(name="context")(emb)

        x = L.Concatenate(name="x1")([context, layer_numeric])
        x = L.Dense(64*9, activation="relu", name="d1")(x)
        x = L.Dense(64*9, activation="relu", name="d2")(x)
        x = L.Dense(32*9, activation="relu", name="d3")(x)

        preds = L.Dense(9, activation="linear", name="preds")(x)

        model = M.Model(inputs, preds, name="M1")
        model.compile(loss=qloss, optimizer="adam")
        return model

    X_train_tf, scaler, dict_label_encoder = make_X_tf(X_train, cols_numericals, cols_categoricals)

    X_learning_tf, scaler, dict_label_encoder = make_X_tf(
        X_learning, cols_numericals, cols_categoricals, scaler, dict_label_encoder)

    X_valid_tf, scaler, dict_label_encoder = make_X_tf(
        X_valid, cols_numericals, cols_categoricals, scaler, dict_label_encoder)

    X_valid_oof_tf, scaler, dict_label_encoder = make_X_tf(
        X_valid_oof, cols_numericals, cols_categoricals, scaler, dict_label_encoder)
    X_prod_tf, scaler, dict_label_encoder = make_X_tf(
        X_prod, cols_numericals, cols_categoricals, scaler, dict_label_encoder)

    model_tf = make_model(len(cols_numericals), categoricals_info)
    # ckpt = ModelCheckpoint("w_%d.h5", monitor='val_loss', verbose=-1, save_best_only=True, mode='min')
    # reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)
    # es = EarlyStopping(monitor='val_loss', patience=3)

    # print(model_tf.summary())
    # tf.keras.utils.plot_model(model_tf, to_file='model.png', show_shapes=True, show_layer_names=True)

    # if X_train.shape[0] > 100_000:
    #     EPOCH = 50

    # if X_train.shape[0] > 10_000_000:
    #     EPOCH = 20

    EPOCH = 25
    # BATCH_SIZE = max(10_000, X_train.shape[0])
    BATCH_SIZE = 50_000

    print('TODO handle early stopping!')
    model_tf.fit(X_train_tf, y_train.values,
                 #  sample_weight=weight_train.values,
                 batch_size=BATCH_SIZE, epochs=EPOCH,
                 validation_data=(X_valid_tf, y_valid.values),
                 verbose=1)

    # model_tf.fit((X_train_tf, y_train.values, weight_train.values), batch_size=BATCH_SIZE, epochs=EPOCH,
    #              validation_data=(X_valid_tf, y_valid.values, weight_valid.values), verbose=1)

    cols_pred = ['quantile_{:.3f}'.format(quantile) for quantile in constants.LIST_QUANTILE]

    preds_valid_oof = model_tf.predict(X_valid_oof_tf)
    preds_prod = model_tf.predict(X_prod_tf)

    for i, col_pred in enumerate(cols_pred):
        df_valid_oof[col_pred] = preds_valid_oof[:, i]
        df_prod[col_pred] = preds_prod[:, i]

    if constants.RETRAIN_FULL_DATA:
        model_tf = make_model(len(cols_numericals), categoricals_info)
        model_tf.fit(X_learning_tf, y_learning.values,
                     #  sample_weight=weight_train.values,
                     batch_size=BATCH_SIZE, epochs=EPOCH,
                     verbose=1)

        preds_prod = model_tf.predict(X_prod_tf)

        for i, col_pred in enumerate(cols_pred):
            df_prod[col_pred] = preds_prod[:, i]

    df_valid_oof = prep.compute_pinball(df_valid_oof)

    return df_valid_oof, df_prod


def _strategy_granularity(df_learning, df_prod, func_train_predict):
    df_valid_predicted = []
    df_prod_predicted = []
    for granularity in df_learning['granularity'].unique():
        print(granularity)
        df_learning_tmp = df_learning[(
            df_learning['granularity'] == granularity)]
        df_prod_tmp = df_prod[df_prod['granularity'] == granularity]
        df_valid_predicted_tmp, df_prod_predicted_tmp = (
            func_train_predict(
                df_learning=df_learning_tmp,
                df_prod=df_prod_tmp,
                verbose_eval=10000,
            )
        )
        print(granularity, ml_utils.wavg(df_valid_predicted_tmp, 'pinball', 'weight').round(3))
        df_valid_predicted.append(df_valid_predicted_tmp)
        df_prod_predicted.append(df_prod_predicted_tmp)

    df_valid_predicted = pd.concat(df_valid_predicted)
    df_prod_predicted = pd.concat(df_prod_predicted)
    return df_valid_predicted, df_prod_predicted


def strategy_granularity(df_learning, df_prod):
    """
    Apply strategy of learning one model for each granularity
    """
    df_valid_predicted, df_prod_predicted = _strategy_granularity(
        df_learning, df_prod, train_predict_lgb_quantile)
    return df_valid_predicted, df_prod_predicted


def strategy_granularity_point_to_uncertainity(df_learning, df_prod):
    """
    Apply strategy of learning one model for each granularity
    """
    df_valid_predicted, df_prod_predicted = _strategy_granularity(
        df_learning, df_prod, func_train_predict=train_predict_lgb_point_to_uncertainity)
    return df_valid_predicted, df_prod_predicted


def strategy_granularity_point_to_uncertainity_tweedie(df_learning, df_prod):
    """
    Apply strategy of learning one model for each granularity
    """
    df_valid_predicted, df_prod_predicted = _strategy_granularity(
        df_learning, df_prod, train_predict_lgb_tweedie)
    return df_valid_predicted, df_prod_predicted


def strategy_granularity_ngboost(df_learning, df_prod):
    """
    Apply strategy of learning one model for each granularity
    """
    df_valid_predicted, df_prod_predicted = _strategy_granularity(
        df_learning, df_prod, train_predict_ngboost)
    return df_valid_predicted, df_prod_predicted


def _strategy_serie(df_learning, df_prod, func_train_predict):
    """
    Apply strategy of learning one model for each serie.
    Applied only on granularity where number of series (group) is small. (eg not item_id).
    Usefull when granularity is small and series more different than they look alike
    """
    df_valid_predicted = []
    df_prod_predicted = []

    granularity_series = [granularity for granularity in df_learning['granularity'].unique(
    ) if 'item_id' not in granularity]

    for granularity in granularity_series:
        print(granularity)
        df_learning_tmp = df_learning[(
            df_learning['granularity'] == granularity)]
        for group in df_learning_tmp['group'].unique().tolist():
            df_learning_tmp.pipe(lambda df: df[df['group'] == group])
            df_valid_predicted_tmp, df_prod_predicted_tmp = (
                func_train_predict(
                    df_learning_tmp.pipe(lambda df: df[df['group'] == group]),
                    df_prod[df_prod['granularity'] == granularity].pipe(
                        lambda df: df[df['group'] == group]),
                    verbose_eval=False,
                )
            )
            print(granularity, df_valid_predicted_tmp.pinball.mean().round(6))
            df_valid_predicted.append(df_valid_predicted_tmp)
            df_prod_predicted.append(df_prod_predicted_tmp)

    df_valid_predicted = pd.concat(df_valid_predicted)
    df_prod_predicted = pd.concat(df_prod_predicted)
    return df_valid_predicted, df_prod_predicted


def strategy_serie(df_learning, df_prod):
    df_valid_predicted, df_prod_predicted = _strategy_serie(
        df_learning, df_prod, train_predict_lgb_quantile)
    return df_valid_predicted, df_prod_predicted


def strategy_serie_tweedie(df_learning, df_prod):
    df_valid_predicted, df_prod_predicted = _strategy_serie(
        df_learning, df_prod, train_predict_lgb_tweedie)
    return df_valid_predicted, df_prod_predicted


def strategy_serie_point_to_uncertainity(df_learning, df_prod):
    df_valid_predicted, df_prod_predicted = _strategy_serie(
        df_learning, df_prod, train_predict_lgb_point_to_uncertainity)
    return df_valid_predicted, df_prod_predicted


def strategy_serie_tensorflow(df_learning, df_prod):
    df_valid_predicted, df_prod_predicted = _strategy_serie(
        df_learning, df_prod, train_predict_tensorflow_quantile)
    return df_valid_predicted, df_prod_predicted


def strategy_granularity_tensorflow(df_learning, df_prod):
    df_learning = df_learning[df_learning['granularity'].str.contains('item_id')]
    df_prod = df_prod[df_prod['granularity'].str.contains('item_id')]
    df_valid_predicted, df_prod_predicted = _strategy_granularity(
        df_learning, df_prod, train_predict_tensorflow_quantile)
    return df_valid_predicted, df_prod_predicted


def _strategy_hierarchical(df_learning, df_prod, func_train_predict):
    """
    Apply strategy of learning one model for each serie.
    Applied only on granularity where number of series (group) is small. (eg not item_id).
    Usefull when granularity is small and series more different than they look alike
    """
    df_valid_predicted = []
    df_prod_predicted = []

    print('Warning - only one hierarchy by granularity')

    # Dict with keys a granularity
    # Values : list of list of hierarchical group
    hierachical_models = {}
    # hierachical_models['dept_id'] = [
    #     ['cat_id']
    # ]

    # No hierarchy here
    # hierachical_models['state_id_&_cat_id'] = [
    #     []
    # ]

    # hierachical_models['state_id_&_dept_id'] = [
    #     ['state_id'],
    # ]

    # hierachical_models['store_id_&_cat_id'] = [
    #     ['cat_id'],
    # ]

    # hierachical_models['store_id_&_dept_id'] = [
    #     ['store_id', 'cat_id']
    # ]

    hierachical_models['item_id'] = [
        ['dept_id'],
    ]

    hierachical_models['state_id_&_item_id'] = [
        ['state_id', 'dept_id'],
    ]

    hierachical_models['item_id_&_store_id'] = [
        # ['cat_id', 'state_id', 'dept_id'],
        ['store_id', 'dept_id'],
    ]

    for granularity in hierachical_models.keys():
        print('granularity:', granularity)
        df_learning_tmp = df_learning[(
            df_learning['granularity'] == granularity)]
        df_prod_tmp = df_prod[(df_prod['granularity'] == granularity)]
        for hierarchy in hierachical_models[granularity]:
            df_learning_groups = df_learning_tmp.groupby(
                hierarchy, observed=True)
            df_prod_groups = df_prod_tmp.groupby(hierarchy, observed=True)
            print('Number of groups in the hierarchy : ',
                  len(df_learning_groups))
            for hierarchy_name, df_learning_group_tmp in df_learning_groups:
                print(hierarchy_name)
                df_prod_groups_tmp = df_prod_groups.get_group(hierarchy_name)
                df_valid_predicted_tmp, df_prod_predicted_tmp = (
                    func_train_predict(
                        df_learning_group_tmp,
                        df_prod_groups_tmp,
                        verbose_eval=False,
                    )
                )
                print(granularity, df_valid_predicted_tmp.pinball.mean().round(6))
                df_valid_predicted.append(df_valid_predicted_tmp)
                df_prod_predicted.append(df_prod_predicted_tmp)

    df_valid_predicted = pd.concat(df_valid_predicted)
    df_prod_predicted = pd.concat(df_prod_predicted)
    return df_valid_predicted, df_prod_predicted


def strategy_hierarchical(df_learning, df_prod):
    df_valid_predicted, df_prod_predicted = _strategy_hierarchical(
        df_learning, df_prod, train_predict_lgb_quantile)
    return df_valid_predicted, df_prod_predicted


def strategy_horizon_chunk(df_learning, df_prod, strategy_func):
    """
    Apply strategy of learning one model for each horizon. with the according sub-strategy (hierachical, serie, granularity...)
    """
    print('Test')
    df_valid_predicted = []
    df_prod_predicted = []

    df_learning['horizon_chunk'] = (df_learning['horizon'] - 1) // 7
    df_prod['horizon_chunk'] = (df_prod['horizon'] - 1) // 7

    list_horizon = df_learning['horizon_chunk'].unique()
    list_horizon.sort()

    for horizon_chunk in list_horizon:
        print(horizon_chunk)
        df_learning_tmp = df_learning[(
            df_learning['horizon_chunk'] == horizon_chunk)]
        df_prod_tmp = df_prod[(df_prod['horizon_chunk'] == horizon_chunk)]
        df_valid_predicted_tmp, df_prod_predicted_tmp = strategy_func(
            df_learning_tmp, df_prod_tmp)

        df_valid_predicted.append(df_valid_predicted_tmp)
        df_prod_predicted.append(df_prod_predicted_tmp)

    df_valid_predicted = pd.concat(df_valid_predicted)
    df_prod_predicted = pd.concat(df_prod_predicted)

    df_learning = df_learning['horizon_chunk'].drop(columns='horizon_chunk')
    df_prod = df_prod['horizon_chunk'].drop(columns='horizon_chunk')

    return df_valid_predicted, df_prod_predicted


def strategy_horizon(df_learning, df_prod, strategy_func):
    """
    Apply strategy of learning one model for each horizon. with the according sub-strategy (hierachical, serie, granularity...)
    """
    print('Test')
    df_valid_predicted = []
    df_prod_predicted = []

    list_horizon = df_learning['horizon'].unique()
    list_horizon.sort()

    for horizon in list_horizon:
        print('horizon:', horizon)
        df_learning_tmp = df_learning[(df_learning['horizon'] == horizon)]
        df_prod_tmp = df_prod[(df_prod['horizon'] == horizon)]
        df_valid_predicted_tmp, df_prod_predicted_tmp = strategy_func(
            df_learning_tmp, df_prod_tmp)

        df_valid_predicted.append(df_valid_predicted_tmp)
        df_prod_predicted.append(df_prod_predicted_tmp)

    df_valid_predicted = pd.concat(df_valid_predicted)
    df_prod_predicted = pd.concat(df_prod_predicted)

    return df_valid_predicted, df_prod_predicted


def strategy_horizon_serie(df_learning, df_prod):
    return strategy_horizon(df_learning, df_prod, strategy_serie)


def strategy_horizon_granularity(df_learning, df_prod):
    # Keep only granularities related to item_id
    granularity_item = [granularity for granularity in df_learning['granularity'].unique(
    ) if 'item_id' in granularity]
    df_learning = df_learning[df_learning['granularity'].isin(
        granularity_item)]
    df_prod = df_prod[df_prod['granularity'].isin(granularity_item)]
    return strategy_horizon(df_learning, df_prod, strategy_granularity)


def strategy_horizon_hierarchical(df_learning, df_prod):
    # Keep only granularities related to item_id
    # granularity_item = [granularity for granularity in df_learning['granularity'].unique() if 'item_id' in granularity]
    granularity_item = [
        # 'item_id_&_store_id',
        # 'state_id_&_item_id',
        'item_id',
    ]
    df_learning = df_learning[df_learning['granularity'].isin(
        granularity_item)]
    df_prod = df_prod[df_prod['granularity'].isin(granularity_item)]
    return strategy_horizon(df_learning, df_prod, strategy_hierarchical)
