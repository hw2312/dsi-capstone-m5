# Allows sampling and stuff for fast protyping/dev
IS_PROTOTYPING = False
str_prototyping = '_is_prototyping' if IS_PROTOTYPING else ''

# See if it should be merge with IS_PROTOTYPING
# Params to wether retrain models on full data
RETRAIN_FULL_DATA = True

# List of quantiles for which we have a make a predictions
# Obtained this way => sample_submission['id'].str.split('_').str[-2].drop_duplicates().astype(float).tolist()
LIST_QUANTILE = [0.005, 0.025, 0.165, 0.25, 0.5, 0.75, 0.835, 0.975, 0.995]

AUGMENTATION_SIZE = 28
assert AUGMENTATION_SIZE <= 28  # If augmentation size, bigger than this, we recreate sales_begin

# Downsampling size for item's granularities
DOWNSAMPLING_SIZE = 2


l_tuple_strategy_normalised = [
    # ('serie', None),
    # ('serie', 'sales_mean_7'),
    # ('serie', 'sales_mean_14'),
    ('serie', 'sales_mean_28'),
    # # ('serie', 'sales_mean_56'),
    # # ('serie', 'sales_mean_112'),
    ('granularity', None),
    # ('granularity', 'sales_mean_28'),
    ('hierarchical', None),
    # ('hierarchical', 'sales_mean_28'),
    ('horizon_granularity', None),
    # Currently wayyyyy to slow on all item's granularities, only run on item_id, then see if it's worth it
    # ('horizon_hierarchical', None),
    # Tweedie never works and a bit slower so remove it
    # ('granularity_tweedie', None),
    # ('granularity_tweedie', 'sales_mean_28'),
    ('granularity_point_to_uncertainity', None),
    ('granularity_point_to_uncertainity', 'sales_mean_28'),
    # ('serie_tweedie', 'sales_mean_28'),
    ('serie_point_to_uncertainity', 'sales_mean_14'),
    ('serie_point_to_uncertainity', 'sales_mean_28'),
    ('serie_point_to_uncertainity', 'sales_mean_56'),
    ('serie_point_to_uncertainity', 'sales_mean_112'),
    # Ngboost is too slow + on total_id is worse than point_to_uncertainity approach
    # ('granularity_ngboost', None),
    # ('granularity_ngboost', 'sales_mean_28'),
    ('granularity_tensorflow', None),
]
