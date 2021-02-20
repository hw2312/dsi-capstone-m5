
# Specified LGBM Hyperparameters

import pandas as pd

#  A1 #######################################################
a1_params = [ {
                    'team':'A1',
                    'file':'1-1. recursive_store_TRAIN.py',
                    'boosting_type': 'gbdt',
                    'objective': 'tweedie',
                    'tweedie_variance_power': 1.1,
                    'metric': 'rmse',
                    'subsample': 0.5,
                    'subsample_freq': 1,
                    'learning_rate': 0.015,
                    'num_leaves': 2**11-1,
                    'min_data_in_leaf': 2**12-1,
                    'feature_fraction': 0.5,
                    'max_bin': 100,
                    'n_estimators': 3000,
                    'boost_from_average': False,
                    'verbose': -1,
                },
{
                    'team':'A1',
                    'file':'1-2. recursive_store_cat_TRAIN.py',
                    'boosting_type': 'gbdt',
                    'objective': 'tweedie',
                    'tweedie_variance_power': 1.1,
                    'metric': 'rmse',
                    'subsample': 0.5,
                    'subsample_freq': 1,
                    'learning_rate': 0.015,
                    'num_leaves': 2**8-1,
                    'min_data_in_leaf': 2**8-1,
                    'feature_fraction': 0.5,
                    'max_bin': 100,
                    'n_estimators': 3000,
                    'boost_from_average': False,
                    'verbose': -1
                } ,
{
                    'team':'A1',
                    'file':'1-3. recursive_store_dept_TRAIN.py',
                    'boosting_type': 'gbdt',
                    'objective': 'tweedie',
                    'tweedie_variance_power': 1.1,
                    'metric': 'rmse',
                    'subsample': 0.5,
                    'subsample_freq': 1,
                    'learning_rate': 0.015,
                    'num_leaves': 2**8-1,
                    'min_data_in_leaf': 2**8-1,
                    'feature_fraction': 0.5,
                    'max_bin': 100,
                    'n_estimators': 3000,
                    'boost_from_average': False,
                    'verbose': -1
                },
{
                    'team':'A1',
                    'file':'2-1. nonrecursive_store_TRAIN.py',
                    'boosting_type': 'gbdt',
                    'objective': 'tweedie',
                    'tweedie_variance_power': 1.1,
                    'metric': 'rmse',
                    'subsample': 0.5,
                    'subsample_freq': 1,
                    'learning_rate': 0.015,
                    'num_leaves': 2**8-1,
                    'min_data_in_leaf': 2**8-1,
                    'feature_fraction': 0.5,
                    'max_bin': 100,
                    'n_estimators': 3000,
                    'boost_from_average': False,
                    'verbose': -1,
                    'seed' : 1995
                } ,

 {
                    'team':'A1',
                    'file':'2-2. nonrecursive_store_cat_TRAIN.py',
                    'boosting_type': 'gbdt',
                    'objective': 'tweedie',
                    'tweedie_variance_power': 1.1,
                    'metric': 'rmse',
                    'subsample': 0.5,
                    'subsample_freq': 1,
                    'learning_rate': 0.015,
                    'num_leaves': 2**8-1,
                    'min_data_in_leaf': 2**8-1,
                    'feature_fraction': 0.5,
                    'max_bin': 100,
                    'n_estimators': 3000,
                    'boost_from_average': False,
                    'verbose': -1,
                    'seed' : 1995
                } ,

 {
                    'team':'A1',
                    'file':'2-3. nonrecursive_store_dept_TRAIN.py',
                    'boosting_type': 'gbdt',
                    'objective': 'tweedie',
                    'tweedie_variance_power': 1.1,
                    'metric': 'rmse',
                    'subsample': 0.5,
                    'subsample_freq': 1,
                    'learning_rate': 0.015,
                    'num_leaves': 2**8-1,
                    'min_data_in_leaf': 2**8-1,
                    'feature_fraction': 0.5,
                    'max_bin': 100,
                    'n_estimators': 3000,
                    'boost_from_average': False,
                    'verbose': -1,
                    'seed' : 1995
                } ]

#  A2 #######################################################

a2_params = [{
        'team':'A2',
        'file':'m5-final-XX.py',
        'boosting_type': 'gbdt',
        'objective': 'tweedie',
        'tweedie_variance_power': 1.1,
        'metric':'rmse',
        'n_jobs': -1,
        'seed': 42,
        'learning_rate': 0.2,
        'bagging_fraction': 0.85,
        'bagging_freq': 1, 
        'colsample_bytree': 0.85,
        'colsample_bynode': 0.85,
        #'min_data_per_leaf': 25,
        #'num_leaves': 200,
        'lambda_l1': 0.5,
        'lambda_l2': 0.5
}]


#  A4 #######################################################

a4_params = [{
            'team':'A4',
            'file':'m5a_main.py',
            'boosting_type': 'gbdt',
            'objective': 'tweedie',
            'tweedie_variance_power': 1.1,
            'metric': 'rmse',
            'subsample': 0.5,
            'subsample_freq': 1,
            'learning_rate': 0.03,
            'num_leaves': 2 ** 11 - 1,
            'min_data_in_leaf': 2 ** 12 - 1,
            'feature_fraction': 0.5,
            'max_bin': 100,
            'n_estimators': 1400,
            'boost_from_average': False,
            'verbose': -1,
        }]
        
#  A5 #######################################################        
        
a5_params = [{
            'team':'A5',
            'file':'3-preproc-fast-4.py',
            "objective" : "poisson",
                  "metric" :"poisson",
                  "learning_rate" : 0.09,
                  "sub_feature" : 0.9,
                  "sub_row" : 0.75,
                  "bagging_freq" : 1,
                  "lambda_l2" : 0.1,
                  'verbosity': 1,
                  'num_iterations':2000,
                  'num_leaves': 32,
                  "min_data_in_leaf": 50}]
        
        
# A18 #######################################################        
a18_params = [{
            'team':'A18',
            'file':'s1a_baseline.ipynb',
        "objective" : "tweedie",
        "metric" :"rmse",
        "force_row_wise" : True,
        "learning_rate" : 0.075,
        "sub_feature" : 0.8,
        "sub_row" : 0.75,
        "bagging_freq" : 1,
        "lambda_l2" : 0.1,
        "metric": ["rmse"],
        "nthread": 8,
        "tweedie_variance_power":1.2,
        'verbosity': 1,
        'num_iterations' : 1500,
        'num_leaves': 128,
        "min_data_in_leaf": 104,
#     'device_type': 'gpu'
    }]

#  U1 #######################################################        
u1_params = [{
         
         'team':'U1'
         }   ]
     
#  U2 #######################################################        
u2_params = [{
    'team':'U2',
    'file':'fe.py',
                    'boosting_type': 'gbdt',
                    'objective': 'tweedie',
                    'tweedie_variance_power': 1.1,
                    'metric': 'rmse',
                    'subsample': 0.5,
                    'subsample_freq': 1,
                    'learning_rate': 0.03,
                    'num_leaves': 2**11-1,
                    'min_data_in_leaf': 2**12-1,
                    'feature_fraction': 0.5,
                    'max_bin': 100,
                    'n_estimators': 1400,
                    'boost_from_average': False,
                    'verbose': -1,
                }]

#  U3 #######################################################        
u3_params = [{'team':'U3',
              'file':'train.py',
                    'boosting_type': 'gbdt',
                    'objective': 'tweedie',
                    'tweedie_variance_power': 1.1,
                    'metric': 'rmse',
                    'subsample': 0.6,
                    'subsample_freq': 1,
                    'learning_rate': 0.02,
                    'num_leaves': 2**11-1,
                    'min_data_in_leaf': 2**12-1,
                    'feature_fraction': 0.6,
                    'max_bin': 100,
                    'n_estimators': 1600,
                    'boost_from_average': False,
                    'verbose': -1,
                    'num_threads': 12
                } ]     

#  U5 #######################################################
params = {
            'boosting_type': 'gbdt',
            'objective': "regression",
            'metric': 'custom',
            'subsample': 0.5,
            'subsample_freq': 1,
            'learning_rate': 0.01,
            'num_leaves': 2**11-1,   
            'min_data_in_leaf': 2**12-1,
            'feature_fraction': 0.8,
            'max_bin': 255,     
            'n_estimators': 1500,
            'boost_from_average': False,
            'verbose': -1,
              }

def default_params():
        params["subsample"] = 0.5
        params["learning_rate"] = 0.01
#         params["learning_rate"] = 0.08
        params["num_leaves"] = 2**11-1
        params["min_data_in_leaf"] = 2**12-1
        params["feature_fraction"] = 0.8
        params["max_bin"] = 255
        params["n_estimators"] =1500
        
def category_param(category):
    default_params()
    if category=="CA_1":
#         params["n_estimators"] =1200
        pass

    elif category=="CA_2":
        params["num_leaves"] = 2**8-1
        params["min_data_in_leaf"] = 2**8-1
        
    elif category=="CA_3":
        params["learning_rate"] = 0.03
        params["num_leaves"] = 2**8-1
        params["min_data_in_leaf"] = 2**8-1
        params["n_estimators"] = 2300
        
    elif category=="CA_4":
        params["feature_fraction"] = 0.5
        
    elif category=="TX_1":
        pass
    elif category=="TX_2":
        pass
    elif category=="TX_3":
        pass
    elif category=="WI_1":
        pass
    elif category=="WI_2":
        params["num_leaves"] = 2**8-1
        params["min_data_in_leaf"] = 2**8-1
        params["feature_fraction"] = 0.5
    elif category=="WI_3":
        pass

u5_params = []    
for x in ['CA_1','CA_2','CA_3','CA_4','TX_1','TX_2','TX_3','WI_1','WI_2','WI_3']:
    params = {
            'boosting_type': 'gbdt',
            'objective': "regression",
            'metric': 'custom',
            'subsample': 0.5,
            'subsample_freq': 1,
            'learning_rate': 0.01,
            'num_leaves': 2**11-1,   
            'min_data_in_leaf': 2**12-1,
            'feature_fraction': 0.8,
            'max_bin': 255,     
            'n_estimators': 1500,
            'boost_from_average': False,
            'verbose': -1,
              }
    category_param(x) 
    params['team'] = 'U5'
    params['file'] = 'train_private.py'
    params['store_id'] = x
    u5_params.append(params)
    
# U12 #######################################################
u12_params = [{
    'team':'U12',
    'file':'U12.ipynb',
    'seed':20,
          'objective':'quantile', 
          'alpha':0.005,
          'num_leaves':63,
          'max_depth':15,                    
          'lambda':0.1, 
          'bagging_fraction':0.66,
          'bagging_freq':1,
          'colsample_bytree':0.77,
          "force_row_wise" : True,
          'learning_rate':0.1
        }   ] 


# U18 #######################################################
u18_params = [{
    'team':'U18',
    'file':'m5-rolling-prediction.ipynb',
        'boosting_type': 'gbdt',
        'metric': 'rmse',
        'objective': 'poisson',
        'seed': 225,
        'learning_rate': 0.02,
        'lambda': 0.4, 
        'reg_alpha': 0.4, 
        'max_depth': 5, 
        'num_leaves': 64, 
        'bagging_fraction': 0.7,
        'bagging_freq' : 1,
        'colsample_bytree': 0.7
}]


# U24 #######################################################
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
        # print('dataset_nrows :', dataset_nrows)
        if dataset_nrows > 100_000:
            # Increase learning_rate
            param['learning_rate'] = 0.2
            param['bagging_fraction'] = 0.7

        if dataset_nrows > 500_000:
            param['bagging_fraction'] = 0.5

        if dataset_nrows > 1_000_000:
            param['learning_rate'] = 0.2
            param['bagging_fraction'] = 0.1

    # if constants.IS_PROTOTYPING:
    #     num_boost_round = 10
    #     param['bagging_fraction'] = param['bagging_fraction'] / 4
    param['team']='U24'
    param['file']='modeling.py'
    param['dataset_nrows']=dataset_nrows
    return param #, num_boost_round, early_stopping_rounds

u24_params = []
for o in ['quantile','regression', 'tweedie']:
    for n in [500001,1000001]:
        p = get_lgb_params(objective=o, dataset_nrows=n)
        u24_params.append(p)
        
        
##############################################################################

# Combine all of the parameters into a DataFrame
all_params = a1_params+a2_params+a4_params+a5_params+u2_params+u3_params+u5_params+ \
                u12_params+u18_params+u24_params
                
all_params_df = pd.concat([pd.DataFrame(d,  index=[0]) for d in all_params])

# Combine the aliased hyperparameters

hyp = {'bagging_fraction':['sub_row', 'subsample', 'bagging'],
       'bagging_freq':['subsample_freq'],
       'feature_fraction':['sub_feature','colsample_bytree'],
       'lambda_l1':['reg_alpha'],
       'lambda_l2':['lambda','reg_lambda'],
       'num_iterations':['num_iteration', 'n_iter', 'num_tree', 'num_trees', 
                         'num_round', 'num_rounds', 'num_boost_round', 
                         'n_estimators'],
       'num_threads':['num_thread', 'nthread', 'nthreads', 'n_jobs'],
       'num_leaves':['num_leaf', 'max_leaves', 'max_leaf'],
       'verbosity':['verbose']
       }    

for h, alias_l in hyp.items():
    for a in alias_l:
        if a in all_params_df.columns:
            all_params_df[h] = np.where(all_params_df[h].isna(), all_params_df[a], all_params_df[h])


# Drop the alised hyperparameters
from itertools import chain
    
alias_l = [c for c in list(chain(*[v for v in hyp.values()])) if c in all_params_df.columns]
all_params_df = all_params_df.drop(columns = alias_l)

# Rename to the primary hyperparameter name
all_params_df = all_params_df.rename(columns = {
    'colsample_bynode':'feature_fraction_bynode',
    'boosting_type':'boosting'
    })


# Fill in the missing values with the default values
defaults = {
    'boosting':'gbdt',
    'objective':'regression',
    'tweedie_variance_power':1.5,
    'metric':'',
    'learning_rate':0.1,
    'num_leaves':31,
    'min_data_in_leaf':20,
    'feature_fraction':1.0,
    'max_bin':255,
    'boost_from_average':True,
    'bagging_fraction':1.0,
    'bagging_freq':0,
    'feature_fraction_bynode':1.0,
    'lambda_l1':0.0,
    'lambda_l2':0.0,
    'num_iterations':100,
    'num_threads':0,
    'alpha':0.9,
    'max_depth':-1,
    'force_row_wise':False,
    'max_cat_to_onehot':4,
    'min_sum_hessian_in_leaf':1e-3,
    'verbosity':1
    }

for hyp, default in defaults.items():
    all_params_df[hyp] = all_params_df[hyp].fillna(default)
    
