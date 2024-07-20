import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterSampler
from scipy.stats import uniform, randint
import os

script_dir = os.path.dirname(os.path.abspath(__file__))

# https://medium.com/all-things-ai/in-depth-parameter-tuning-for-gradient-boosting-3363992e9bae (reference for selecting the  below values)[n_iter_no_change]
# https://towardsdatascience.com/machine-learning-part-18-boosting-algorithms-gradient-boosting-in-python-ef5ae6965be4

param_dist = {
    'loss': ['squared_error', 'absolute_error', 'huber', 'quantile'],
    'learning_rate': uniform(0.001, 1.2),   # Learning rate of ML algorithms is usually between 0 to 1 
    'n_estimators': randint(50, 500),   # General values used
    'subsample': uniform(0.001, 1), 
    'criterion': ['friedman_mse', 'squared_error'],
    'min_samples_split': uniform(0.001, 0.999), 
    'min_samples_leaf': uniform(0.01, 0.999), 
    'min_weight_fraction_leaf': uniform(0.0, 0.5), 
    'max_depth': [1, 2, 3, 4, 5, None],
    'min_impurity_decrease': uniform(0.0, 0.5), # As default is 0, selecting space closer to 0
    'init': [None, 'zero'],
    'max_features': ['sqrt', 'log2', uniform(0.001, 0.999)], 
    'alpha': uniform(0.001, 0.999),
    'max_leaf_nodes': [None, randint(2, 50)],
    'warm_start': [False, True],
    'validation_fraction': uniform(0.001, 0.999), 
    'n_iter_no_change': [None, randint(5, 20)], 
    'tol': uniform(0.5e-4, 1e-3),  
    'ccp_alpha': uniform(0.0, 0.02) # As default is 0, selecting space closer to 0
}

n_iter = 1000

random_search = list(ParameterSampler(param_dist, n_iter=n_iter, random_state=42))

valid_samples = []
for params in random_search:
    # Filter out invalid combinations
    if params['loss'] in ['huber', 'quantile'] or 'alpha' not in params:
        pass
    else:
        params.pop('alpha', None)
    
    if params['n_iter_no_change'] is not None or 'validation_fraction' not in params:
        pass
    else:
        params.pop('validation_fraction', None)

    # Fetch random from parameter space 
    if params['n_iter_no_change']!= None:
        params['n_iter_no_change'] = params['n_iter_no_change'].rvs()

    if params['max_leaf_nodes'] != None:
        params['max_leaf_nodes'] = params['max_leaf_nodes'].rvs()

    if not isinstance(params['max_features'], str):
        params['max_features'] = params['max_features'].rvs()
    
    # Add the parameter row to the result
    valid_samples.append(params)

df = pd.DataFrame(valid_samples)
df = df.replace(np.nan, None)

# print(df)

output_file = os.path.join(script_dir, '..', '..', 'data', 'gbr_hyperparameters_'+str(n_iter)+'.csv')
output_file = os.path.abspath(output_file)

df.to_csv(output_file, index=False, na_rep='None')