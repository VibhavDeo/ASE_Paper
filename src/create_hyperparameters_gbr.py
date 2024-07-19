import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterSampler
from scipy.stats import uniform, randint
import os

script_dir = os.path.dirname(os.path.abspath(__file__))

param_dist = {
    'loss': ['squared_error', 'absolute_error', 'huber', 'quantile'],
    'learning_rate': uniform(0.0001, ), 
    'n_estimators': randint(50, 300),
    'subsample': uniform(0.6, 0.4), 
    'criterion': ['friedman_mse', 'squared_error'],
    'min_samples_split': uniform(0.01, 0.99), 
    'min_samples_leaf': uniform(0.01, 0.99), 
    'min_weight_fraction_leaf': uniform(0.0, 0.5), 
    'max_depth': [3, 5, None],
    'min_impurity_decrease': uniform(0.0, 0.1), 
    'init': [None, 'zero'],
    'max_features': ['sqrt', 'log2', uniform(0.1, 0.9)], 
    'alpha': uniform(0.1, 0.9), 
    'verbose': [0, 1],
    'max_leaf_nodes': [None, randint(10, 100)], 
    'warm_start': [False, True],
    'validation_fraction': uniform(0.1, 0.3), 
    'n_iter_no_change': [None, randint(5, 20)], 
    'tol': uniform(1e-4, 1e-3),  
    'ccp_alpha': uniform(0.0, 0.01)  
}

n_iter = 50

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

output_file = os.path.join(script_dir, '..', 'data', 'gbr_hyperparameters_'+str(n_iter)+'.csv')
output_file = os.path.abspath(output_file)

df.to_csv(output_file, index=False, na_rep='None')