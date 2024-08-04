from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import optuna
import warnings
import os
import shutil
import time


from sklearn.ensemble import GradientBoostingRegressor
# from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error

script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(script_dir, 'optuna_outputs')
# data_dir = "/Users/priyaandurkar/Documents/Spring 2024/ASE/Project/se_data"

if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
os.makedirs(output_dir, exist_ok=True)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

def objective(trial):
    # Define the hyperparameter space with conditional logic
    max_features_option = trial.suggest_categorical('max_features_option', ['sqrt', 'log2', 'fraction'])
    if max_features_option == 'fraction':
        max_features = trial.suggest_float('max_features', 0.001, 0.999)
    else:
        max_features = max_features_option

    max_leaf_nodes_option = trial.suggest_categorical('max_leaf_nodes_option', ['none', 'int'])
    if max_leaf_nodes_option == 'int':
        max_leaf_nodes = trial.suggest_int('max_leaf_nodes', 2, 50)
    else:
        max_leaf_nodes = None

    n_iter_no_change_option = trial.suggest_categorical('n_iter_no_change_option', ['none', 'int'])
    if n_iter_no_change_option == 'int':
        n_iter_no_change = trial.suggest_int('n_iter_no_change', 5, 20)
    else:
        n_iter_no_change = None

    param = {
        'loss': trial.suggest_categorical('loss', ['squared_error', 'absolute_error', 'huber', 'quantile']),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 1.2),
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'subsample': trial.suggest_float('subsample', 0.01, 1),
        'criterion': trial.suggest_categorical('criterion', ['friedman_mse', 'squared_error']),
        'min_samples_split': trial.suggest_float('min_samples_split', 0.01, 0.999),
        'min_samples_leaf': trial.suggest_float('min_samples_leaf', 0.01, 0.5),
        'min_weight_fraction_leaf': trial.suggest_float('min_weight_fraction_leaf', 0.0, 0.5),
        'max_depth': trial.suggest_categorical('max_depth', [1, 2, 3, 4, 5, None]),
        'min_impurity_decrease': trial.suggest_float('min_impurity_decrease', 0.0, 0.5),
        'init': trial.suggest_categorical('init', [None, 'zero']),
        'max_features': max_features,  # Use the conditionally chosen value
        'alpha': trial.suggest_float('alpha', 0.01, 0.999),
        'max_leaf_nodes': max_leaf_nodes,  # Use the conditionally chosen value
        'warm_start': trial.suggest_categorical('warm_start', [False, True]),
        'validation_fraction': trial.suggest_float('validation_fraction', 0.01, 0.999),
        'n_iter_no_change': n_iter_no_change,  # Use the conditionally chosen value
        'tol': trial.suggest_float('tol', 0.5e-4, 1e-3),
        'ccp_alpha': trial.suggest_float('ccp_alpha', 0.0, 0.02)
    }

    # Create the model with the current hyperparameters
    model = GradientBoostingRegressor(**param)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    return rmse

inputs = ['dtlz2','dtlz3','pom3a','pom3b','pom3c','SS-A','SS-B','SS-C','SS-D','Wine_quality']
n_trials_list = [6,12,20,60,100] #50,100,200,500
iters = 20

res = {}
for inputf in inputs:
    acc_dict = {}
    time_dict = {}
    res[inputf] = {}
    for n_trials in n_trials_list:
        start_time = time.time()
        with open((os.path.join(output_dir,inputf+'.txt')), 'a') as file:
            file.write('\n_______________________________\noptuna' + str(n_trials)+'\n_______________________________\n')
        acc_list = []
        acc = 0
        i = iters
        while i > 0:
            X = pd.read_csv(os.path.join(script_dir, '..', '..', 'base_model', 'data', 'x_'+inputf+'.csv')).values
            y = pd.read_csv(os.path.join(script_dir, '..', '..', 'base_model', 'data', 'y_'+inputf+'.csv')).values
            study = optuna.create_study(direction = 'maximize' , pruner = optuna.pruners.HyperbandPruner())
            study.optimize(objective, n_trials = n_trials)
            acc_list.append(study.best_value)
            acc += float(study.best_value)
            with open((os.path.join(output_dir,inputf+'.txt')), 'a') as file:
                file.write('\n\nthe best params:' + str(study.best_trial.params))
                file.write('\nthe best value:' + str(study.best_value))
            i -= 1
        with open((os.path.join(output_dir,inputf+'.txt')), 'a') as file:
            file.write('\n*****************************\nAverage optimal accuracy:' + str(acc/iters)+'\n*****************************\n')
        end_time = time.time()
        acc_dict['optuna'+str(n_trials)] = acc_list
        acc_r_list = [round(x, 2) for x in acc_list]
        res[inputf]['optuna'+str(n_trials)] = acc_r_list
        time_dict['optuna'+str(n_trials)] = end_time - start_time
    with open((os.path.join(output_dir,inputf+'.txt')), 'a') as file:
        file.write('\n---------------------\nAccuracy\n---------------------\n' + str(acc_dict)+'\n---------------------\n')
        file.write('\n---------------------\nTime\n---------------------\n' + str(time_dict)+'\n---------------------\n')
with open((os.path.join(output_dir,'optuna_final.txt')), 'a') as file:
    file.write('\n---------------------\nAggregate\n---------------------\n' + str(res)+'\n---------------------\n')