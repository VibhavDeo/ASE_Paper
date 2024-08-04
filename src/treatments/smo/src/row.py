import math
from config import the

from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import warnings
from sklearn.exceptions import DataConversionWarning,ConvergenceWarning
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import os
import warnings
import numpy as np
import json
import os

class ROW:
    # Initializing ROW instance
    def __init__(self, t):
        self.cells = t


    def d2h(self,input,seed ,data=None):
        warnings.filterwarnings("ignore", message="Setting penalty=None will ignore the C and l1_ratio parameters")
        warnings.filterwarnings("ignore", category=DataConversionWarning)
        warnings.filterwarnings("ignore", category=ConvergenceWarning)

        if(len(self.cells)==19):
            # X, Y = pd.read_csv("/Users/priyaandurkar/Documents/ASE_Paper/src/base_model/data/x_"+input+".csv"),pd.read_csv("/Users/priyaandurkar/Documents/ASE_Paper/src/base_model/data/y_"+input+".csv")
            X, Y = pd.read_csv("/home/pandurk/ASE_Paper/src/base_model/data/x_"+input+".csv"),pd.read_csv("/home/pandurk/ASE_Paper/src/base_model/data/y_"+input+".csv")
       
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=seed)

            # Train a MultiOutput Gradient Boosting Regressor
            # logistic = linear_model.LogisticRegression(max_iter=int(self.cells[0]), C=(self.cells[1]), tol=self.cells[2], fit_intercept=self.cells[3], dual=self.cells[4], penalty=self.cells[5])
            gbr = GradientBoostingRegressor(random_state=seed,
                                            alpha=self.cells[0],
                                            ccp_alpha=self.cells[1],
                                            criterion=self.cells[2],
                                            init=self.cells[3],
                                            learning_rate=self.cells[4],
                                            loss=self.cells[5],
                                            max_depth=int(self.cells[6]) if self.cells[6]!=None else None,
                                            max_features=self.cells[7],
                                            max_leaf_nodes=int(self.cells[8]) if self.cells[8]!=None else None,
                                            min_impurity_decrease=self.cells[9],
                                            min_samples_leaf=int(self.cells[10]) if self.cells[10]>1.0 else math.ceil(self.cells[10]),
                                            min_samples_split=self.cells[11],
                                            min_weight_fraction_leaf=self.cells[12],
                                            n_estimators=int(self.cells[13]) if self.cells[13]!=None else None,
                                            n_iter_no_change=int(self.cells[14]) if self.cells[14]!=None else None,
                                            subsample=self.cells[15] if self.cells[15]<1 else 0.0,
                                            tol=self.cells[16],
                                            warm_start=self.cells[17],
                                            validation_fraction=self.cells[18] if self.cells[18]!=None else 0.1)
                                
            multi_target_regressor = MultiOutputRegressor(gbr)
            multi_target_regressor.fit(X_train, Y_train)

            # Predict the targets
            Y_pred = multi_target_regressor.predict(X_test)

            percentage_rmse_values = []
            for i in range(Y_test.shape[1]):
                mean_value = np.mean(Y_test.iloc[:, i])
                rmse = np.sqrt(mean_squared_error(Y_test.iloc[:, i], Y_pred[:, i]))
                percentage_rmse = (rmse / mean_value) * 100
                percentage_rmse_values.append(percentage_rmse)

            # Calculate the combined percentage RMSE
            score = round(np.mean(percentage_rmse_values),3 )
            
            self.cells.append(score)
            return score
        else:
            return self.cells[19]
        

    #Finding out how much a row likes the data
    def like(self, data, n, nHypotheses, the):
        # print(the)
        prior = (len(data.rows) + the['k']) / (n + the['k'] * nHypotheses)
        out = math.log(prior)

        for col in data.cols.x:
            v = self.cells[col.at]
            if v != "?":
                inc = col.like(v, prior, the)
                if inc == 0:
                    out += float('-inf')
                else:
                    out += math.log(inc)

        return math.exp(1) ** out
    
    # Classifier
    def likes(self,datas):
        n, nHypotheses = 0, 0
        most, out = None, None
        # print(the)

        for k, data in datas.items():
            n += len(data.rows)
            nHypotheses += 1

        for k, data in datas.items():
            tmp = self.like(data, n, nHypotheses, the)
            if most is None or tmp > most:
                most, out = tmp, k

        return out, most
