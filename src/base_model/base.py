import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import os
import warnings

# Suppress the specific FutureWarning
warnings.simplefilter(action='ignore', category=FutureWarning)

script_dir = os.path.dirname(os.path.abspath(__file__))

x_file = os.path.join(script_dir,'data', 'dtlz2_X.csv')
x_file = os.path.abspath(x_file)
y_file = os.path.join(script_dir, 'data', 'dtlz2_Y.csv')
y_file = os.path.abspath(y_file)

X = pd.read_csv(x_file)
Y = pd.read_csv(y_file)

# Split the data into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Train a MultiOutput Gradient Boosting Regressor
gbr = GradientBoostingRegressor(random_state=42)
multi_target_regressor = MultiOutputRegressor(gbr)
multi_target_regressor.fit(X_train, Y_train)

# Predict the targets
Y_pred = multi_target_regressor.predict(X_test)

# Evaluate the model
# for i, target in enumerate(Y.columns):
#     mse = mean_squared_error(Y_test.iloc[:, i], Y_pred[:, i])
#     print(f'Mean Squared Error for {target}: {mse}')

# Display the trained model and its predictions
# print(multi_target_regressor, Y_pred)
