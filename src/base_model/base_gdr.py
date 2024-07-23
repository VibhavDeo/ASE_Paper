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


warnings.simplefilter(action='ignore', category=FutureWarning)

script_dir = os.path.dirname(os.path.abspath(__file__))

input_files = [
    ('data/x_dtlz2.csv', 'data/y_dtlz2.csv'),
    ('data/x_dtlz3.csv', 'data/y_dtlz3.csv'),
    ('data/x_dtlz4.csv', 'data/y_dtlz4.csv'),
    ('data/x_dtlz5.csv', 'data/y_dtlz5.csv'),
    ('data/x_dtlz6.csv', 'data/y_dtlz6.csv'),
    ('data/x_dtlz7.csv', 'data/y_dtlz7.csv'),
    ('data/x_pom3a.csv', 'data/y_pom3a.csv'),
    ('data/x_pom3b.csv', 'data/y_pom3b.csv'),
    ('data/x_pom3c.csv', 'data/y_pom3c.csv'),
    ('data/x_pom3d.csv', 'data/y_pom3d.csv'),
    ('data/x_SS-A.csv', 'data/y_SS-A.csv'),
    ('data/x_SS-B.csv', 'data/y_SS-B.csv'),
    ('data/x_SS-C.csv', 'data/y_SS-C.csv'),
    ('data/x_SS-D.csv', 'data/y_SS-D.csv'),
]


def calculate_combined_percentage_rmse(x_file, y_file):
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

    percentage_rmse_values = []
    for i in range(Y_test.shape[1]):
        mean_value = np.mean(Y_test.iloc[:, i])
        rmse = np.sqrt(mean_squared_error(Y_test.iloc[:, i], Y_pred[:, i]))
        percentage_rmse = (rmse / mean_value) * 100
        percentage_rmse_values.append(percentage_rmse)

    # Calculate the combined percentage RMSE
    combined_percentage_rmse = np.mean(percentage_rmse_values)

    return combined_percentage_rmse


data = {}
for x_file, y_file in input_files:
    x_file_path = os.path.join(script_dir, x_file)
    y_file_path = os.path.join(script_dir, y_file)
    combined_percentage_rmse = calculate_combined_percentage_rmse(x_file_path, y_file_path)
    print(f"Combined Percentage RMSE for {x_file} and {y_file}: {combined_percentage_rmse}")

    data[x_file] = combined_percentage_rmse

print(data)

