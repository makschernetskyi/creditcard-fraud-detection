import pandas as pd
from sklearn.preprocessing import StandardScaler

from services import base_data_preprocessor
from sklearn.pipeline import Pipeline
from .functions import grid_search_best_params
from sklearn.ensemble import RandomForestClassifier
from .grid_search_cv_paramaters import random_forest_parameters
from utils import add_prefix_to_dict_keys
from sklearn.model_selection import train_test_split


file_csv = "./dataset/creditcard.csv"

df = pd.read_csv(file_csv)

preprocessed_df = base_data_preprocessor(df, ['V1','V2','V3','V4','V5','V6','V7','V9','V10','V11','V12','V14','V16','V17','V18','Class'])

X = preprocessed_df[['V1','V2','V3','V4','V5','V6','V7','V9','V10','V11','V12','V14','V16','V17','V18']]
y = preprocessed_df['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

random_forest_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier())
])

params = add_prefix_to_dict_keys(random_forest_parameters, "classifier__")




best_params = grid_search_best_params(random_forest_pipeline, params, X_train, y_train)


