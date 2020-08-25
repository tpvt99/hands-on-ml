import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from fetch_data import load_housing_data
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.base import TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from utils import find_metrics
from sklearn.tree import DecisionTreeRegressor
from utils import display_scores
from sklearn.model_selection import cross_val_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

class CombinedAttributesAdder(TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True):
        self.add_bedrooms_per_room = add_bedrooms_per_room
        self.room_idx = 3
        self.bedrooms_idx = 4
        self.population_idx = 5
        self.households_indx = 6

    def fit(self, X, y = None):
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame): # convert to np.ndarray
            X = X.copy().values

        assert isinstance(X, np.ndarray)

        rooms_per_household = X[:, self.room_idx] / X[:, self.households_indx]
        population_per_household = X[:, self.room_idx] / X[:, self.households_indx]
        if self.add_bedrooms_per_room:
            bedrooms_per_household = X[:, self.bedrooms_idx] / X[:, self.households_indx]
            return np.c_[X, rooms_per_household, population_per_household,
                         bedrooms_per_household]
        else:
            return np.c_[X, rooms_per_household, population_per_household]


class MainPrediction():
    def __init__(self):
        self.housing_data = load_housing_data()
        self.linear_regressor = LinearRegression()
        self.decision_tree = DecisionTreeRegressor()
        self.random_forest = RandomForestRegressor()

    def split_data_on_income(self):
        self.housing_data["income_cat"] = pd.cut(self.housing_data["median_income"],
                                    bins=[0, 1.5, 3, 4.5, 6, np.inf],
                                    labels=[1,2,3,4,5])
        split = StratifiedShuffleSplit(n_splits = 1, test_size=0.2,
                                       random_state=42)
        for train_index, test_index in split.split(self.housing_data,
                                                   self.housing_data["income_cat"]):
            strat_train_set = self.housing_data.loc[train_index]
            strat_test_set = self.housing_data.loc[test_index]

        for data_ in (strat_train_set, strat_test_set):
            data_.drop('income_cat', axis=1, inplace = True)

        return strat_train_set, strat_test_set

    # data should be a DataFrame
    def data_cleaning(self, data, num_attribs, cat_attribs):
        num_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy='mean')),
            ("combine_attribs", CombinedAttributesAdder(True)),
            ("scaler", StandardScaler())
        ])
        cat_pipeline = Pipeline([
            ("one_hot", OneHotEncoder())
        ])

        full_pipeline = ColumnTransformer([
            ("num", num_pipeline, num_attribs),
            ("cat", cat_pipeline, cat_attribs)
        ])
        return full_pipeline.fit_transform(data)

    def data_preparation(self):
        # Step 1. Download the data - Do a separate script
        pass

        # Step 2. Gain insights in data: numerical/categorical, shapes, null values,
        # values scaling
        pass

        # Step 3. Do the split train vs test dataset
        self.train_set, self.test_set = self.split_data_on_income()

        # Step 4. Visualizing the data and display correlations -> see if we can
        # add any attributes from available columns
        pass

        # Step 5. Seperate labels/data, do this before cleaning
        self.train_data = self.train_set.drop('median_house_value', axis=1)
        self.train_labels = self.train_set["median_house_value"].copy()

        self.test_data = self.test_set.drop('median_house_value', axis=1)
        self.test_labels = self.test_set["median_house_value"].copy()

        # Step 5. Data Cleaning: + Fill the null values by mean or nearby-column values or 0s
        #                        + Handling the categorical values (one-hot-encoder, ...)
        #                        + Scaling features
        numerical_attributes = list(self.train_data.columns)
        numerical_attributes.remove('ocean_proximity')
        categorical_attributes = ['ocean_proximity']

        self.cleaned_train_data = self.data_cleaning(self.train_data, numerical_attributes,
                                                    categorical_attributes)

    def train_model(self):
        self.linear_regressor.fit(self.cleaned_train_data, self.train_labels)
        lin_mae, lin_rmse = find_metrics(self.linear_regressor, self.cleaned_train_data,
                                         self.train_labels)
        print("Linear Regression MAE: {:.3f} and RMSE: {:.3f}".format(lin_mae, lin_rmse))

        self.decision_tree.fit(self.cleaned_train_data, self.train_labels)
        tree_mae, tree_rmse = find_metrics(self.decision_tree, self.cleaned_train_data,
                                         self.train_labels)
        print("Tree Regression MAE: {:.3f} and RMSE: {:.3f}".format(tree_mae, tree_rmse))

    def train_model_with_cross_validation(self):
        scores = cross_val_score(self.linear_regressor,
                                 self.cleaned_train_data, self.train_labels,
                                 scoring="neg_mean_squared_error", cv=10)
        lin_scores = np.sqrt(-scores)
        display_scores(lin_scores)

        scores = cross_val_score(self.decision_tree,
                                 self.cleaned_train_data, self.train_labels,
                                 scoring="neg_mean_squared_error", cv=10)
        tree_scores = np.sqrt(-scores)
        display_scores(tree_scores)

        scores = cross_val_score(self.random_forest,
                                 self.cleaned_train_data, self.train_labels,
                                 scoring="neg_mean_squared_error", cv=10)
        forest_scores = np.sqrt(-scores)
        display_scores(forest_scores)

    def fine_tune_best_model(self):
        param_grid = [
            {'n_estimators': [3, 10, 20, 30 ], 'max_features':[2,4,6]},
            {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]}
        ]

        grid_search = GridSearchCV(self.random_forest, param_grid,
                                   cv = 10, scoring = 'neg_mean_squared_error',
                                   return_train_score=True)
        grid_search.fit(self.cleaned_train_data, self.train_labels)
        print("Best model: ", grid_search.best_params_)

        cvres = grid_search.cv_results_
        for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
            print(np.sqrt(-mean_score), params)

    def fine_tune_best_model_using_random(self):
        param_random = {'bootstrap': [False, True],
                        'n_estimators': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
                        'max_features': np.arange(3,11)}

        grid_search = RandomizedSearchCV(self.random_forest, param_random,
                                   cv = 10, scoring = 'neg_mean_squared_error',
                                   return_train_score=True, n_iter = 10)
        grid_search.fit(self.cleaned_train_data, self.train_labels)
        print("Best model: ", grid_search.best_params_)

        cvres = grid_search.cv_results_
        for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
            print(np.sqrt(-mean_score), params)

if __name__ == "__main__":
    main_prediction = MainPrediction()
    main_prediction.data_preparation()
    #main_prediction.train_model()
    # after this step, we find randomForest has good mse, we use this to fine-tune
    #main_prediction.train_model_with_cross_validation()
    main_prediction.fine_tune_best_model_using_random()