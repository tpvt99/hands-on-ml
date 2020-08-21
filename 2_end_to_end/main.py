import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

from config import HOUSING_PATH

def load_housing_data(housing_path = HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)

def split_train_test(data, test_ratio):
    shuffle_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffle_indices[:test_set_size]
    train_indices = shuffle_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]


housing = load_housing_data()
#print(housing.head())
#print(housing.info())
#print(housing['ocean_proximity'].value_counts())
print(housing.value_counts())

#print(housing.describe())

housing.hist(bins=50, figsize=(20,20))
plt.show()

train_set, test_set = split_train_test(housing, 0.2)
print(len(train_set))
print(len(test_set))
