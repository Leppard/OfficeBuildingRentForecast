#!/usr/bin/python
# -*- coding:utf8 -*-

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt
from sklearn.preprocessing import Imputer
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import mean_squared_error

import CombineAttributes as ca
import DataSelector as ds


raw_data = pd.read_csv("housing.csv")
# 生成收入属性的分层，这个分层用来在不同层中抽取样本作为训练集，这样比单纯随机抽样更科学
# 大于5的当作5处理
raw_data["income_cat"] = np.ceil(raw_data["median_income"]/1.5)
raw_data["income_cat"].where(raw_data["income_cat"]<5, 5.0, inplace=True)

split = StratifiedShuffleSplit(n_splits=1, test_size=0.05, random_state=42)
for train_index, test_index in split.split(raw_data, raw_data["income_cat"]):
    strat_train_set = raw_data.loc[train_index]
    strat_test_set = raw_data.loc[test_index]

for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)

# train set
housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()

num_attribs = list(housing.drop("ocean_proximity", axis=1))
cat_attribs = ["ocean_proximity"]

# 'imputer'：      数据填充
# 'attribs_adder'：变换
# 'std_scaler'：   数值型数据的特征缩放
num_pipeline = Pipeline([
        ('selector', ds.DataFrameSelector(num_attribs)),
        ('imputer', Imputer(strategy="median")),
        ('attribs_adder', ca.CombinedAttributesAdder()),
        ('std_scaler', StandardScaler()),
    ])

# 选择
cat_pipeline = Pipeline([
        ('selector', ds.DataFrameSelector(cat_attribs)),
        ('label_binarizer', LabelBinarizer(sparse_output=True)),
    ])

# 拼接
full_pipeline = FeatureUnion(transformer_list=[
        ("num_pipeline", num_pipeline),
        ("cat_pipeline", cat_pipeline)
    ])

housing_prepared = full_pipeline.fit_transform(housing)


# test set
test_housing = strat_test_set.drop("median_house_value", axis=1)
test_housing_labels = strat_test_set["median_house_value"].copy()

test_housing_prepared = full_pipeline.fit_transform(test_housing)

# Linear Reg
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)
predict = lin_reg.predict(test_housing_prepared)

rmse = np.sqrt(mean_squared_error(predict, test_housing_labels))
print(rmse)

x = range(len(test_housing_labels))
plt.figure()
plt.plot(x, predict, label='PREDICT')
plt.plot(x, test_housing_labels, label='TRUE')
plt.legend(loc='upper right')
plt.show()


# DecisionTree
from sklearn.tree import DecisionTreeRegressor
tree_reg=DecisionTreeRegressor()
tree_reg.fit(housing_prepared, housing_labels)
predict = tree_reg.predict(test_housing_prepared)

rmse = np.sqrt(mean_squared_error(predict, test_housing_labels))
print(rmse)


x = range(len(test_housing_labels))
plt.figure()
plt.plot(x, predict, label='PREDICT')
plt.plot(x, test_housing_labels, label='TRUE')
plt.legend(loc='upper right')
plt.show()


# RandomForest
from sklearn.ensemble import RandomForestRegressor
forest_reg = RandomForestRegressor(random_state=42)
forest_reg.fit(housing_prepared, housing_labels)
predict = forest_reg.predict(test_housing_prepared)

rmse = np.sqrt(mean_squared_error(predict, test_housing_labels))
print(rmse)


x = range(len(test_housing_labels))
plt.figure()
plt.plot(x, predict, label='PREDICT')
plt.plot(x, test_housing_labels, label='TRUE')
plt.legend(loc='upper right')
plt.show()








