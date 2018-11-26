#!/usr/bin/python
# -*- coding:utf8 -*-

import pandas as pd
import numpy as np
import pymysql
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt
from sklearn.preprocessing import Imputer
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder

import CombineAttributes as ca
import DataSelector as ds


conn = pymysql.connect(host='180.169.165.19', \
               user='root', password='Tongji@123456', \
               db='officebuilding', charset='utf8', \
               use_unicode=True)

sql = 'select * from rent_forecast'
raw_data = pd.read_sql(sql, con=conn)

raw_data = raw_data.drop("id", axis=1).drop("name", axis=1)

train_set, test_set= train_test_split(raw_data, test_size=0.1, random_state=42)

# train set
train_features = train_set.drop("rent", axis=1).drop("rent_haozu", axis=1).drop("rent_BC_haozu", axis=1)
train_labels_anjuke = train_set["rent"].copy()
train_labels_haozu = train_set["rent_haozu"].copy()
train_labels_bc_haozu = train_set["rent_BC_haozu"].copy()

# test set
test_features = test_set.drop("rent", axis=1).drop("rent_haozu", axis=1).drop("rent_BC_haozu", axis=1)
test_labels_anjuke = test_set["rent"].copy()
test_labels_haozu = test_set["rent_haozu"].copy()
test_labels_bc_haozu = test_set["rent_BC_haozu"].copy()


feature_attr = list(train_features.drop("type", axis=1).drop("level", axis=1))
type_cat_attr = ["type"]
level_cat_attr = ["level"]


# 不需要数值化，已经在数据库中完成了
feature_attr = list(train_features)

# 'imputer'：      数据填充
# 'attribs_adder'：变换
# 'std_scaler'：   数值型数据的特征缩放
num_pipeline = Pipeline([
        ('selector', ds.DataFrameSelector(feature_attr)),
        ('imputer', Imputer(strategy="median")),
        # ('attribs_adder', ca.CombinedAttributesAdder()),
        ('std_scaler', StandardScaler()),
    ])

# 选择
type_cat_pipeline = Pipeline([
        ('selector', ds.DataFrameSelector(type_cat_attr)),
        ('type_label_binarizer', LabelBinarizer(sparse_output=True)),
    ])

level_cat_pipeline = Pipeline([
        ('selector', ds.DataFrameSelector(level_cat_attr)),
        ('level_label_binarizer', LabelBinarizer(sparse_output=True)),
    ])


# 拼接
full_pipeline = FeatureUnion(transformer_list=[
        ("num_pipeline", num_pipeline),
        # ("type_cat_pipeline", type_cat_pipeline)
        # ("level_cat_pipeline", level_cat_pipeline)
    ])

train_rent_prepared = full_pipeline.fit_transform(train_features)

test_rent_prepared = full_pipeline.fit_transform(test_features)


# DecisionTree
from sklearn.svm import SVR
svr_reg = SVR(kernel='rbf', gamma=0.001, C=100)
svr_reg.fit(train_rent_prepared, train_labels_anjuke)
predict = svr_reg.predict(test_rent_prepared)


################################
from sklearn.model_selection import GridSearchCV
param_grid=[
                    {'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]},
                    {'kernel': ['poly'], 'degree': [x for x in range(1, 9)], 'C': [1, 10, 100, 1000]},
                    ]

grid_search=GridSearchCV(svr_reg, param_grid, cv=5,scoring='neg_mean_squared_error')
grid_search.fit(train_rent_prepared, train_labels_anjuke)
print(grid_search.best_params_)
################################



x = range(len(test_labels_anjuke))
plt.figure()
# plt.bar(x, +predict, color='red', label='Predict Value')
# plt.bar(x, [-y for y in test_labels_anjuke], label='Actual Value')

plt.plot(x, predict, 'r', label='Predict Value')
plt.plot(x, test_labels_anjuke, label='Actual Value')

plt.xlabel('Test Sample')
plt.ylabel('Rent Price')
plt.legend(loc='upper right')
plt.show()


mse = mean_squared_error(predict, test_labels_anjuke)
rmse = np.sqrt(mse)
print("MSE: "+str(mse))
print("RMSE: "+str(rmse))

mae = np.sum(np.absolute(predict-test_labels_anjuke))/len(test_labels_anjuke)
print("MAE: "+str(mae))

from sklearn.metrics import r2_score
r2 = r2_score(test_labels_anjuke, predict)
print("R2: "+str(r2))