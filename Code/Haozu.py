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

sql = 'select * from haozu_all_copy'
raw_data = pd.read_sql(sql, con=conn)

raw_data = raw_data.drop("id", axis=1)

corr_matrix = raw_data.corr()
print(corr_matrix['rent'].sort_values(ascending=False))

train_set, test_set= train_test_split(raw_data, test_size=0.1, random_state=42)

# train set
train_features = train_set.drop("rent", axis=1)
train_labels = train_set["rent"].copy()

# test set
test_features = test_set.drop("rent", axis=1)
test_labels = test_set["rent"].copy()


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

# 拼接
full_pipeline = FeatureUnion(transformer_list=[
        ("num_pipeline", num_pipeline),
    ])

train_rent_prepared = full_pipeline.fit_transform(train_features)

test_rent_prepared = full_pipeline.fit_transform(test_features)

# Linear Reg
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(train_rent_prepared, train_labels)
predict = lin_reg.predict(test_rent_prepared)

rmse = np.sqrt(mean_squared_error(predict, test_labels))
print(rmse)

x = range(len(test_labels))
plt.figure()
plt.plot(x, predict, label='PREDICT')
plt.plot(x, test_labels, label='TRUE')
plt.legend(loc='upper right')
plt.show()

