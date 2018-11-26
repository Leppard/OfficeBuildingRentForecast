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

corr_matrix = raw_data.corr()
print(corr_matrix['rent_anjuke'].sort_values(ascending=False))

raw_data.plot(kind="scatter", x="longitude", y="latitude",alpha=0.4,
                label="OFFICE BUILDING",
             c="rent", cmap=plt.get_cmap("jet"), colorbar=True, sharex=False)
plt.legend()
plt.show()


