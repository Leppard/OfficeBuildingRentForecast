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

raw_data = raw_data.drop("id", axis=1).drop("name", axis=1).drop("rent_haozu", axis=1).drop("rent_BC_haozu", axis=1)

labelArr = ['store_nearby', 'mall_nearby',  'park_nearby', 'hotel_nearby', 'metro_nearby']

rents = raw_data["rent"]


from scipy.optimize import leastsq
##需要拟合的函数func :指定函数的形状 k= 0.42116973935 b= -8.28830260655
def func(p,x):
    k,b=p
    return k*x+b

##偏差函数：x,y都是列表:这里的x,y更上面的Xi,Yi中是一一对应的
def error(p,x,y):
    return func(p,x)-y

#k,b的初始值，可以任意设定,经过几次试验，发现p0的值会影响cost的值：Para[1]
p0=[1,0]

# 调整图像的宽高
plt.figure(figsize=(12, 10))
for i in range(0, len(labelArr)):
    dataArr = raw_data[labelArr[i]]

    plt.subplot(2, 3, i+1)
    plt.xlabel(labelArr[i])
    plt.ylabel("rent")
    plt.scatter(dataArr, rents, alpha=0.5)

    Xi = np.array(dataArr)
    Yi = np.array(rents)
    Para = leastsq(error, p0, args=(Xi, Yi))

    # 读取结果
    k, b = Para[0]
    # print("k=", k, "b=", b)
    preg_rents = []
    for x in dataArr:
        preg_rents.append(k * x + b)
    print(mean_squared_error(rents, preg_rents))

    # xmin, xmax = plt.xlim()
    # x = np.linspace(0, xmax, xmax)
    # y = k * x + b
    # plt.plot(x, y, color="red", label="nihe", linewidth=2)

plt.show()