import pandas as pd
import numpy as np

data = pd.read_csv("adult.csv", sep="\s+", header=None, names=["age","workclass", "fnlwgt", "education", "education_num", "maritial_status",
                                                               "occupation", "relationship", "race", "sex", "capital_gain", "capital_loss",
                                                               "hours_per_week", "native_country", "income"])

data = data.dropna(how='all') # 清除全为空的数据
# data.head()

data.loc[data['income'] == '>50K', 'income'] = 1
data.loc[data['income'] == '>50K.', 'income'] = 1
data.loc[data['income'] == '<=50K', 'income'] = 0
data.loc[data['income'] == '<=50K.', 'income'] = 0