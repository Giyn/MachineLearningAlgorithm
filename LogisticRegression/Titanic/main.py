import pandas as pd
import numpy as np
import re
from ML.model_selection import train_test_split
from ML.LogisticRegression import LogisticRegression


titanic = pd.read_csv("Titanic.csv")

# print(titanic.describe()) # 按列统计特征
titanic['Age'] = titanic['Age'].fillna(titanic['Age'].mean()) # 缺失值填充
# print(titanic.describe())

# print(titanic['Sex'].unique()) # 查看Sex特征有哪些值

# loc定位到目标行，对Sex特征进行独热编码
titanic.loc[titanic['Sex'] == 'male', 'Sex'] = 0 # 令Sex等于male那行的Sex值为1
titanic.loc[titanic['Sex'] == 'female', 'Sex'] = 1 # 令Sex等于female那行的Sex值为0

# print(titanic['Embarked'].unique()) # 查看有哪些值
titanic['Embarked'] = titanic['Embarked'].fillna('S') # S数量多，可以用S补充缺失值
# 独热编码
titanic.loc[titanic['Embarked'] == 'S', "Embarked"] = 0
titanic.loc[titanic['Embarked'] == 'C', "Embarked"] = 1
titanic.loc[titanic['Embarked'] == 'Q', "Embarked"] = 2

# 构造特征：亲属数量
titanic["FamilySize"] = titanic["SibSp"] + titanic["Parch"]
# 构造特征：名字长度
titanic["NameLength"] = titanic["Name"].apply(lambda x:len(x))


# 特征工程
def get_title(name):
    title_search = re.search('([A-Za-z]+)\.', name) # \.匹配.(转义)
    if title_search:
        return title_search.group(1)
    return ""

titles = titanic["Name"].apply(get_title)
# print(pd.value_counts(titles))

title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Dr": 5, "Rev": 6, "Major": 7, "Col": 7, "Mlle": 8,
                 "Mme": 8, "Don": 9, "Lady": 10, "Countess": 10, "Jonkheer": 10, "Sir": 9, "Capt": 7, "Ms": 2}
for k, v in title_mapping.items():
    titles[titles == k] = v

# print(pd.value_counts(titles))

titanic["Title"] = titles

titanic = titanic.drop(['Name', 'Ticket', 'Cabin'] ,axis=1)
predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"] # 输入逻辑回归算法的特征
# 转换为数组类型
X = np.array(titanic[predictors])
y = np.array(titanic["Survived"])

X_train, X_test, y_train, y_test = train_test_split(X, y, seed=412) # 分割数据集

log_reg = LogisticRegression()
log_reg.fit(X_train, y_train) # 训练模型

print("准确率为:", "{:.3%}".format(log_reg.score(X_test, y_test))) # 预测准确率