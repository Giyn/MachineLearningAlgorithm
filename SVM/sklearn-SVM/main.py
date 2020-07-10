import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn import svm  # sklearn自带SVM分类器
from sklearn import datasets # 导入数据集
from sklearn.model_selection import train_test_split

cm_light = mpl.colors.ListedColormap(['#A0FFA0', '#FFA0A0', '#A0A0FF'])
cm_dark = mpl.colors.ListedColormap(['g', 'r', 'b'])


# 导入数据集
iris = datasets.load_iris()
X = iris.data
Y = iris.target
X = X[:, 0:2] # 取前两列特征向量，用来作二特征分类
x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size = 0.8, random_state = 1)


# SVM 分类器
clf = svm.SVC(C = 0.8, kernel = 'rbf', gamma = 20, decision_function_shape = 'ovo')
clf.fit(x_train, y_train.ravel())


plt.scatter(X[:, 0], X[:, 1], c = Y, edgecolors = 'k', s = 50, cmap = cm_dark)  # 样本
plt.xlabel('Length')
plt.ylabel('Width')
plt.title('Iris SVM classifier')
plt.show()


# 计算准确率
print("训练集准确率: %f" %(clf.score(x_train, y_train))) # 训练集准确率
print("测试集准确率: %f" %(clf.score(x_test, y_test))) # 测试集准确率