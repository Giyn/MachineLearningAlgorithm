# -*- coding: utf-8 -*-
"""
Created on Sat Jul 18 20:08:04 2020

@author: Giyn
"""

from random import randrange
from data_handle import data_spilt


class RandomForest():

    def __init__(self):
        pass

    # 随机采样
    def get_sample(self, dataSet, ratio):
        subdataSet = []
        subdata_len = round(len(dataSet) * ratio)  # round返回浮点数的四舍五入值
        while len(subdataSet) < subdata_len:
            index = randrange(len(dataSet) - 1)  # 返回序列中随机一个数
            subdataSet.append(dataSet[index])
        return subdataSet


    # 计算分割代价
    def spilt_loss(self, left, right, class_values):
        loss = 0.0
        for class_value in class_values:
            left_size = len(left)
            if left_size != 0:  # 防止除数为零
                prop = [row[-1] for row in left].count(class_value) / float(left_size)
                loss += (prop * (1.0 - prop))
            right_size = len(right)
            if right_size != 0:
                prop = [row[-1] for row in right].count(class_value) / float(right_size)
                loss += (prop*(1.0-prop))
        return loss


    # 选取任意的n个特征，在这n个特征中，选取分割时的最优特征
    def get_best_spilt(self, dataSet, n_features):
        features = []
        class_values = list(set(row[-1] for row in dataSet))
        b_index, b_value, b_loss, b_left, b_right = 999, 999, 999, None, None
        while len(features) < n_features:
            index = randrange(len(dataSet[0]) - 1)
            if index not in features:
                features.append(index)
        for index in features:
            for row in dataSet:
                left, right = data_spilt(dataSet, index, row[index])
                loss = self.spilt_loss(left, right, class_values)
                if loss < b_loss:
                    b_index, b_value, b_loss, b_left, b_right = index, row[index], loss, left, right
        return {'index': b_index, 'value': b_value, 'left': b_left, 'right': b_right}


    # 决定输出标签
    def decide_label(self, data):
        output = [row[-1] for row in data]
        return max(set(output), key=output.count)


    # 不断构建叶节点
    def sub_spilt(self, root, n_features, max_depth, min_size, depth):
        left = root['left']
        right = root['right']
        del(root['left'])
        del(root['right'])
        if not left or not right:
            root['left'] = root['right'] = self.decide_label(left + right)
            return
        if depth > max_depth:
            root['left'] = self.decide_label(left)
            root['right'] = self.decide_label(right)
            return
        if len(left) < min_size:
            root['left'] = self.decide_label(left)
        else:
            root['left'] = self.get_best_spilt(left, n_features)
            self.sub_spilt(root['left'], n_features, max_depth, min_size, depth+1)
        if len(right) < min_size:
            root['right'] = self.decide_label(right)
        else:
            root['right'] = self.get_best_spilt(right, n_features)
            self.sub_spilt(root['right'], n_features, max_depth, min_size, depth+1)


    # 构造决策树
    def build_tree(self, dataSet, n_features, max_depth, min_size):
        root = self.get_best_spilt(dataSet, n_features)
        self.sub_spilt(root, n_features, max_depth, min_size, 1)
        return root


    # 预测测试集结果
    def predict(self, tree, row):
        predictions = []
        if row[tree['index']] < tree['value']:
            if isinstance(tree['left'], dict):
                return self.predict(tree['left'], row)
            else:
                return tree['left']
        else:
            if isinstance(tree['right'], dict):
                return self.predict(tree['right'], row)
            else:
                return tree['right']

    def bagging_predict(self, trees, row):
        predictions = [self.predict(tree, row) for tree in trees]
        return max(set(predictions), key=predictions.count)


    # 创建随机森林
    def RF(self, train, test, ratio, n_feature, max_depth, min_size, n_trees):
        trees = []
        for i in range(n_trees):
            train = self.get_sample(train, ratio)
            tree = self.build_tree(train, n_feature, max_depth, min_size)
            trees.append(tree)
        predict_values = [self.bagging_predict(trees, row) for row in test]
        return predict_values


    # 计算准确率
    def accuracy(self, predict_values, actual):
        correct = 0
        for i in range(len(actual)):
            if actual[i] == predict_values[i]:
                correct += 1
        return correct / float(len(actual))