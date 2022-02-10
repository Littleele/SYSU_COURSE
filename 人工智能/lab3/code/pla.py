import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt


def k_fold(dataset, k, i):
    """
    划分训练集和验证集
    :param dataset: 数据集
    :param k: 将数据集分为k个子集
    :param i: 选取第i个子集作为验证集
    :return: 划分出的训练集和验证集
    """
    total = len(dataset)
    step = total // k  # 每一步的步长 向下取整
    start = i * step
    end = start + step
    train_set = np.vstack((dataset[:start], dataset[end:]))
    valid_set = dataset[start:end]
    return train_set, valid_set


def read_csv(filename):
    dataset = pd.read_csv(filename, header=None)
    dataset = dataset.values
    return dataset


def split(dataset):
    """
    分割出特征集和标签
    """

    data = []
    label = []
    for row in dataset:
        if row[-1] == 0:
            label.append(-1)  # 给定数据集的label取值为0和1 这里将0标签转换为-1
        else:
            label.append(1)
        data.append(row[0:39])  # 前40列为特征
    # 将特征集和标签集转为array，加快后续的运算
    data_mat = np.array(data)
    label_mat = np.array(label)
    return data_mat, label_mat


def cal_wrong(w, b, data_mat, label):
    """
    pocket_pla 计算当前该点的所犯错误数
    """
    count = 0
    for i in range(len(data_mat)):
        judge = label[j] * (np.dot(w, data_mat[j]) + b)
        if judge <= 0:
            count += 1
    return count


def pla(data_mat, label, iteration, rate):
    """
    该pla方法为每次选择第一个误分类的点，进行更新
    :param data_mat:特征集合
    :param label:label集合
    :param iteration:最大迭代次数
    :param rate:学习率
    :return:训练出的w和b
    """
    w = np.zeros(len(data_mat[0]))  # 初始化w为0向量
    b = 0  # 初始化b为0

    for i in range(iteration):  # 最大迭代次数
        flag = 1  # 标记位 标识是否有误分类的点
        for j in range(len(data_mat)):  # 遍历所有样本点
            judge = label[j] * (np.dot(w, data_mat[j]) + b)  # 判断该点是否被误分类
            if judge <= 0:  # 一点发现该点误分类，就立即对w和b进行更新
                w = w + rate * label[j] * data_mat[j]
                b = b + rate * label[j]
                flag = 0
                break
        if flag == 1:  # 若没有误分类的点
            break

    return w, b


def pla2(data_mat, label, iteration, rate):
    """
    该pla方法为每次从所有误分类的点中随机选取一个，进行更新
    """
    w = np.zeros(len(data_mat[0]))
    b = 0
    for i in range(iteration):
        f = (np.dot(data_mat, w.T) + b) * label  # 矩阵运算，直接得到所有点的分类结果
        # 获取误分类点的位置索引
        idx = np.where(f <= 0)  # idx为一个元组 第一个元素为误分类点的行索引 第二个元素为列索引
        num = f[idx].size
        if num != 0:  # 若有误分类点
            point = np.random.randint((f[idx].shape[0]))  # 随机挑选一个误分类点
            # idx[0][point]为所取点的行索引
            temp_x = data_mat[idx[0][point]]  # 取出该行对应的x
            temp_y = label[idx[0][point]]  # 取出该行所对的label
            w = w + rate * temp_x * temp_y
            b = b + rate * temp_y
        else:
            break
    return w, b


def pocket_pla(data_mat, label, iteration, rate):
    """

    """
    w = np.zeros(len(data_mat[0]))  # 初始化w为0向量
    b = 0  # 初始化b为0
    best_wrong = len(data_mat) + 1
    i = 0

    while i < iteration:
        f = (np.dot(data_mat, w.T) + b) * label  # 矩阵运算，直接得到所有点的分类结果
        # 获取误分类点的位置索引
        idx = np.where(f <= 0)  # idx为一个元组 第一个元素为误分类点的行索引 第二个元素为列索引
        num = f[idx].size
        if num != 0:  # 若有误分类点
            point = np.random.randint((f[idx].shape[0]))  # 随机挑选一个误分类点
            temp_x = data_mat[idx[0][point]]  # 取出该行对应的x
            temp_y = label[idx[0][point]]  # 取出该行所对的label
            temp_w = w + rate * temp_x * temp_y
            temp_b = b + rate * temp_y
            temp_wrong = cal_wrong(temp_w, temp_b, data_mat, label)
            if temp_wrong < best_wrong:
                best_wrong = temp_wrong
                w = temp_w
                b = temp_b
            i += 1
        else:break

    return w, b


def predict(w, b, valid_set, label):
    """
    将训练得到的w和b带入验证集 检测正确率
    """
    total = len(valid_set)
    cnt = 0
    for i in range(total):
        temp = np.dot(valid_set[i], w) + b
        if label[i] == np.sign(temp) or temp == 0:  # 要注意当temp=0时也认为是预测正确
            cnt += 1
    return cnt / total


if __name__ == "__main__":
    dataset = read_csv("train.csv")
    x = []
    y = []
    k = 10
    j = 1


    while j <= 100:
        temp = 0
        for i in range(k):
            train_set, valid_set = k_fold(dataset, k, i)
            train_met, train_label = split(train_set)
            valid_met, valid_label = split(valid_set)
            w, b = pla2(train_met, train_label, j, 1)
            temp += predict(w, b, valid_met, valid_label)
        print("PLA在迭代次数为%s时，对数据集进行%s折划分后的准确率为%s" % (j, k, temp / k))
        x.append(j)
        y.append(temp / k)
        j += 1

    plt.plot(x, y)
    plt.title("pocket pla")
    plt.grid(True)
    plt.xlabel('Iteration times')
    plt.ylabel('Accuracy')
    plt.show()
