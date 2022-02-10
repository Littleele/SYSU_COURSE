import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv


def read_csv(filename):
    dataset = pd.read_csv(filename, header=None)
    dataset = dataset.values
    return dataset


def split(dataset):
    data = []
    label = []
    for row in dataset:
        label.append(row[-1])
        data.append(row[0:39])
    data_mat = np.array(data)
    label_mat = np.array(label)
    return data_mat, label_mat


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


def sigmod(w, x):
    """
    logistic函数
    :param w:
    :param x:
    :return:
    """
    temp = w.dot(x)
    #防止溢出
    if temp >= 0:
        temp = 1.0 / (1 + np.exp(-1 * temp))
    else:
        temp = np.exp(temp) / (1 + np.exp(temp))

    return temp


def gradient(data_mat, label, iteration, rate):
    """
    计算梯度下降，进行w和b的参数更新
    :param data_mat:特征集合
    :param label:标签集合
    :param iteration:最大迭代次数
    :param rate:学习率
    :return:返回训练的w和b
    """
    w = np.zeros(len(data_mat[0]) + 1)  # 多拓展一维
    diff = 1e-3
    data_temp = np.insert(data_mat, data_mat.shape[1], 1, axis=1) #x多拓展一维
    for i in range(iteration):
        sum = 0
        for j in range(len(data_temp)):
            sum += (label[j]-sigmod(w, data_temp[j])) * data_temp[j]
        w_new = w + rate * sum
        diff1 = np.linalg.norm(w_new - w)  # 求两者的欧式距离
        if (diff1 <= diff):
            print("梯度下降收敛")
            break
        w = w_new
    return w


def predict(valid_mat, lable, w):
    total = len(valid_mat)
    cnt = 0
    valid_temp = np.insert(valid_mat, valid_mat.shape[1], 1, axis=1)
    for i in range(len(valid_temp)):
        temp=sigmod(w,valid_temp[i])
        if(temp>=0.5 and lable[i]==1) or (temp<0.5 and lable[i]==0):
            cnt+=1
    return cnt / total



if __name__ == "__main__":
    dataset = read_csv("train.csv")



    x=[]
    y=[]
    k=10
    j=1


    while j<100:
        temp = 0
        for i in range(k):
            train_set, valid_set = k_fold(dataset, k, i)
            train_met, train_label = split(train_set)
            valid_met, valid_label = split(valid_set)
            w = gradient(train_met, train_label, j, 0.0001)
            temp += predict(valid_met, valid_label, w)
        print("LR在迭代次数为%s时，对数据集进行%s折划分后的准确率为%s" % (j, k, temp / k))
        x.append(j)
        y.append(temp/k)
        j+=1

    plt.title("learning rate=0.000001")
    plt.plot(x,y)
    plt.grid(True)

    plt.xlabel('Iteration times')
    plt.ylabel('Accuracy')

    plt.show()
