import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt


def readfile(filename):
    """
    读取数据
    """
    dataset = pd.read_csv(filename)
    attriset = list(dataset.columns)  # 读取属性名
    dataset = list(dataset.values)  # 读取数据
    return dataset, attriset


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
    train_set = dataset[:start] + dataset[end:]
    valid_set = dataset[start:end]
    return train_set, valid_set


def get_attribute(dataset, attriset):
    """
    记录各属性的可能取值
    :param dataset: 数据集
    :param attriset: 属性名集合
    :return: 返回属性字典 key值为属性对应在attriset的下标 value值为对应属性可能取值
    """
    num = len(attriset) - 1
    attribute_dic = {}
    for i in range(num):
        temp = set()
        for line in dataset:
            temp.add(line[i])
        attribute_dic[i] = temp
    return attribute_dic


def get_sub_dataset(dataset, index, label):
    """
    用于计算条件熵 提取和该子属性取值相同的条目
    :param dataset:数据集
    :param index: 下标
    :param attri: 子属性取值
    :return:
    """
    record = []
    for line in dataset:
        if line[index] == attri:
            record.append(line)
    return record


def cal_entropy(subdataset, index):
    """
    用于计算当前数据集的经验熵(index=-1时，相当于传入label)、条件熵
    :param subdataset:前六列为属性值 最后一列为标签值
    :param index: 目标列号
    :return: 熵值
    """
    size = len(subdataset)
    count = {}  # 统计各个属性的数量
    for line in subdataset:
        temp = line[index]
        count[temp] = count.get(temp, 0) + 1

    result = 0.0
    for i in count.values():
        p = float(i) / size
        if p != 0:
            result -= p * math.log2(p)

    return result


def cal_gini(dataset, attri):
    """
    用于计算GINI系数
    :param dataset:数据集
    :param attri:要计算GINI系数的属性的下标
    :return:
    """
    subattri_count = {}  # 记录该特征 的 子属性的 个数
    subatrri_label = {}  # 记录子属性 对应的label的 个数
    total = len(dataset)
    for line in dataset:
        temp = line[attri]
        # 子属性取值的个数统计
        subattri_count[temp] = subattri_count.get(temp, 0) + 1
        # 子属性的标签个数统计
        subatrri_label[temp] = subatrri_label.get(temp, {})
        if line[-1] not in subatrri_label[temp]:
            subatrri_label[temp][line[-1]] = 0
        subatrri_label[temp][line[-1]] += 1

    gini = 0
    for i in subattri_count.keys():
        num = subattri_count[i]  # 该子属性的个数
        gini_temp = 1
        for value in subatrri_label[i].values():
            gini_temp -= np.square(value / num)
        gini += num / total * gini_temp
    return gini


def choose_attribute(dataset, attribute_dict, avail_index, method):
    """

    :param dataset:数据集
    :param attribute_dict:属性取值集合
    :param avail_index: 当前可选的属性的下标 下标号与attribut_dict存放的下标对应
    :param method:建树算法
    :return:最佳属性的下标
    """
    if method == "ID3":
        empirical_entropy = cal_entropy(dataset, -1)  # 计算经验熵
        info_gain = []  # 信息增益列表

        # 遍历可选属性，计算每个属性的条件熵
        for i in avail_index:
            conditional_entopy = 0.0
            for sub_attribute in attribute_dict[i]:  # 遍历该属性的所有可能取值
                sub_dataset = get_sub_dataset(dataset, i, sub_attribute)
                p = len(sub_dataset) / len(dataset)  # 该属性在数据集中的比例
                conditional_entopy += p * cal_entropy(sub_dataset, -1)  # 条件熵计算
            info_gain.append(empirical_entropy - conditional_entopy)
        # 找出信息增益最大的属性
        best_index = np.argmax(info_gain)
        return avail_index[best_index]  # 返回信息增益最大的属性的下标

    elif method == "C4.5":
        empirical_entropy = cal_entropy(dataset, -1)
        info_gain_ratio = []
        for i in avail_index:
            conditional_entopy = 0.0
            for sub_attribute in attribute_dict[i]:  # 遍历该属性的所有可能取值
                sub_dataset = get_sub_dataset(dataset, i, sub_attribute)
                p = len(sub_dataset) / len(dataset)  # 该属性在数据集中的比例
                conditional_entopy += p * cal_entropy(sub_dataset, -1)  # 条件熵计算
            split_info = cal_entropy(dataset, i)
            if split_info == 0:  # 说明这个属性所有取值相同，对决策分裂没有意义
                continue
            info_gain_ratio.append(((empirical_entropy - conditional_entopy) / split_info, i))
        best_index = max(info_gain_ratio, key=lambda x: x[0])
        return best_index[1]  # 返回信息增益率最大的属性的下标

    elif method == "CART":
        gini_list = []
        for i in avail_index:
            gini_temp = cal_gini(dataset, i)
            gini_list.append(gini_temp)
        min_index = np.argmin(gini_list)
        return avail_index[min_index]  # 返回GINI系数最小的属性的下标


def build_tree(dataset, attri_dict, avail_index, parent_lable, method, labelset):
    """
    建树
    :param dataset:数据集
    :param attri_dict:属性取值集合
    :param avail_index:可选属性下标集合
    :param parent_lable:父节点的label
    :param method:建树算法
    :param labelset:属性名称集合
    :return:根节点
    """
    label_list = [record[-1] for record in dataset]
    # 边界条件
    # 若dataset为空集，则取父节点的label
    if len(dataset) == 0:
        return parent_lable
    # 若所有样本的label都相同，直接取label
    if label_list.count(label_list[0]) == len(label_list):
        return label_list[0]
    # 若属性集为空
    if len(avail_index) == 0:
        temp = max(label_list, key=label_list.count)
        return temp

    best_attri = choose_attribute(dataset, attri_dict, avail_index, method)  # 选择最佳属性作为根节点
    avail_index.remove(best_attri)  # 从可选属性中去除该属性
    my_tree = {labelset[best_attri]: {}}  # 以该属性的 名称 为根创建树
    parent_lable = max(label_list, key=label_list.count)  # 获得父节点的label

    # 划分子属性 递归建树
    for sub_attri in attri_dict[best_attri]:
        sub_dataset = get_sub_dataset(dataset, best_attri, sub_attri)
        my_tree[labelset[best_attri]][sub_attri] = build_tree(sub_dataset, attri_dict, avail_index[:], parent_lable,
                                                              method, labelset)
    return my_tree


def predict(line, input_tree, labelset):
    """
    预测数据的label
    :param line: 带预测的一行数据
    :param input_tree: 决策树
    :param labelset: 属性名称集合
    :return: 预测标签
    """
    first_attri = list(input_tree.keys())[0]  # 获得根节点的属性名称
    second_dict = input_tree[first_attri]  # 获得根节点对应子树
    index = labelset.index(first_attri)  # 获得该属性在属性集合中的对应下标
    key = line[index]  # 获得该数据在根属性的取值
    value = second_dict[key]  # 进入该取值对应的子树
    if isinstance(value, dict):  # 若非叶子结点，继续递归
        label = predict(line, value, labelset)
    else:
        label = value

    return label


def cal_accuracy(valid_set, input_tree, labelset):
    """
    计算验证集的准确率
    :param valid_set: 验证集
    :param input_tree: 决策树
    :param labelset: 属性名称集合
    :return: 准确率
    """
    cnt = 0
    for line in valid_set:
        label = predict(line, input_tree, labelset)
        if label == line[-1]:
            cnt += 1
    return cnt / len(valid_set)


def showmax(y):
    y_max = np.argmax(y)
    show_max = '[' + str(y_max+3) + ' ' + str(y[y_max]) + ']'
    plt.plot(y_max+3, y[y_max], 'ko')
    plt.annotate(show_max, xy=(y_max+3, y[y_max]))



if __name__ == "__main__":

    x = []
    ID3_y = []
    C45_y = []
    CART_y = []

    for k in range(3, 10):
        temp = 0
        for i in range(k):
            dataset, labelset = readfile('car_train.csv')
            attribute_dic = get_attribute(dataset, labelset)
            available_attribute = list(range(0, len(labelset) - 1))
            method = "ID3"
            train_set, valid_set = k_fold(dataset, k, i)
            mytree = build_tree(train_set, attribute_dic, available_attribute, -1, method, labelset)
            temp += cal_accuracy(valid_set, mytree, labelset)
        # print("利用%s方法，对数据集进行%s折划分后的准确率为%s" %(method,k,temp/k))
        x.append(k)
        ID3_y.append(temp / k)

    for k in range(3, 10):
        temp = 0
        for i in range(k):
            attribute_dic = get_attribute(dataset, labelset)
            available_attribute = list(range(0, len(labelset) - 1))
            method = "C4.5"
            train_set, valid_set = k_fold(dataset, k, i)
            mytree = build_tree(train_set, attribute_dic, available_attribute, -1, method, labelset)
            temp += cal_accuracy(valid_set, mytree, labelset)
        # print("利用%s方法，对数据集进行%s折划分后的准确率为%s" %(method,k,temp/k))
        C45_y.append(temp / k)

    for k in range(3, 10):
        temp = 0
        for i in range(k):
            dataset, labelset = readfile('car_train.csv')
            attribute_dic = get_attribute(dataset, labelset)
            available_attribute = list(range(0, len(labelset) - 1))
            method = "CART"
            train_set, valid_set = k_fold(dataset, k, i)
            mytree = build_tree(train_set, attribute_dic, available_attribute, -1, method, labelset)
            temp += cal_accuracy(valid_set, mytree, labelset)
        CART_y.append(temp / k)

    plt.plot(x, ID3_y, label='ID3')
    plt.plot(x, C45_y, label='C4.5')
    plt.plot(x, CART_y, label='CART')
    showmax(ID3_y)
    showmax(C45_y)
    showmax(CART_y)
    plt.legend()
    plt.grid(True)
    plt.xlabel('k')
    plt.ylabel('accuracy')
    plt.show()



