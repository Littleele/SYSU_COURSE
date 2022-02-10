import numpy as np
import csv
import pandas as pd
import matplotlib.pyplot as plt
def read_csv(file_name): # 读取csv文件
    """

    :param file_name: 要读入的csv文件名
    :return:返回单词列表 情感标签列表
    """
    with open(file_name,'r') as f:
        reader=csv.reader(f)

        word=[] # 存储每一行的单词
        feeling=[] # 存储每一行情感标签
        if file_name=="train_set.csv" or file_name=="validation_set.csv":
            for row in reader:
                word.append(row[0])
                feeling.append(row[1])
            del(word[0]) # 由于第一行是表头 要删除
            del(feeling[0])

            return word,feeling
        else:
            for row in reader:
                word.append(row[1])
            del(word[0])
            return word

def count_word(train,validation,test):
    """

    :param train:train单词列表
    :param validation:validation单词列表
    :param test:test单词列表
    :return:返回无重复单词的列表
    """
    wordset=set() # 利用set加快查找速度
    wordlist=[]
    for row in train: #依次处理三个单词列表
        temp=row.split(' ')
        for i in temp:
            if i not in wordset:
                wordset.add(i)
                wordlist.append(i)

    for row in validation:
        temp=row.split(' ')
        for i in temp:
            if i not in wordset:
                wordset.add(i)
                wordlist.append(i)

    for row in test:
        temp=row.split(' ')
        for i in temp:
            if i not in wordset:
                wordset.add(i)
                wordlist.append(i)

    return wordlist

def onehot(givenword, wordlist):
    """

    :param givenword: 要求onehot矩阵的单词列表
    :param wordlist: 所有单词构成的表
    :return: 所求单词表的onehot矩阵
    """
    onehot=np.zeros(shape=(len(givenword), len(wordlist))) # 创建一个二维矩阵
    for index in range(len(givenword)):
        temp=givenword[index].split(' ') # 将一行内的单词分割出来
        for i, word in enumerate(wordlist):
            if word in temp: # 如果单词在该文档内出现 onehot矩阵对应位置为1
                onehot[index][i]=1
    return onehot


def tf_idf(wordlist,train_word,valid_word,test_word):
    """

    :param wordlist: 所有单词构成的表
    :param train_word: 训练集单词
    :param valid_word: 验证集单词
    :param test_word: 测试集单词
    :return: 三个集合的tfidf矩阵
    """
    word_num = len(wordlist) #计算出单词个数
    times=np.zeros(word_num) #创建一个times向量，用来记录一个单词在文档中出现的总次数，用于计算idf
    #times的下标和wordlist中单词的下标对应
    train_tf=np.zeros((len(train_word), word_num)) #创建训练集单词的tf矩阵 规格为文档数x总单词数
    valid_tf=np.zeros((len(valid_word), word_num)) #创建验证集单词的tf矩阵
    test_tf=np.zeros((len(test_word), word_num)) #创建测试集单词的的tf矩阵
    filenum=len(train_word)+len(valid_word)+len(test_word) #计算文件总数
    for i,row in enumerate(train_word):
        temp=row.split(' ') #将一个文档内的的单词分离出来
        num=len(temp) #num为一个文档中的单词数
        for word in temp:
            index=wordlist.index(word) # 获取该单词在wordlist中的下标
            if train_tf[i][index]==0:
                times[index]+=1  # 若该单词在该文档中第一次出现 更新times值
            train_tf[i][index]=(train_tf[i][index]*num+1)/num #对tf值进行更新

    for i,row in enumerate(valid_word):
        temp=row.split(' ')
        num=len(temp)
        for word in temp:
            index=wordlist.index(word)
            if valid_tf[i][index]==0:
                times[index]+=1
            valid_tf[i][index]=(valid_tf[i][index]*num+1)/num

    for i,row in enumerate(test_word):
        temp=row.split(' ')
        num=len(temp)
        for word in temp:
            index=wordlist.index(word)
            if test_tf[i][index]==0:
                times[index]+=1
            test_tf[i][index]=(test_tf[i][index]*num+1)/num

    idf=np.log10(filenum/(times+1)) # 计算得到idf向量
    train_tfidf=idf*train_tf # 计算得到tfidf矩阵
    valid_tfidf=idf*valid_tf
    test_tfidf=idf*test_tf
    return train_tfidf,valid_tfidf,test_tfidf

def cal(row,train_met,n):
    """

    :param row: 一行单词
    :param train_met: 训练集tfidf或onehot矩阵
    :param n: 计算方式 n=1为曼哈顿距离 n=2为欧氏距离
    :return: 记录该行和训练集的距离的列表
    """
    row=row.reshape(1,-1)
    temp=np.repeat(row,train_met.shape[0],axis=0) # 将该行复制 将temp扩充为一个维度和train_met维度相同的矩阵 便于计算

    if n==2: # 计算欧式距离
        temp1=np.square(temp-train_met)
        sum_list=np.sum(temp1,axis=1) #行内求和
        sum_list=np.sqrt(sum_list)

    if n==1: #计算曼哈顿距离
        temp1=np.abs(temp-train_met)
        sum_list=np.sum(temp1,axis=1)
    return sum_list


def knn_predict(valid_met,train_met,train_feeling,k):
    """

    :param valid_met: 待测集矩阵
    :param train_met: 训练集矩阵
    :param train_feeling: 训练集情感标签
    :param k: knn k值选定
    :return: 预测出的情感标签列表
    """
    result=[0 for i in range(valid_met.shape[0])] # 创建大小为待测集文件数的列表
    for index in range(0,valid_met.shape[0]): #遍历待测集矩阵的每一行
        row=valid_met[index]
        sum_list=cal(row,train_met,n=2)
        sum_index=np.argsort(sum_list) # 将sum_list中的元素从小到大排列，提取其对应的index
        dict = {}
        for i in range(k):#取k个样本
            if train_feeling[sum_index[i]] in dict:
                dict[train_feeling[sum_index[i]]]+=1
            else:
                dict[train_feeling[sum_index[i]]]=1
        result[index] = max(dict,key=lambda x: dict[x]) # 求众数

    return result

def cal_accuracy(predict_tag,valid_feeling):
    """

    :param predict_tag: 预测出的情感标签
    :param valid_feeling: 实际的真实情感标签
    :return: 精确度
    """
    total=len(valid_feeling)
    count=0
    for i in range(total):
        if predict_tag[i]==valid_feeling[i]:
            count+=1

    return count/total


def main():
    train_word,train_feeling=read_csv("train_set.csv")
    valid_word,valid_feeling=read_csv("validation_set.csv")
    test_word=read_csv("test_set.csv")
    wordlist=count_word(train_word,valid_word,test_word)
    #valid_onehot=onehot(valid_word,wordlist)
    #train_onehot=onehot(train_word,wordlist

    train_tfidf,valid_tfidf,test_tfidf=tf_idf(wordlist,train_word,valid_word,test_word)
    test_predict = knn_predict(test_tfidf, train_tfidf, train_feeling, k=13)
    test_output = pd.DataFrame({'Words (split by space)': test_word, 'label': test_predict})
    test_output.to_csv('19335262_zhanghangyue_KNN_classification.csv', index=None,encoding='utf8')  # 参数index设为None则输出的文件前面不会再加上行


'''
    for k in range(3,17):
        predict_tag=knn_predict(valid_onehot,train_onehot,train_feeling,k)
        accuracy=cal_accuracy(predict_tag,valid_feeling)
        print('k = ' + str(k) + ', accuracy = ' + str(accuracy))


'''

main()