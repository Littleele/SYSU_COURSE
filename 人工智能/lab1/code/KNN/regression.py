import numpy as np
import csv
import pandas as pd
import  matplotlib.pyplot as plt

def read_csv(file_name):
    """

    :param file_name: 要读入的csv文件名
    :return:返回单词列表 情感标签列表
    """

    with open(file_name,'r') as f:
        reader=csv.reader(f)
        first_row=next(reader)
        feeling_num=len(first_row)-1

        word=[] #存储单词
        feeling=[[] for i in range(feeling_num)] #创建一个情感标签数x文档数的二维列表
        #第0-5行分别代表对应的5个情感

        if file_name=="train_set1.csv" or file_name=="validation_set1.csv":
            for row in reader:
                word.append(row[0])
                for j in range(feeling_num):
                    feeling[j].append(float(row[j+1])) #一个文档的情感标签是一列 注意要转成float 不然无法运算

            return word,feeling
        else:
            for row in reader:
                word.append(row[1])
            return word

def count_word(train,validation,test):
    """

    :param train:train单词列表
    :param validation:validation单词列表
    :param test:test单词列表
    :return:返回无重复单词的列表
    """
    wordset=set()
    wordlist=[]
    for row in train:
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
    onehot=np.zeros(shape=(len(givenword), len(wordlist)))
    for index in range(len(givenword)):
        temp=givenword[index].split(' ')
        for i, word in enumerate(wordlist):
            if word in temp:
                onehot[index][i]=1
    return onehot


def tf_idf(wordlist,train_word,valid_word,test_word):
    word_num = len(wordlist)
    times=np.zeros(word_num) #创建一个times向量，用来记录一个单词在文档中出现的总次数，用于计算idf
    #times的下标和单词列表的下标对应
    train_tf=np.zeros((len(train_word), word_num)) #创建训练集单词的的tf矩阵
    valid_tf=np.zeros((len(valid_word), word_num)) #创建验证集单词的的tf矩阵
    test_tf=np.zeros((len(test_word), word_num)) #创建测试集单词的的tf矩阵
    filenum=len(train_word)+len(valid_word)+len(test_word) #计算文件总数
    for i,row in enumerate(train_word):
        temp=row.split(' ') #将一个文档内的的单词分离出来
        num=len(temp) #num为一个文档中的单词数
        for word in temp:
            index=wordlist.index(word)
            if train_tf[i][index]==0:
                times[index]+=1  #若该单词在该文档中第一次出现 更新idf值
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

    idf=np.log10(filenum/(times+1)) #计算得到idf向量
    train_tfidf=idf*train_tf
    valid_tfidf=idf*valid_tf
    test_tfidf=idf*test_tf
    return train_tfidf,valid_tfidf,test_tfidf

def cal(row,train_met,n):
    row=row.reshape(1,-1)
    temp=np.repeat(row,train_met.shape[0],axis=0)

    if n==2:
        temp1=np.square(temp-train_met)
        sum_list=np.sum(temp1,axis=1)
        sum_list=np.sqrt(sum_list)

    if n==1:
        temp1=np.abs(temp-train_met)
        sum_list=np.sum(temp1,axis=1)
    return sum_list


def knn_predict(valid_met,train_met,train_feeling,k):
    feeling_num=len(train_feeling)
    result=[[0.0 for i in range(valid_met.shape[0])] for j in range(feeling_num)]
    # 为便于后续写入csv文件 将result列表存为情感数x文件数的规格

    for index in range(0,valid_met.shape[0]):
        row=valid_met[index]
        sum_list=cal(row,train_met,n=1)
        sum_index=np.argsort(sum_list)
        total=0
        for i in range(k):
            for j in range(feeling_num):
                if sum_list[sum_index[i]]==0:
                    sum_list[sum_index[i]]=0.001 # 若这两个标签完全相同，则将距离改为0.001 避免分母为0的情况出现
                result[j][index]+=train_feeling[j][sum_index[i]]/float(sum_list[sum_index[i]]) # 根据公式计算概率
                total+=train_feeling[j][sum_index[i]]/float(sum_list[sum_index[i]]) # 计算总和 用于做归一化处理

        for i in range(feeling_num):
            result[i][index]/=total #归一化处理

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

def calculate_cor(valid_real, valid_predict):
    valid_real = np.array(valid_real) # 转成np矩阵
    valid_predict = np.array(valid_predict)
    label_num = valid_predict.shape[0]
    test_num = valid_predict.shape[1]
    correlation = np.corrcoef(valid_real, valid_predict)
    cor_sum = 0
    for i in range(label_num):
        cor_sum += correlation[i][label_num+i]
    cor = 0
    cor = cor_sum / label_num
    return cor

def main():
    train_word,train_feeling=read_csv("train_set1.csv")
    valid_word,valid_feeling=read_csv("validation_set1.csv")
    test_word=read_csv("test_set1.csv")
    wordlist=count_word(train_word,valid_word,test_word)

    valid_onehot=onehot(valid_word,wordlist)
    train_onehot=onehot(train_word,wordlist)
    #train_tfidf,valid_tfidf,test_tfidf=tf_idf(wordlist,train_word,valid_word,test_word)
    for k in range(3,21):
        predict_tag=knn_predict(valid_onehot, train_onehot, train_feeling, k)
        print('k='+str(k)+', coefficient='+str(calculate_cor(valid_feeling,predict_tag)))
    '''
    test_output=pd.DataFrame(
            {'Words (split by space)': valid_word, 'anger': predict_tag[0], 'disgust': predict_tag[1],
             'fear': predict_tag[2], 'joy': predict_tag[3], 'sad': predict_tag[4], 'surprise': predict_tag[5]})
    test_output.to_csv('19335262_Zhanghangyue_KNN_regression.csv',encoding='utf8')  # 参数index设为None则输出的文件前面不会再加上行号
    '''

main()