import os
import csv
import math
def read_csv():
    file=open("semeval.txt")
    lines=file.readlines() # 将每行文档内容存入一个列表中
    file_num=len(lines) # 记录文件数目
    record=[] # 创建一个列表 记录每个文档包含的单词数
    dictionary={} # 创建一个空字典 用来统计各单词在文档中出现的总数 和在每个文档中出现的次数

    for line in lines:
        this_line=line.split('\t',2)[2] # 取一行中的单词部分
        this_line.strip('\n') # 去掉末尾换行符
        word_list=this_line.split() # 分割出每一个单词
        record.append(len(word_list)) # 将文档的单词数记录在record中

        for word in word_list: # 遍历这一行包含的单词
            if word not in dictionary: # 如果该单词还未加入字典，即该单词第一次在文档中出现
                temp=[0 for i in range(file_num + 1)] # 创建一个临时列表 插入字典
                temp[0]=1 # 0号单元用于存储该单词在文档中出现的总次数
                temp[lines.index(line)+1]=1 # 更新该单词在该文档中的次数
                dictionary[word]=temp # 插入字典

            else: # 如果该单词已存在
                if dictionary[word][lines.index(line) + 1]== 0: # 若该单词首次在该文档中出现
                    dictionary[word][0]= dictionary[word][0] + 1 # 更新该单词在文档中出现的总次数
                dictionary[word][lines.index(line) + 1]= dictionary[word][lines.index(line) + 1] + 1 # 更新该单词在该文档中的次数


    dict=sorted(dictionary) # 结果要求按字典序排序 对键值进行排序
    return dict,record,dictionary,file_num

def tfidf(dict,file_num,dictionary,record):
    word_num=len(dict) # 得到总单词个数（去除重复）
    result=[[0 for i in range(word_num)] for i in range(file_num)] # 创建存储结果的二维列表

    for i in range(len(dict)): # 遍历单词
        key=dict[i] # 当前单词
        itf=math.log(file_num/(dictionary[key][0]+1),10) # idf计算公式

        for j in range(1, len(dictionary[key])): # 计算该单词在各个文档中的tfidf
            val=dictionary[key][j]
            if val!=0:
                result[j-1][i]=val/record[j-1]*itf # 因为从j=1开始存储第0个文件的信息 所以要j-1

    return result

def write_csv(result):
    file = open("19335262_Zhanghangyue_TFIDF.txt", 'w')
    for i in range(len(result)):
        file.write(str(i + 1) + '\t')  # str函数将数字转成字符串
        for j in range(len(result[i])):
            if result[i][j] != 0: #将结果不为0的写入
                file.write(str(result[i][j]) + ' ')
        file.write('\n')
    file.close()

def main():
    dict,record,dictionary,file_num=read_csv()
    result=tfidf(dict,file_num, dictionary, record)
    write_csv(result)

main()












