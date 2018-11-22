import csv
import os
import numpy as np
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
# 过采样
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import KFold

import input
import choose_fea

data_dir = 'result_f1/'


def clf(num, fea, labels):
    # 运用过采样技术解决数据标签的不平衡
    # ros = RandomOverSampler(random_state=0)
    # fea, labels = ros.fit_sample(fea, labels)
    # 模型列表：决策树、朴素贝叶斯、SVM分类
    models = [DecisionTreeClassifier(class_weight='balanced'), GaussianNB(), SVC(class_weight='balanced')]
    # num作为标号，方便文件写入记录
    record = [num]
    # data = np.hstack(fea, labels)
    # kf = KFold(n_splits=2)
    # for train, test in kf.split(data):
    #     print("k折划分：%s %s" % (train.shape, test.shape))
    #     break

    for model in models:
        # cv=10  10折交叉验证
        # scoring也可以用f1，但是这里训练集数量少，当TP==0时，P和R为0，导致f1计算时分母为零，报错(重采样后不报错)
        # 采用roc_auc，避免报错
        record.append(cross_val_score(model, fea, labels, scoring='f1', cv=5, n_jobs=4).mean())
    return record


def data_normal(data):
    data = np.array(data)
    for i in range(data.shape[1] - 1):
        tmp = data[:, i].astype('float')
        std = np.std(tmp, ddof=1)
        mean = np.mean(tmp)
        for j in range(len(data[:, i])):
            data[j][i] = (tmp[j] - mean) / std
    return data


if __name__ == '__main__':
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    # 读取全部数据集（字典结构）
    total_data = input.input_data('./Data')
    for key in total_data:
        # 单个数据集下特征排序
        fea_rank = choose_fea.choose(key, total_data[key])
        # 数据标准化
        data = data_normal(total_data[key])
        result = []
        for j in range(1, len(fea_rank)):
            # 以Top-j个特征，预测分类
            top_fea = fea_rank[:j]
            fea, labels = input.data_clear(data)
            fea = fea[:, top_fea]
            result.append(clf(j, fea, labels))
            print(key + ' data set:' + str(len(fea_rank)-j)+' jobs need to be done!')
        filename = os.path.join(data_dir, key + '_eval.csv')
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Fea Num", "Decision Tree", "Naive Bayes", "SVM"])
            writer.writerows(result)
        print(key + ' data set is written!')
