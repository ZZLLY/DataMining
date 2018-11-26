import csv
import os
import numpy as np
import warnings
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from imblearn.over_sampling import SMOTE
# 网格调参
from sklearn.model_selection import GridSearchCV
# 随机搜索
from sklearn.model_selection import RandomizedSearchCV
# 评价指标
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score

import input
import choose_fea
import scipy.stats
warnings.filterwarnings(module='sklearn*', action='ignore', category=DeprecationWarning)

data_dir = 'result_f1/'

"this is a test for pycharm"
def clf(num, fea, labels):
    record = []
    # 模型列表：决策树、朴素贝叶斯、SVM分类、投票
    dt_clf = DecisionTreeClassifier(class_weight='balanced')
    nb_clf = GaussianNB()
    # 采用网格搜索调参
    # parameters = {'kernel': ('linear', 'rbf'), 'C': [1, 2, 4], 'gamma': [0.125, 0.25, 0.5, 1, 2, 4]}
    # svc = SVC(class_weight='balanced')
    # svm_clf = GridSearchCV(svc, parameters, scoring='f1', pre_dispatch=8)
    # 采用随机搜索调参
    svc = SVC(class_weight='balanced')
    n_iter_search = 20
    parameters = {'C': scipy.stats.expon(scale=100), 'gamma': scipy.stats.expon(scale=.1), 'kernel': ['rbf']}
    svm_clf = RandomizedSearchCV(svc, param_distributions=parameters, scoring='f1', n_iter=n_iter_search)
    # 对上述三种模型投票
    voting_clf = VotingClassifier(estimators=[("dt", dt_clf), ("nb", nb_clf), ("svc", svm_clf)], voting="hard")
    models = [dt_clf, nb_clf, svm_clf, voting_clf]
    kf = KFold(n_splits=5)
    for train_index, test_index in kf.split(fea):
        tmp = []
        x_train, y_train = fea[train_index], labels[train_index]
        x_test, y_test = fea[test_index], labels[test_index]
        # 用smote对训练集数据过采样
        x_train, y_train = SMOTE().fit_sample(x_train, y_train)

        for model in models:
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            tmp.append(f1_score(y_test, y_pred))
            # tmp.append(roc_auc_score(y_test, y_pred))
        record.append(tmp)
    record = np.array(record)
    # num作为标号，方便文件写入记录
    result = [num]
    for i in range(record.shape[1]):
        result.append(np.mean(record[:, i]))
    return result


if __name__ == '__main__':
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    # 读取全部数据集（字典结构）
    total_data = input.input_data('./Data')
    for key in total_data:
        # 分出特征和标签
        total_fea, total_labels = input.data_clear(total_data[key])
        # 单个数据集下特征排序
        fea_rank = choose_fea.choose(key, total_data[key])
        result = []
        for j in range(1, len(fea_rank)):
            # 以Top-j个特征，预测分类
            top_fea = fea_rank[:j]
            x = total_fea[:, top_fea]
            result.append(clf(j, x, total_labels))
            print(key + ' data set:' + str(len(fea_rank)-j)+' jobs need to be done!')
        filename = os.path.join(data_dir, key + '_eval.csv')
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Fea Num", "Decision Tree", "Naive Bayes", "SVM", "Voting"])
            writer.writerows(result)
        print(key + ' data set is written!')
