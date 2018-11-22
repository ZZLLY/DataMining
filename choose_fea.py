import numpy as np
import math
import os

data_dir = 'rank/'


def choose(key, data):
    if os.path.exists(data_dir + key + '_rank.txt'):
        filename = os.path.join(data_dir, key + '_rank.txt')
        with open(filename, 'r') as f:
            rank = []
            for row in f:
                rank = row.strip().split('\t')
                rank = [int(t) for t in rank]
    else:
        if not os.path.exists(data_dir):
            os.mkdir(data_dir)

        defect = []
        no_defect = []
        # back_defect = []
        # back_no_defect = []
        count1 = 0
        count2 = 0
        data = np.array(data)
        fea_num = np.size(data, 1) - 1
        # # 对原始数据备份，以便后续做特征差值
        # backup = data
        # for row in backup:
        #     if row[-1] == 'Y':
        #         back_defect.append(row[:-1])
        #     else:
        #         back_no_defect.append(row[:-1])
        # back_defect = np.array(back_defect).astype('float')
        # back_no_defect = np.array(back_no_defect).astype('float')

        # 对列进行标准化处理
        data = data.T
        for i in range(np.size(data, 0) - 1):
            temp = data[i].astype('float')
            std = np.std(temp, ddof=1)
            mean = np.mean(temp)
            for j in range(len(data[i])):
                data[i][j] = (temp[j] - mean) / std
        data = data.T
        # 根据行最后一位，区分为有缺陷和无缺陷组
        for row in data:
            if row[-1] == 'Y':
                defect.append(row[:-1])
                count1 += 1
            else:
                no_defect.append(row[:-1])
                count2 += 1

        # 计算不平衡率，向下取整
        k = math.floor(count2 / count1)

        defect = np.array(defect).astype('float')
        no_defect = np.array(no_defect).astype('float')
        # 存放特征权重
        W = np.zeros((1, fea_num))

        for i in range(len(defect)):
            # 一个有缺陷样本特征对所有无缺陷求欧氏距离
            dist = []
            for j in range(len(no_defect)):
                # 欧氏距离计算公式，即两向量差求二范数
                dis = np.linalg.norm(defect[i] - no_defect[j])
                # 将样本编号和距离值对应记录下
                dist.append([j, dis])
            # 根据距离值按升序排序，取出最临近的K个
            dist = np.array(sorted(dist, key=lambda x: x[1]))
            near = dist[:k]
            # K临近无缺陷样本与该有缺陷样本，计算特征差值
            for j in range(len(near)):
                ww = []
                # 取出样本编号
                index = int(near[j][0])
                for jj in range(fea_num):
                    # 将特征编号和差值对应记录下
                    ww.append([jj, abs(defect[i][jj] - no_defect[index][jj])])
                # 根据差值按升序排序
                ww = np.array(sorted(ww, key=lambda x: x[1]))
                # 差值越小，权重越小，权重范围：1-37
                num = 0
                for jj in range(fea_num):
                    # 取出特征编号
                    index = int(ww[jj][0])
                    # 权重累加, 除了第一个，如果当前差值与上一个不同，则权重加1
                    if ww[jj][1] != ww[jj - 1][1] or jj == 0:
                        num += 1
                    W[0][index] += num
        fea_weight = []
        for i in range(fea_num):
            # 将特征编号和权重对应记录下
            fea_weight.append([i, W[0, i]])
        #     # 根据权重按降序排序
        fea_weight = sorted(fea_weight, key=lambda x: x[1], reverse=True)
        rank = []
        for f in fea_weight:
            rank.append(f[0])

        filename = os.path.join(data_dir, key + '_rank.txt')
        with open(filename, 'w') as f:
            for i in rank:
                f.write(str(i) + '\t')
    return rank
