import os
import numpy as np
from sklearn import preprocessing


def read_arff(file_name):
    output = []
    with open(file_name, 'r') as arff_file:
        for line in arff_file.readlines():
            if not line.startswith('@'):
                if not line.startswith('%'):
                    if line != '\n':
                        t = line.strip().split(',')
                        output.append(t)
    return output


def input_data(path):
    files = os.listdir(path)
    total_data = {}
    for i in range(len(files)):
        key = files[i].split('.')[0]
        total_data[key] = read_arff(path + '/' + files[i])
        total_data[key] = np.array(total_data[key])
    return total_data


def data_clear(data):
    fea = data[:, :-1]
    labels = data[:, -1]
    fea = fea.astype('float')
    min_max_scaler = preprocessing.MinMaxScaler()
    fea = min_max_scaler.fit_transform(fea)

    for i in range(len(labels)):
        if labels[i] == 'Y':
            labels[i] = 1
        else:
            labels[i] = 0
    labels = np.array(labels).astype('float')
    return fea, labels
