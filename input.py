import os
import numpy as np


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
    return total_data


def data_clear(data):
    data = np.array(data)
    fea = data[:, :-1]
    labels = data[:, -1]
    fea = fea.astype('float')
    for i in range(len(labels)):
        if labels[i] == 'Y':
            labels[i] = 1
        else:
            labels[i] = 0
    labels = np.array(labels).astype('int')
    return fea, labels
