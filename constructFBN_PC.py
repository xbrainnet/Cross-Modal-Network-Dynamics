import numpy as np


def cal_pearson_value(A, B):
    '''
    calculate the PC coefficient of vector A and B
    :param A:
    :param B:
    :return:
    '''
    X = np.vstack([A, B])
    pearson_value = np.corrcoef(X)[0][1]
    return pearson_value


def gen_pearson_FBN(raw_data, conn_threshold):
    '''
    calculate the PC coefficient matrix of each brain region pair
    :param raw_data: 90*235
    :return: FBN_pearson array 90*90
    '''
    num_region = raw_data.shape[0]
    FBN_pearson = []
    for i in range(0, num_region):
        temp = []
        for j in range(0, num_region):
            if i == j:
                p = 0
            else:
                p = cal_pearson_value(raw_data[i], raw_data[j])
                if p < 0:
                    p = -p
                if p < conn_threshold:
                    p = 0
            temp.append(p)
        FBN_pearson.append(temp)
    return np.array(FBN_pearson)
