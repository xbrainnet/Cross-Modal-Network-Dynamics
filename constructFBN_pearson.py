import scipy.io
import numpy as np
import math


def cal_pearson_value(A, B):
    '''
    计算A B向量的皮尔森相关系数
    :param A:
    :param B:
    :return:
    '''
    # input array
    X = np.vstack([A, B])
    pearson_value = np.corrcoef(X)[0][1]
    # pearson_matrix = np.corrcoef(A, B)
    # pearson_value = pearson_matrix[0][1]
    return pearson_value


# 皮尔森相关 系数构造脑网络
def gen_pearson_FBN(raw_data, conn_threshold):
    '''
    计算两两脑区的皮尔森相关系数矩阵
    :param raw_data: 90*235
    :return: FBN_pearson array 90*90
    '''
    num_region = raw_data.shape[0]  # number of brain regions
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


def gen_p_FBN_set(f_data, conn_ts):
    '''
    生成svm的输入，n*8100（不过这里没考虑上三角，会有冗余）
    :param f_data:
    :param conn_ts:
    :return:
    '''
    p_EG_set = []
    num_user = f_data.shape[0]  # number of  participants
    num_region = f_data.shape[1]  # number of brain regions
    for i in range(num_user):
        f_matrix = f_data[i, :, :]  # 90*235
        EG = gen_pearson_FBN(f_matrix, conn_threshold=conn_ts)  # 90*90
        # EG_half_temp = []
        # for j in range(num_region):
        #     EG_half_temp.append(EG[j][j:num_region])  # 90*90的对角矩阵 list
        # EG_half = [n for a in EG_half_temp for n in a]  # 对角矩阵拉成一维
        # p_EG_set.append(EG_half)  # n*4050
        EG_temp = EG.flatten()
        p_EG_set.append(EG_temp)  # n*8100
    p_EG_set = np.array(p_EG_set)
    return p_EG_set


def gen_p_FBN_adj(f_data, conn_ts):
    '''
    生成GCN的输入adj矩阵，和上面的函数比最后不要拉成一维
    :param f_data:
    :param conn_ts:
    :return: n*90*90
    '''
    p_EG_set = []
    num_user = f_data.shape[0]  # number of  participants
    num_region = f_data.shape[1]  # number of brain regions
    for i in range(num_user):
        f_matrix = f_data[i, :, :]  # 90*235
        EG = gen_pearson_FBN(f_matrix, conn_threshold=conn_ts)  # 90*90
        p_EG_set.append(EG)  # n*8100
    p_EG_set = np.array(p_EG_set)
    return p_EG_set


def gen_dti_set(dti_net_set):
    dti_vector_set = []
    num_user = dti_net_set.shape[0]
    for i in range(num_user):
        net = dti_net_set[i]  # 90*90
        net = net.flatten()     # 8100
        dti_vector_set.append(net)
    return np.array(dti_vector_set)     # n*8100


if __name__ == '__main__':
    data_fMRI = scipy.io.loadmat('../data/X_data_gnd.mat')
    f_data = data_fMRI['data']
    Y = data_fMRI['gnd']  # [[0,0,0……]] 记得squeeze
    data_DTI = scipy.io.loadmat('../data/G_all.mat')
    s_data = data_DTI['G']
    # X_fmri, Y_label, D_dti = choose_2class_task(f_data, np.squeeze(Y), s_data, choice=4)  # X=fmri, Y=label, D=dti, 都是array
    Y_label = np.squeeze(Y)

    # 所有受试的pearson
    pearson_net_set0_2 = gen_p_FBN_adj(f_data, conn_ts=0.2)  # [n,90,90]
    mat_path1 = '../data/pearson_net_set0_2.mat'
    scipy.io.savemat(mat_path1, {'pearson_net_set0_2': pearson_net_set0_2})

    # pearson_vector_set = gen_p_FBN_set(f_data, conn_ts=0)  # [n,8100]
    # mat_path1 = './pearson_vector_set.mat'
    # scipy.io.savemat(mat_path1, {'pearson_vector_set': pearson_vector_set})

    # # 所有受试的DTI
    # dti_net_set = s_data.transpose(2, 0, 1)  # [n,90,90]
    # # math_path2 = './dti_net_set.mat'
    # # scipy.io.savemat(math_path2, {'dti_net_set': dti_net_set})
    # dti_vector_set = gen_dti_set(dti_net_set)
    # mat_path = '../data/dti_vector_set.mat'
    # scipy.io.savemat(mat_path, {'dti_vector_set': dti_vector_set})