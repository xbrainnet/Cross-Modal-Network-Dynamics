import scipy.io
import numpy as np
import ot
import FBN.constructFBN_pearson as BN_P
import networkx as nx


def gen_ot_FBN(A, B, C, lamda_skinhorn):
    G0_sinkhorn = ot.bregman.sinkhorn(A, B, C, reg=lamda_skinhorn, verbose=True)
    for i in range(90):
        G0_sinkhorn[i][i] = 0
    return G0_sinkhorn


def gen_ot_FBN_adj(f_data, s_data, conn_ts, lamda_skinhorn=0.1):
    '''
    return：n*90*90
    '''
    ot_EG_set = []
    NUM_USER = f_data.shape[0]
    for i in range(NUM_USER):
        f_matrix = f_data[i, :, :]  # 90*240
        s_matrix = s_data[:, :, i]  # 90*90

        # ot_X : the pg vector of SC
        s_nx = nx.from_numpy_array(s_matrix)
        ot_X = np.array(list(nx.pagerank(s_nx).values()))

        # ot_Y : the pg vector of FC
        EG = BN_P.gen_pearson_FBN(f_matrix, conn_threshold=conn_ts)
        EG_nx = nx.from_numpy_array(EG)
        ot_Y = np.array(list(nx.pagerank(EG_nx).values()))

        from sklearn import preprocessing
        M = s_matrix.copy()
        min_max_scaler = preprocessing.MinMaxScaler()
        M_minMax = min_max_scaler.fit_transform(M)  # normalization
        M_new = np.ones((90, 90)) - M_minMax  # 1 minus each element
        M_new = np.power(M_new, 10)  # rescale

        G0_sinkhorn = gen_ot_FBN(ot_X, ot_Y, M_new, lamda_skinhorn)  # ot matrix 90*90
        ot_EG_set.append(G0_sinkhorn)  # n*90*90
    return np.array(ot_EG_set)

