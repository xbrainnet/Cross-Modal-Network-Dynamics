import torch
import argparse
from torch.utils.data import random_split
from torch_geometric.data import DataLoader, Data, Dataset
import os.path as osp
from model import Siamese
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, f1_score, recall_score
from sklearn.model_selection import KFold
import numpy as np
import random
import scipy.io


parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=777, help='seed')
parser.add_argument('--batch_size', type=int, default=16, help='batch size')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
parser.add_argument('--lr_step', type=int, default=50, help='learning rate step')
parser.add_argument('--weight_decay', type=float, default=0.00001, help='weight decay')
parser.add_argument('--nhid', type=int, default=128, help='hidden size')
parser.add_argument('--pooling_ratio', type=float, default=0.5, help='pooling ratio')
parser.add_argument('--dropout_ratio', type=float, default=0.8, help='dropout ratio')
parser.add_argument('--epochs', type=int, default=200, help='maximum number of epochs')
parser.add_argument('--patience', type=int, default=50, help='patience for earlystopping')
parser.add_argument('--pooling_layer_type', type=str, default='GCNConv', help='DD/PROTEINS/NCI1/NCI109/Mutagenicity')


# ======================= Data ==============================

def choose_2class_task(X, Y, D, ot, choice):
    '''
    choose a classification task (114 NC, 103 FLE, 89 TLE)
    choice = 1 : NC VS FLE，label 0 VS 1 ，0->0，1->1
    choice = 2 : NC VS TLE，label 0 VS 2 ，0->0，2->1
    choice = 3 : FLE VS TLE，label 0 VS 1.2 ， 2->1
    choice = 4 : NC VS (FLE and TLE)，label 1 VS 2 ，1->0，2->1
    :param X: all re-fMRI time series (n,90,240)
    :param Y: all labels (n,)
    :param D: all DTI  (90,90,n)
    :param choice:
    :return: array
    '''
    # 0 VS 1 ，0->0，1->1
    if choice == 1:
        X = X[0:217]
        Y = Y[0:217]
        D = D[:, :, 0:217]
        ot = ot[0:217]
    # 0 VS 2 ，0->0，2->1
    elif choice == 2:
        X = X.tolist()
        Y = Y.tolist()
        ot = ot.tolist()
        X = X[0:114] + X[217:306]
        Y = Y[0:114] + Y[217:306]
        Y = np.array(Y)
        Y[Y == 2] = 1
        X = np.array(X)
        ot = ot[0:114] + ot[217:306]
        ot = np.array(ot)
        D = D.transpose(2, 0, 1)
        D1 = D[0:114, :, :].tolist()
        D2 = D[217:306, :, :].tolist()
        D = D1 + D2
        D = np.array(D)
        D = D.transpose(1, 2, 0)
    # 0 VS 1.2 ， 2->1
    elif choice == 3:
        Y[Y == 2] = 1
    # 1 VS 2 ，1->0，2->1
    elif choice == 4:
        X = X[114:306]
        Y = Y[114:306]
        ot = ot[114:306]
        D = D[:, :, 114:306]
        Y[Y == 1] = 0
        Y[Y == 2] = 1
    return X, Y, D, ot


class create_Dataset(Dataset):
    '''
    transfer arrays into GNN input by pyg
    edge_feature：the weighted adjacency arrays (n,90,90)
    node_feature：fMRI time series，(n,90,240)
    Y：
    file_name：
    root：the root directory to save pt files
    '''

    def __init__(self, edge_feature, node_feature, Y, file_name, root, transform=None, pre_transform=None):
        self.edge_feature = edge_feature  # (n,90,90)
        self.node_feature = node_feature  # (n,90,num_node_fea)
        A = edge_feature
        A[A > 0] = 1
        self.A = A
        self.graph_label = Y.astype(int)
        self.num_graph = node_feature.shape[0]
        self.num_node = node_feature.shape[1]
        self.num_edge_feature = 1
        self.file_name = file_name
        super(create_Dataset, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        l = []
        for i in range(self.num_graph):
            s = '.\\' + str(self.file_name) + str(i) + '.pt'
            l.append(s)
        return l

    def download(self):
        pass

    def process(self):
        for i in range(self.num_graph):
            source_nodes, target_nodes = np.nonzero(self.A[i, :, :])
            source_nodes = source_nodes.reshape((1, -1))
            target_nodes = target_nodes.reshape((1, -1))
            edge_index = torch.tensor(np.concatenate((source_nodes, target_nodes), axis=0), dtype=torch.long)
            edge_weight = self.edge_feature[i, source_nodes, target_nodes]
            edge_weight = torch.tensor(edge_weight.reshape((-1, self.num_edge_feature)), dtype=torch.float)
            x = torch.tensor(self.node_feature[i, :, :], dtype=torch.float)
            y = torch.tensor([self.graph_label[i]], dtype=torch.long)
            data = Data(x=x, edge_index=edge_index, y=y, edge_attr=edge_weight)
            torch.save(data, osp.join(self.processed_dir, 'graphDataset1_{}.pt'.format(i)))

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, 'graphDataset1_{}.pt'.format(idx)))
        return data


class SiameseDataset(Dataset):
    '''
    package the input of Siamese network
    '''

    def __init__(self, dataset1, dataset2, Y, phase):
        self.set1 = dataset1
        self.set2 = dataset2
        self.Y_set = Y
        self.length = len(self.set1)
        self.phase = phase

    def __len__(self):
        return len(self.set1)

    def len(self):
        return self.__len__()

    def __getitem__(self, idx):
        graph1 = self.set1[idx]
        label1 = graph1.y
        if self.phase == 'train':
            np.random.seed(10)
            idx2 = random.randint(0, self.length - 1)
            graph2 = self.set2[idx2]
            label2 = graph2.y
            label = int(self.Y_set[idx] == self.Y_set[idx2])
            return graph1, label1, graph2, label2, label
        else:
            return graph1, label1

    def get(self, idx):
        return self.__getitem__(idx)


def load_kfold_data(A_dti, A_ot, X_fmri, Y_label, train_index, test_index):
    A_dti_train_val, A_ot_train_val, X_fmri_train_val, Y_train_val = A_dti[train_index], A_ot[train_index], X_fmri[train_index], Y_label[
        train_index]  # array train+val
    A_dti_test, A_ot_test, X_fmri_test, Y_test = A_dti[test_index], A_ot[test_index], X_fmri[test_index], Y_label[test_index]  # array test

    dti_train_val_set = create_Dataset(A_dti_train_val, X_fmri_train_val, Y_train_val, file_name='dti_Dataset', root='./train_val_dti_Dataset')
    dti_test_set = create_Dataset(A_dti_test, X_fmri_test, Y_test, file_name='dti_Dataset', root='./test_dti_Dataset')
    ot_train_val_set = create_Dataset(A_ot_train_val, X_fmri_train_val, Y_train_val, file_name='ot_Dataset', root='./train_val_ot_Dataset')
    ot_test_set = create_Dataset(A_ot_test, X_fmri_test, Y_test, file_name='ot_Dataset', root='./test_ot_Dataset')

    num_train = int(len(train_index) * 0.9)
    num_val = len(train_index) - num_train
    dti_train_set, dti_val_set = random_split(dti_train_val_set, [num_train, num_val])
    ot_train_set, ot_val_set = random_split(ot_train_val_set, [num_train, num_val])
    y_train, y_val = random_split(Y_train_val, [num_train, num_val])

    train_sia_set = SiameseDataset(ot_train_set, dti_train_set, y_train, phase='train')
    valid_sia_set = SiameseDataset(ot_val_set, dti_val_set, y_val, phase='valid')
    test_sia_set = SiameseDataset(ot_test_set, dti_test_set, Y_test, phase='test')

    train_loader = DataLoader(train_sia_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(valid_sia_set, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_sia_set, batch_size=args.batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

# ======================= Data over==============================


def contrastive_loss(output1, output2, label):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """
    MARGIN = 1.0
    euclidean_dis = F.pairwise_distance(output1, output2, keepdim=True)
    loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_dis, 2) +
                                  label * torch.pow(torch.clamp(MARGIN - euclidean_dis, min=0), 2))
    return loss_contrastive


def train(model, optimizer, train_loader, alpha, beta):
    model.train()
    running_loss = 0
    train_acc, train_SEN, train_SPE, train_auc, train_f1 = 0.0, 0.0, 0.0, 0.0, 0.0
    pre, y_true = None, None
    for i, data in enumerate(train_loader):
        input1, label1, input2, label2, label = data
        input1, input2, label1, label2 = input1.cuda(), input2.cuda(), label1.cuda(), label2.cuda()
        label1 = label1.view(-1).cuda()
        label2 = label2.view(-1).cuda()
        label = label.view(-1).cuda()

        output1, output2 = model(input1, input2)
        x1_norm, x1 = output1
        x2_norm, x2 = output2
        predict = torch.softmax(torch.cat((x1, x2)), -1)
        loss1 = torch.nn.CrossEntropyLoss().cuda()(predict, torch.cat((label1, label2)))
        loss2 = contrastive_loss(x1_norm, x2_norm, label)
        loss = alpha * loss1 + beta * loss2

        running_loss += loss.item()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        predict = torch.argmax(predict, dim=-1)
        label_cat = torch.cat((label1, label2))
        label_cat = label_cat.cuda()
        label_cat = label_cat.view(-1)
        if i == 0:
            pre = predict
            y_true = label_cat
        else:
            pre = torch.cat((pre, predict))
            y_true = torch.cat((y_true, label_cat))
    train_loss = running_loss / len(train_loader)
    accuracy = accuracy_score(y_true.cpu(), pre.cpu())
    auc = roc_auc_score(y_true.cpu(), pre.cpu())
    TP = torch.sum(y_true & pre)
    TN = len(y_true) - torch.sum(y_true | pre)
    sample_sum = len(y_true)
    true_sum = torch.sum(y_true)
    neg_sum = len(y_true) - true_sum
    FN = true_sum - TP
    FP = neg_sum - TN
    SEN = TP / true_sum
    SPE = TN / neg_sum
    F1 = (2 * TP) / (sample_sum + TP - TN)
    train_acc += accuracy
    train_SEN += SEN.cpu().numpy()
    train_SPE += SPE.cpu().numpy()
    train_auc += auc
    train_f1 += F1

    return sample_sum, TP, TN, FN, FP, train_loss, train_acc, train_SEN, train_SPE, train_auc, train_f1


def evaluate(model, eva_loader):
    pre_acc, pre_SEN, pre_SPE, pre_auc, pre_f1 = 0.0, 0.0, 0.0, 0.0, 0.0
    pre, y_true = None, None
    loss = 0.0

    model.eval()
    for i, data in enumerate(eva_loader):
        input, label = data
        input, label = input.cuda(), label.view(-1).cuda()
        output1, output2 = model(input, input)
        _, x = output1

        predict = torch.softmax(x, dim=-1)
        loss += torch.nn.CrossEntropyLoss().cuda()(predict, label)
        predict = torch.argmax(predict, dim=-1)
        if i == 0:
            pre = predict
            y_true = label
        else:
            pre = torch.cat((pre, predict))
            y_true = torch.cat((y_true, label))

    accuracy = accuracy_score(y_true.cpu(), pre.cpu())
    auc = roc_auc_score(y_true.cpu(), pre.cpu())
    TP = torch.sum(y_true & pre)
    TN = len(y_true) - torch.sum(y_true | pre)
    sample_sum = len(y_true)
    true_sum = torch.sum(y_true)
    neg_sum = len(y_true) - true_sum
    FN = true_sum - TP
    FP = neg_sum - TN
    SEN = TP / true_sum
    SPE = TN / neg_sum
    F1 = (2 * TP) / (sample_sum + TP - TN)

    pre_loss = loss / len(eva_loader.dataset)
    pre_acc += accuracy
    pre_SEN += SEN.cpu().numpy()
    pre_SPE += SPE.cpu().numpy()
    pre_auc += auc
    pre_f1 += F1
    return sample_sum, TP, TN, FN, FP, pre_loss, pre_acc, pre_SEN, pre_SPE, pre_auc, pre_f1


def adjust_lr(optimizer, epoch):
    learning_rate = args.lr * (0.1 ** (epoch // args.lr_step))
    for parameter in optimizer.param_groups:
        parameter['lr'] = learning_rate


if __name__ == "__main__":
    args = parser.parse_args()
    args.device = 'cpu'
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        args.device = 'cuda:0'
    args.num_classes = 2
    args.num_features = 240

    data_fMRI = scipy.io.loadmat('../data/X_data_gnd.mat')
    f_data = data_fMRI['data']
    Y = data_fMRI['gnd']  # [[0,0,0……]] remember to squeeze
    data_DTI = scipy.io.loadmat('../data/G_all.mat')
    s_data = data_DTI['G']
    data_ot = scipy.io.loadmat('../data/ot_net_PC.mat')
    ot_data = data_ot['ot_net_set0_2']

    X_fmri, Y_label, D_dti, edge_feature_ot = choose_2class_task(f_data, np.squeeze(Y), s_data, ot_data, choice=1)  # X=fmri, Y=label, D=dti. type=array
    edge_feature_dti = D_dti.transpose(2, 0, 1)

    alpha_list = [0.9]
    beta_list = [0.1]
    lr_list = [0.00005]
    weight_decay_list = [0.005, 0.001, 0.0005, 0.0001]
    for alpha in alpha_list:
        for beta in beta_list:
            for lr in lr_list:
                for weight_decay in weight_decay_list:
                    np.random.seed(10)
                    np.random.shuffle(edge_feature_dti)
                    np.random.seed(10)
                    np.random.shuffle(edge_feature_ot)
                    np.random.seed(10)
                    np.random.shuffle(X_fmri)
                    np.random.seed(10)
                    np.random.shuffle(Y_label)
                    kf = KFold(n_splits=10, shuffle=False)
                    current_fold = 0
                    fold_test_acc_list, fold_test_SEN_list, fold_test_SPE_list, fold_test_auc_list, fold_test_f1_list = [], [], [], [], []
                    kfold_test_acc, kfold_test_SEN, kfold_test_SPE, kfold_test_auc, kfold_test_f1 = 0, 0, 0, 0, 0
                    for train_index, test_index in kf.split(edge_feature_dti):
                        current_fold += 1
                        print("=" * 25 + "current fold：" + str(current_fold) + "=" * 25)
                        train_loader, val_loader, test_loader = load_kfold_data(edge_feature_dti, edge_feature_ot, X_fmri, Y_label, train_index, test_index)
                        model = Siamese(args).to(args.device)
                        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
                        max_acc, patience = 0, 0
                        for epoch in range(args.epochs):
                            adjust_lr(optimizer, epoch)
                            _, _, _, _, _, train_loss, train_acc, train_SEN, train_SPE, train_auc, train_f1 = train(model, optimizer, train_loader, alpha, beta)
                            _, _, _, _, _, val_loss, val_acc, val_SEN, val_SPE, val_auc, val_f1 = evaluate(model, val_loader)
                            _, _, _, _, _, test_loss, test_acc, test_SEN, test_SPE, test_auc, test_f1 = evaluate(model, test_loader)
                            print("*" * 10 + "epoch：" + str(epoch) + "*" * 10)
                            print("Train acc:{:.4f}, loss:{:.4f}".format(train_acc, train_loss))
                            print("Valid acc:{:.4f}, loss:{:.4f}".format(val_acc, val_loss))
                            print("Test acc:{:.4f}, loss:{:.4f}, SEN:{:.4f}, SPE:{:.4f}, auc:{:.4f}, f1:{:.4f}".format(test_acc, test_loss, test_SEN, test_SPE, test_auc,
                                                                                                                       test_f1))

                            # # early stop
                            # if val_loss < min_loss:
                            #     min_loss = val_loss
                            #     patience = 0
                            # else:
                            #     patience += 1
                            # if patience > args.patience:
                            #     print("epoch：" + str(epoch) + "  Train loss:{:.4f}, acc:{:.4f}, Valid loss:{:.4f}, acc:{:.4f}, Test loss:{:.4f}, acc:{:.4f}".format(
                            #         train_loss,
                            #         train_acc,
                            #         val_loss, val_acc,
                            #         test_loss,
                            #         test_acc))
                            #     break

                        test_num, test_TP, test_TN, test_FN, test_FP, test_loss, test_acc, test_SEN, test_SPE, test_auc, test_F1 = evaluate(model, test_loader)
                        print("Test acc:{:.4f}, SEN:{:.4f}, SPE:{:.4f}, auc:{:.4f}, F1:{:.4f}".format(test_acc, test_SEN, test_SPE, test_auc, test_F1))
                        fold_test_acc_list.append(test_acc)
                        fold_test_SEN_list.append(test_SEN)
                        fold_test_SPE_list.append(test_SPE)
                        fold_test_auc_list.append(test_auc)
                        fold_test_f1_list.append(test_f1)
                    kfold_test_acc = np.mean(np.array(fold_test_acc_list))
                    kfold_test_SEN = np.mean(np.array(fold_test_SEN_list))
                    kfold_test_SPE = np.mean(np.array(fold_test_SPE_list))
                    kfold_test_auc = np.mean(np.array(fold_test_auc_list))
                    kfold_test_f1 = torch.mean(torch.tensor(fold_test_f1_list))
                    print("=" * 25 + "  alpha:" + str(alpha) + "  beta:" + str(beta) + "  lr:" + str(lr) + "  weight_decay:" + str(
                        weight_decay) + "=" * 25)
                    print("Kfold result:")
                    print("Test acc:{:.4f}, SEN:{:.4f}, SPE:{:.4f}, auc:{:.4f}, f1:{:.4f}".format(kfold_test_acc, kfold_test_SEN, kfold_test_SPE, kfold_test_auc,
                                                                                                  kfold_test_f1))