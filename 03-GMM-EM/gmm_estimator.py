import numpy as np

import scipy.cluster.vq as vq
import tqdm

from utils import *


class GMM:
    def __init__(self, num_gaussian=5):
        self.num_gaussian = num_gaussian
        self.data = None
        self.mean = None
        self.var = None
        self.pi = None

    def read_data(self, feature_file):
        self.data = read_all_data(feature_file)

    def kmeans_init(self):
        """
        使用k-mean初始化GMM的均值方差（mean, var）和混合权重pi
        :return:
        """
        mean = []
        var = []
        centroids, labels = vq.kmeans2(self.data, self.num_gaussian, minit="points", iter=100)
        clusters = [[] for i in range(self.num_gaussian)]
        for label, data in zip(labels, self.data):
            clusters[label].append(data)

        # 计算每一类的均值和方差
        for cluster in clusters:
            mean.append(np.mean(cluster, axis=0))
            var.append(np.cov(cluster, rowvar=0))
        pi = np.array([len(c) * 1.0 / len(self.data) for c in clusters])
        self.mean = mean
        self.var = var
        self.pi = pi

    def get_probability(self, x, mean, var):
        dim = x.shape[0]
        var_det = np.linalg.det(var)
        var_inv = np.linalg.inv(var)
        mahalanobis = np.dot(np.transpose(x - mean), var_inv)
        mahalanobis = np.dot(mahalanobis, (x - mean))
        c = 1 / (2 * np.pi) ** (dim / 2)
        return c * var_det ** (-0.5) * np.exp(-0.5 * mahalanobis)

    def get_log_likelihood(self, x):
        n, k = x.shape[0], len(self.pi)
        pdfs = np.zeros((n, k))
        for i in range(n):
            for j in range(k):
                pdfs[i, j] = self.pi[j] * self.get_probability(x[i], self.mean[j], self.var[j])

        return np.sum(np.log(pdfs.sum(axis=1)))

    def E_step(self, x, mean, var, pi):
        """
        更新gamma
        :param x:
        :param mean:
        :param var:
        :param pi:
        :return:
        """
        n, d = x.shape
        gamma = np.zeros((n, self.num_gaussian))

        # 计算每一个样本关于每一个高斯分量的概率
        prob = np.zeros((n, self.num_gaussian))
        for i in range(0, n):
            for j in range(0, self.num_gaussian):
                prob[i, j] = self.get_probability(x[i], mean[j], var[j])

        # 计算样本对每个高斯分量的响应度
        for k in range(self.num_gaussian):
            gamma[:, k] = pi[k] * prob[:, k]

        for i in range(n):
            gamma[i, :] /= np.sum(gamma[i, :])

        return gamma

    def M_step(self, x, gamma):
        """
        更新mean, var, pi
        :param x:
        :param gamma:
        :return:
        """
        N, D = x.shape

        mean = np.zeros((self.num_gaussian, D))
        var = np.zeros((self.num_gaussian, D, D))
        pi = np.zeros(self.num_gaussian)

        for k in range(self.num_gaussian):
            Nk = np.sum(gamma[:, k])
            # 求第k维特征的均值
            for d in range(D):
                mean[k] = np.sum(np.multiply(gamma[:, k], x[:, d])) / Nk

            # 更新方差
            for i in range(N):
                l = np.reshape((x[i] - mean[k]), (D, 1))
                r = np.reshape((x[i] - mean[k]), (1, D))
                var[k] += gamma[i, k] * l * r
            var[k] /= Nk
            pi[k] = Nk / N

        return mean, var, pi

    def update_parameters(self, x, iterations):
        tq = tqdm.tqdm(range(iterations))
        for _ in tq:
            gamma = self.E_step(x, self.mean, self.var, self.pi)
            self.mean, self.var, self.pi = self.M_step(x, gamma)

        log_llh = self.get_log_likelihood(x)
        return log_llh


def train(gmms, num_iterations, feat_file, text_file):
    dict_utt2feat, dict_target2utt = read_feats_and_targets(feat_file, text_file)

    # 针对每一类样本单独训练一个GMM
    for target in targets:
        print("Target: " + target)
        feats = get_feats(target, dict_utt2feat, dict_target2utt)
        _ = gmms[target].update_parameters(feats, num_iterations)
    return gmms


def test(gmms, feat_file, text_file):
    dict_utt2feat, dict_target2utt = read_feats_and_targets(feat_file, text_file)

    correction_num = 0
    error_num = 0
    dict_utt2target = {}
    for target in targets:
        utts = dict_target2utt[target]
        for utt in utts:
            dict_utt2target[utt] = target

    for utt in dict_utt2feat.keys():
        feats = kaldi_io.read_mat(dict_utt2feat[utt])
        scores = []
        # 计算当前样本在每一个GMM上的得分
        for target in targets:
            scores.append(gmms[target].get_log_likelihood(feats))
        predict_target = targets[scores.index(max(scores))]
        if predict_target == dict_utt2target[utt]:
            correction_num += 1
        else:
            error_num += 1

    acc = correction_num * 1.0 / (correction_num + error_num)
    return acc


if __name__ == '__main__':
    num_gussian = 5
    num_iteration = 5

    train_feat_file = "train/feats.scp"
    train_text_file = "train/text"
    test_feat_file = "test/feats.scp"
    test_text_file = "test/text"
    targets = ['Z', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O']

    # 针对每一类创建GMM并初始化
    gmms = {}
    for target in targets:
        gmms[target] = GMM(num_gaussian=num_gussian)
        gmms[target].read_data(train_feat_file)
        gmms[target].kmeans_init()

    gmms = train(gmms, num_iteration, train_feat_file, train_text_file)
    acc = test(gmms, test_feat_file, test_text_file)
    print("ACC: " + str(acc))
    print("ERROR: " + str(1 - acc))
