import operator
import math
import numpy
from functools import reduce
import pandas as pd
import numpy as np
class Instance:

    def __init__(self, label):
        # tuple key where key is the neighbor and the value is the selection weight
        self.neighbors = {}
        self.nk1 = {}
        self.Fs_Fd = {}
        self.cl = 0
        self.label = label

    def set_fs_fd(self, fs):
        self.Fs_Fd = fs

    def get_fs(self):
        return list(self.Fs_Fd.keys())

    def get_fd(self):
        return list(self.Fs_Fd.values())

    def get_nk1(self):
        return self.nk1.keys()

    def add_neighbor(self, neighbor):
        self.neighbors[neighbor] = 0

    def get_list_neighbor(self):
        return list(self.neighbors.keys())

    def get_neighbor_weight(self, neighbor):
        return self.neighbors[neighbor]

    def get_neighbor_high_weight(self):
        return max(self.neighbors, key=self.neighbors.get)

    def set_selection_weight(self, key, w1):
        self.neighbors[key] = (1 + w1)/math.e

    @staticmethod
    def ss(xi, xj, fs_fd):
        """Step 5.3 article
            :return list of neighbors Fs which the distance is no longer than
            the distance between this instance and the parameter
        """
        threshold = numpy.linalg.norm(xi - xj)
        return list({k: v for k, v in fs_fd.items() if numpy.linalg.norm(xi-v) > threshold}.keys())

    @staticmethod
    def dpn(pri, ari, ss, data):
        """
        Used to find the instances set, NPN , which belong to the PN neighborhood xi.P N(xj)
        Step 5.3 article + supplementary material
        :param pri: primary reference instance (xi)
        :param ari: assistant reference instance (xj)
        :param ss: returned set from function ss
        :return: the Npn: the set of instances in a radius distance from xi and xj
        """

        npn = [pri, ari]
        midaux = tuple(map(operator.add, pri, ari))
        mid = tuple(map(lambda x: operator.truediv(x, 2), midaux))
        dist_pri_ari = reduce(operator.add, pri+ari)

        for xl in ss:
            x = data.iloc[xl]
            dist_mid_xl = numpy.linalg.norm(x - mid) 
            if dist_mid_xl <= dist_pri_ari:
                npn.append(xl)

        return npn
    @staticmethod
    def class_entropy(gama_mi, class_set):
        gama_m = class_set.shape[0]
        ret = 0

        for index, c in class_set.iterrows():
            ret += pd.Series.div((pd.Series.div(c, gama_mi) * pd.Series.div(c, gama_mi).apply(np.log10)), np.log10(gama_m))

        return ret
    
    @staticmethod
    def selection_weight_formula(npn, w1, w2, r1, r2, mc, ma, mi, minorities):
        """formula (2) article
            :param mc class c instances
            :param ma majority class instance
            :param mi minority class instance
            :param minorities minorities classes"""

        gama_ma = ma.shape[0]
        gama_mi = mi.shape[0]
        gama_c = mc.shape[0]

        # Minority classes entropy
        e_mi = Instance.class_entropy(gama_mi, minorities)

        # Majority class entropy
        e_ma = Instance.class_entropy(gama_mi, minorities)

        overgeneralization_factor_aux = r1 * (gama_mi/ gama_c) + r2 * e_mi + w2 * (r1* gama_ma / gama_c + 2 * e_ma)
        overgeneralization_factor = overgeneralization_factor_aux.apply(lambda x: x**2)
        difficulty_factor = math.exp(-1 * (gama_c / gama_c + gama_ma + gama_mi))

        return 1 / (overgeneralization_factor + w1 * difficulty_factor)
