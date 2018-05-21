import operator
from math import e
import k2nn
from functools import reduce

class Instance:

    def __init__(self):
        # tuple key where key is the neighbor and the value is the selection weight
        self.neighbors = {}

        self.Fs_Fd = {}

    def set_fs_fd(self, fs):
        self.Fs_Fd = fs

    def get_fs(self):
        return list(self.Fs_Fd.keys())

    def get_fd(self):
        return list(self.Fs_Fd.values())

    def add_neighbor(self, neighbor):
        self.neighbors[neighbor] = 0

    def get_list_neighbor(self):
        return list(self.neighbors.keys())

    def get_neighbor_weight(self, neighbor):
        return self.neighbors[neighbor]

    def set_selection_weight(self, key, w1):
        self.neighbors[key] = (1 + w1)/e

    def ss(self, instance):
        """Step 5.3 article
            :return list of neighbors Fs which the distance is no longer than
            the distance between this instance and the parameter
        """
        threshold = k2nn.euclideanDistance(self, instance, len(instance))
        return list({k: v for k, v in self.Fs_Fd.items() if k2nn.euclideanDistance(self, v, len(v)) > threshold}.keys())

    def dpn(self, pri, ari, ss):
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
            dist_mid_xl = k2nn.euclideanDistance(xl, mid, len(xl))
            if dist_mid_xl <= dist_pri_ari:
                npn.append(xl)

        return npn

    # def selection_weight_formula(self, npn, w1, w2, r1, r2):
    #     """formula (2) article"""
    #     overgeneralization_factor =
    #     1()