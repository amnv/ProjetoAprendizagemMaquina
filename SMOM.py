import pandas as pd
import k2nn as knn
import math
from instance import Instance
import random as rd

class SMOM:
    def __init__(self):
        self.outstanding = []
        self.trapped = []

    @staticmethod
    def split_classes(data, minority_class):
        c = data[data.iloc[:, -1] == minority_class]
        not_c = data[data.iloc[:, -1] != minority_class]
        return c, not_c

    @staticmethod
    def nearestK3Instances(xi, sc, k1, k2):
        k3 = max([k1,k2])
        k3neighbors = knn.getNeighbors(sc, xi, k3)
        k1neighbors = knn.getNeighbors(sc, xi, k1)
        k1th = k1neighbors[0]
        
        distance = knn.euclideanDistance(xi, k1th, len(xi)-1)

        return k3neighbors, k1th, distance

    def selection_weigth(self, trapped_instances, w1, w2, r1, r2):
        """ Return a dict of weight for each instance
            @:param trapped_instance list of type instance
        """
        for instance in trapped_instances:
            for neighbor in instance.neighbors:
                neighbor_weight = neighbor.get_list_neighbor()
                if (neighbor in self.outstanding) and (instance in neighbor.get_list_neighbor()):
                    instance.set_selection_weight(neighbor, w1)
                elif (instance in neighbor.get_list_neighbor) and (neighbor_weight != 0):
                    instance.set_selection_weight(neighbor_weight)
                else:
                    smaller_distances = instance.ss(instance)
                    npn = instance.dpn(instance, neighbor, smaller_distances)
                    ma_class, mi_class, minorities_class_set = self.get_classes(trapped_instances)
                    ma = self.get_class_set(trapped_instances, ma_class)
                    mi = self.get_class_set(trapped_instances, mi_class)
                    minorities = self.get_class_set(trapped_instances, minorities_class_set)
                    instance.set_selection_weight(instance.selection_weight_formula(npn, w1, w2, r1, r2, trapped_instances, ma, mi, minorities))


    @staticmethod
    def filterOutstanding(sc, cl):
        outstanding = []
        trapped = []

        for instance in sc:
            if cl[instance] != 0:
                outstanding.append(instance)
            else:
                trapped.append(instance)

        return outstanding, trapped

    def get_classes(self, data):
        # Get majority class
        ma = data.iloc[:, -1].value_counts().idxmax()

        # Get minority class
        mi = data.iloc[:, -1].value_counts().idxmin()

        # Get minority classes set
        mi_aux = data.iloc[:, -1].value_counts() == data.iloc[:, -1].value_counts().min()
        mi_set = mi_aux[mi_aux == True].keys()

        return ma, mi, mi_set

    def get_class_set(self, data, class_name):
        return data[data.iloc[:, -1] == class_name]

    def generate_synthetic_instances(self, sc, g):
        xj = 0
        si = []
        for times in range(g):
            for i in sc:
                if i in self.trapped:
                    xj = i.get_neighbor_high_weight()
                else:
                    xj = rd.choice(i.get_nk1())

                diff = (xj - i)
                gama = [rd.randrange(0, 2) for i in range(len(xj))]
                new_instance = i + diff.dot(gama)

                si.append(new_instance)

        return si

    @staticmethod
    def get_g_for_each_xi(xi, zeta, sc):
        g, remainder = SMOM.get_floor_remainder(zeta, len(sc))

        if xi == sc[-1]: ##
            return g+remainder
        else:
            return g

    @staticmethod
    def get_floor_remainder(zeta, sc_size):
        if (zeta%sc_size) == 0:
            return zeta/sc_size, 0
        else:
            g_for_each_xi = int(math.floor(zeta/sc_size))
            remainder = zeta%sc_size
            return g_for_each_xi, remainder
        

    @staticmethod
    def main():
        """neighbors = [[1,1,1,'a'], [2,2,2,'a'], [3,3,3,'b']]
        instance = [2,3,6,'f']"""

        instances = [Instance(), Instance(), Instance()]
        instances[0]

        # zeta = 16
        # sc = [1,2,3]
        
        # for i in sc:
        #   print(SMOM.get_g_for_each_xi(i, zeta, sc))

        #print(SMOM.nearestK3Instances(instance, neighbors,1,2))


SMOM.main()
