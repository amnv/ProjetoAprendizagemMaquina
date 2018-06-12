import pandas as pd
import k2nn as knn
import math
from instance import Instance


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

    @staticmethod
    def main():
        """neighbors = [[1,1,1,'a'], [2,2,2,'a'], [3,3,3,'b']]
        instance = [2,3,6,'f']"""

        instances = [Instance(), Instance(), Instance()]
        instances[0]


        #print(SMOM.nearestK3Instances(instance, neighbors,1,2))


SMOM.main()
