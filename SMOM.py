import pandas as pd
import k2nn as knn
import math
from instance import Instance


class SMOM:
    def __init__(self):
         self.outstanding  = []
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

    def selection_weigth(self, trapped_instances, w1):
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
                    instance.set_selection_weight()


    @staticmethod
    def main():
        """neighbors = [[1,1,1,'a'], [2,2,2,'a'], [3,3,3,'b']]
        instance = [2,3,6,'f']"""

        instances = [Instance(), Instance(), Instance()]
        instances[0]


        #print(SMOM.nearestK3Instances(instance, neighbors,1,2))

SMOM.main()