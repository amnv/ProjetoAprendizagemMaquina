import pandas as pd
import k2nn as knn
import math

class SMOM:
    # def __init__(self, k1, k2):
    #     self.k1 = k1
    #     self.k2 = k2

    def split_classes(self, data, minority_class):
        c = data[data.iloc[:, -1] == minority_class]
        not_c = data[data.iloc[:, -1] != minority_class]
        return c, not_c

    def nearestK3Instances(xi, sc, k1, k2): 
        k3 = max([k1,k2])
        k3neighbors = knn.getNeighbors(sc, xi, k3)
        k1neighbors = knn.getNeighbors(sc, xi, k1)
        k1th = k1neighbors[0]
        
        distance = knn.euclideanDistance(xi, k1th, len(xi)-1)

        return k3neighbors, k1th, distance

    def main():
        neighbors = [[1,1,1,'a'], [2,2,2,'a'], [3,3,3,'b']]
        instance = [2,3,6,'f']

        print(SMOM.nearestK3Instances(instance, neighbors,1,2))

SMOM.main()