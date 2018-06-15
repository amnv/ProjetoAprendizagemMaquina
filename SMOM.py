import pandas as pd
import math
import random as rd
from sklearn.neighbors import NearestNeighbors
import numpy
import pandas as pd

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

        knn3 = NearestNeighbors(n_neighbors=k3)
        knn3.fit(sc.iloc[:, :sc.shape[1]], sc.iloc[:, -1])
        dist, k3neighbors = knn3.kneighbors([xi], return_distance=True)
        
        k3neighbors = k3neighbors.tolist()[0]
        dist = dist.tolist()[0]

        knn1 = NearestNeighbors(n_neighbors=k1)
        knn1.fit(sc.iloc[:, :sc.shape[1]], sc.iloc[:, -1])
        k1neighbors = knn1.kneighbors([xi], return_distance=False).tolist()[0]

        # k3neighbors = knn.getNeighbors(sc, xi, k3)
        # k1neighbors = knn.getNeighbors(sc, xi, k1)        
        k1th = k1neighbors[0]
        distance = numpy.linalg.norm(xi-k1th)
    
        return k3neighbors, dist, k1th, distance, k3

    @staticmethod
    def selection_weigth(data, outstandings, trapped_instances, k3, w1, w2, r1, r2, xi_fs_fd):
        """ Return a dict of weight for each instance
            @:param trapped_instance list of type instance
        """
        knn3 = NearestNeighbors(n_neighbors=k3)

        sw = {}

        # print(trapped_instances)

        for xi in trapped_instances:
            # index = data[data == instance].index
            index = data[data == xi].dropna(axis=0).index[0] 
            # print(index)

            # print(xi_fs_fd[index].keys())

            for xj in xi_fs_fd[index].keys():
                xi_sw = {}
                # print(xi_fs_fd[xj])
                # lista = list(filter(lambda x: len(x) > 0, xi_fs_fd.values()))  
                
                vis_xj = xi_fs_fd.get(xj, {})
                print(xi)
                if (xj in outstandings) and (xi in xi_fs_fd[xj].keys()):
                    xi_sw[xj] = 1 + w1/math.e              
                    # print(neighbor)
                #elif (xj in xi_fs_fd):
                #     bol = xi in xi_fs_fd[xj].keys() 
                #     print(bol)
                
                #elif (len(vis_xj) > 0) and (vis_xj.get(index, None) != None):
                elif (xj in xi_fs_fd.keys()) and (index in xi_fs_fd[xj].keys()) and (xj in sw) and (index in sw[xj]):
                    print("foi")
                    sw[index][xj] = sw[xj][index]
        print sw
        return 0
                # if (neighbor in self.outstanding) and (instance in neighbor.get_list_neighbor()):
                #     instance.set_selection_weight(neighbor, w1)
                # elif (instance in neighbor.get_list_neighbor) and (neighbor_weight != 0):
                #     instance.set_selection_weight(neighbor_weight)
                # else:
                #     smaller_distances = instance.ss(neighbor)
                #     npn = instance.dpn(instance, neighbor, smaller_distances)
                #     ma_class, mi_class, minorities_class_set = self.get_classes(trapped_instances)
                #     ma = self.get_class_set(trapped_instances, ma_class)
                #     mi = self.get_class_set(trapped_instances, mi_class)
                #     minorities = self.get_class_set(trapped_instances, minorities_class_set)
                #     instance.set_selection_weight(instance.selection_weight_formula(npn, w1, w2, r1, r2, trapped_instances, ma, mi, minorities))
                # weightDict[instance][neighbor] = instance.get_neighbor_weight(neighbor)
        
        # return weightDict


    @staticmethod
    def filterOutstanding(sc, cl, minor):
        outstanding = []
        trapped = []
        
        for instanceIndex in range(sc[sc.iloc[:, -1] == 16].shape[0]):
            #print(instanceIndex)
            instance = sc.iloc[instanceIndex]
            if cl[instanceIndex] != 0:
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
    def nearEnemy(xi, others, k3, distance, neighbors, neighbors_distance):
        knn3 = NearestNeighbors(n_neighbors=k3)
        knn3.fit(others.iloc[:, :others.shape[1]], others.iloc[:, -1])
        dist, k3neighbors = knn3.kneighbors([xi], return_distance=True)
        
        k3neighbors = k3neighbors.tolist()[0]
        dist = dist.tolist()[0]

        fs = []
        fd = []
        for i in range(len(dist)):
            if dist[i] < distance:
                fs.append(k3neighbors[i])
                fd.append(dist[i])

        fs = fs + neighbors
        fd = fd + neighbors_distance
        
        return k3neighbors, fs, fd        
        
    @staticmethod
    def k2_neighbors(data, k3_neighbor, k3_enemy, xi, k2, k3):
        frames = numpy.concatenate([k3_neighbor, k3_enemy])
        #print(frames)
        knn = NearestNeighbors(n_neighbors=k2)
        knn.fit(data.iloc[frames, :len(frames)], data.iloc[frames, -1])
        return knn.kneighbors([xi], return_distance=False).tolist()[0] 