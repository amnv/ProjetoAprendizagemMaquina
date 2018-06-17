import pandas as pd
import math
import random as rd
from sklearn.neighbors import NearestNeighbors
import numpy
import pandas as pd
from instance import Instance
import operator
from functools import reduce

class SMOM:
    def __init__(self):
        self.outstanding = []
        self.trapped = []

    @staticmethod
    def kneigbor(data, xi, k, return_distance):
        knn = NearestNeighbors(n_neighbors=k)
        knn.fit(data.iloc[:, :data.shape[1]], data.iloc[:, -1])
        return knn.kneighbors([xi], return_distance=return_distance)
        


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
    def selection_weigth(data, outstandings, trapped_instances, k3, w1, w2, r1, r2, xi_fs_fd, minor):
        """ Return a dict of weight for each instance
            @:param trapped_instance list of type instance
        """
        knn3 = NearestNeighbors(n_neighbors=k3)

        sw = {}

        for xi in trapped_instances:
            index = data[data == xi].dropna(axis=0).index[0] 

            for xj in xi_fs_fd[index].keys():
                xi_sw = {}

                vis_xj = xi_fs_fd.get(xj, {})
                # print(xi)
                if (xj in outstandings) and (xi in xi_fs_fd[xj].keys()):
                    xi_sw[xj] = 1 + w1/math.e              
                elif (xj in xi_fs_fd.keys()) and (index in xi_fs_fd[xj].keys()) and (xj in sw.keys()) and (index in sw[xj].keys()):
                    print("foi")
                    sw[index][xj] = sw[xj][index]
                else:    
                    v_xj = data.iloc[xj]
                    smaller_distances = Instance.ss(xi, v_xj, xi_fs_fd[index]) 
                    npn = Instance.dpn(xi, v_xj, smaller_distances, data)
                    
                    vals = data[data.iloc[:, -1] == minor]
                    ma_class, mi_class, minorities_class_set = SMOM.get_classes(data)
                    ma = SMOM.get_class_set(data, ma_class)
                    mi = SMOM.get_class_set(data, minor)

                    minorities = data[(data.iloc[:, -1] == 1) | (data.iloc[:, -1] == 25) | (data.iloc[:, -1] == 2) | (data.iloc[:, -1] == 26) | (data.iloc[:, -1] == 29)]
                    # minorities_class_set = {}
                    ret = Instance.selection_weight_formula(npn, w1, w2, r1, r2, vals, ma, mi, minorities)
                    #print(ret)
                    xi_sw[xj] = ret
               
            sw[index] = xi_sw
            print("foi aqui")
        
        return sw


    @staticmethod
    def filterOutstanding(sc, cl, minor):
        outstanding = []
        trapped = []
        index_trapped = []
        for instanceIndex in range(sc[sc.iloc[:, -1] == minor].shape[0]):
            #print(instanceIndex)
            instance = sc.iloc[instanceIndex]
            if cl[instanceIndex] != 0:
                outstanding.append(instance)
            else:
                trapped.append(instance)
                index_trapped.append(instanceIndex)
        
        return outstanding, trapped, index_trapped

    @staticmethod
    def get_classes(data):
        # Get majority class
        ma = data.iloc[:, -1].value_counts().idxmax()

        # Get minority class
        mi = data.iloc[:, -1].value_counts().idxmin()

        # Get minority classes set
        mi_aux = data.iloc[:, -1].value_counts() == data.iloc[:, -1].value_counts().min()
        mi_set = mi_aux[mi_aux == True]
        return ma, mi, mi_set
    
    @staticmethod
    def get_class_set(data, class_name):
        # print(">>> " + str(data.iloc[:, -1].size) )
        # print(class_name)
        return data[data.iloc[:, -1] == class_name]

    def generate_synthetic_instances(self, sc, g, trapped, weight):
        xj = 0
        si = []
        for times in range(g):
            for index, i in sc.iterrows():
                if i in trapped:
                    #fazer knn distancia tolist()[0]
                    #weight[vis] passo 6 xipj
                    xj = i.get_neighbor_high_weight()# fazer isso
                else:
                    #vizinho knn
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

    @staticmethod
    def probability_distribution(xi, index_trapped, k1neighbors, w1, fs_fd, data, sw): # the return is a list of probabilities with the ormat [xj, xipj], xj is the instance from a different class and xipj it probability
        xipj = {} 
        contains_class_c = True
        weightlist = []

        for xj_index in k1neighbors: 
            xj = data.iloc[xj_index]
            smaller_distances= Instance.ss(xi, xj, fs_fd)
            npn = Instance.dpn(xi, xj, smaller_distances, data)
            
            for element in npn:
                if element.iloc[-1] == xi.iloc[-1]: #element.class == c 
                    contains_class_c = False
                    break
                
            if(xj.iloc[-1] != xi.iloc[-1]):
                print("passou aqui")
                xipj[xj_index] = fs_fd.get(xj_index, 0)/(math.sum(weightlist))

        if not contains_class_c:
            print("contains_class_c")
            weight = 1 + (w1/math.e)
            fs_fd[index_trapped] = weight
            sw[index_trapped] = weight

        
        for xj_index in fs_fd.keys():
            count = 0
            for xl in k1neighbors:
                if xj_index in sw:
                    count += sw[xj_index][xl]
                    xipj[xj_index] = pd.Series.div(sw[xj_index], count)
        
        print(len(sw))
        return xipj