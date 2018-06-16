import pandas as pd
from SMOM import SMOM
from nbdos import Nbdos
import pandas as pd
# import main2

data = pd.read_csv("data/data.csv", header=None)
data = data.iloc[:, 1:].head(500)
#min_class = data.iloc[:, -1].value_counts().min()
min_class = 16 #classe com 67 instancias
k1 = 12
k2 = 8
rTh = 5/8
nTh = 10
k = 2
w1 = 0.2
w2 = 0.5
r1 = 1/3
r2 = 0.2

#1. Dividindo dados entre classe minoriataria e outros
sc, others =  SMOM.split_classes(data, min_class) 
    
xi_fs_fd = {}

#2.
for index, xi in sc.iterrows():
    fs_fd = {}
# 2.1)
    k3neighbors, neighbors_distance, k1th, distance, k3 = SMOM.nearestK3Instances(xi, sc, k1, k2)
# 2.2)
    k3EnemyNeighbor, fs, fd = SMOM.nearEnemy(xi, others, k3, distance, k3neighbors, neighbors_distance)
# 2.3)
    k2_neighbors = SMOM.k2_neighbors(data, k3neighbors, k3EnemyNeighbor, xi, k2, k3)
    
    for i in range(len(fs)):
        fs_fd[fs[i]] = fd[i]
    xi_fs_fd[index] = fs_fd
#3.
cl = Nbdos().nbdos(data, sc, k2, k2_neighbors, rTh, nTh)
#4.
outstandings, trappeds, index_trapped = SMOM.filterOutstanding(sc, cl, min_class)       
#5.
dic = SMOM.selection_weigth(data, outstandings, trappeds, k3, w1, w2, r1, r2, xi_fs_fd, min_class)

#6.

for i in range(len(trappeds)):
    k1neighbors = SMOM.kneigbor(data, trappeds[i], k1, False)[0]
    if (xi_fs_fd.get(index_trapped[i], None) != None):
        print()
        xipj = SMOM.probability_distribution(xi, index_trapped, k1neighbors, w1, xi_fs_fd.get(index_trapped[i]), data)
        print(xipj)

#  6.1)
#  6.2)


#7.

#8.
# instance
#   8.1.
#   8.2
#   8.3.
