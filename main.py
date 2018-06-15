import pandas as pd
from SMOM import SMOM

data = pd.read_csv("data/data.csv", header=None)
data = data.iloc[:, 1:]
#min_class = data.iloc[:, -1].value_counts().min()
min_class = 16 #classe com 67 instancias
k1 = 12
k2 = 8

#1. Dividindo dados entre classe minoriataria e outros
sc, others =  SMOM.split_classes(data, min_class) 

#2.
for index, xi in sc.iterrows():
# 2.1)
    k3neighbors, neighbors_distance, k1th, distance, k3 = SMOM.nearestK3Instances(xi, sc, k1, k2)
# 2.2)
    k3EnemyNeighbor, fd_fs = SMOM.nearEnemy(xi, others, k3, distance, k3neighbors, neighbors_distance)
# 2.3)
    k2_neighbors = SMOM.k2_neighbors(k3neighbors, k3EnemyNeighbor, xi, k2)
#3.

#4.

#5.
#  5.1)
#  5.2)
#  5.3)

#6.
#  6.1)
#  6.2)

#7.

#8.
#   8.1.
#   8.2
#   8.3.




