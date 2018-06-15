import pandas as pd
import math

from SMOM import SMOM

def probability_distribution(xi, k1neighbors, w1): # the return is a list of probabilities with the ormat [xj, xipj], xj is the instance from a different class and xipj it probability
    xipj = [] 
    flag = True
    weightlist = []

    for xj in k1neighbors: #k1neighbors should follows the same format of the return of this method: i.e: [[[0.00078,0.9475,0.084,0.00009,"g"], 0.002568],[[0.045,0.0645,0.084,0.00009,"t"], 0.00568]]  while xi is a simple instance without the distance
        weightlist.append(xi.get_neighbor_weight(xj))
        smaller_distances= xi.ss(xj)
        npn = instance.dpn(xi, xj, smaller_distances)
            
        for element in npn:
            if element[-2][-1] == xi[-1]:
                flag = False
                break
            
        if(xj[-2][-1] != xi[-1]):
                xipj.append([xj,xi.get_neighbor_weight(xj)/(math.sum(weightlist))])

    if flag:
        weight = 1 + (w1/math.e)
        k1neighbors.append([xi, weight])
        
    return xipj


names = ["Sex", "Length", "Diameter", "Height", "Whole weight", "Shucked weight", "Viscera weight", "Shell weight", "Rings"]
text = pd.read_csv("data.csv", names=names)

text.iloc[:, -1].value_counts()

s = SMOM()
a, b = s.split_classes(text, 1)
print(a)