import csv
import random
import math
import operator
import time


def euclideanDistance(instance1, instance2, length):
    distance = 0
    for x in range(length):
        distance += pow((instance1[x] - instance2[x]), 2)
    return math.sqrt(distance)

def getNeighbors(trainingSet, testInstance, k):
    distances = []
    length = len(testInstance)-1
    for x in range(len(trainingSet)):
        dist = euclideanDistance(testInstance, trainingSet[x], length)
        distances.append((trainingSet[x], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors

def getResponse(neighbors):
    classVotes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    sortedVotes = sorted(classVotes.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]
    
def getUnionset(instances1, instances2): #merges two lists into one
    return instances1 + instances2
    
def getNeighborsfromUnion(instancex, instances1, instances2, k): #gets the k neighbors from instancex and the union of instances1 and instances2
    return getNeighbors(getUnionset(instances1, instances2),instancex, k)

def getListIndexdistances(instance, instancesList): #returns a list of indexes and distances from instance to every element of instancesList 
    list = []
    for x in range(len(instancesList)):
        list.append([x, euclideanDistance(instance, instancesList[0], len(instance)-1)])
    return list
            

 
def main():
    neighbors = [[1,1,1,'a'], [2,2,2,'a'], [3,3,3,'b']]
    neighbors1 = [[3,4,5,'g'], [3,2,1,'h']]
    instance = [2,3,6,'f']
    response = getListIndexdistances(instance, neighbors1)
    print(response)

main()