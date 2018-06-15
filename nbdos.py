from sklearn.neighbors import NearestNeighbors

class Nbdos:
	
    def __init__(self):
        self.hck = {}

    def list_neighbor(self, data, sfs, xi, k):
        neigh = NearestNeighbors(n_neighbors=k)
        neigh.fit(data.iloc[sfs, :data.shape[1]], data.iloc[sfs, -1])
        return neigh.kneighbors([xi], return_distance=False).tolist()[0]

    def nbdos(self, data, sc, k, k2_neighbors, rTh, nTh):
        sfs = []
        cl = [0] * sc.shape[0]
        #print("aqui foi")
        for indeces, xi in sc.iterrows():
            tem = [a for a in self.list_neighbor(data, k2_neighbors, xi, k) if data.iloc[a, -1] == xi.iloc[-1]]
            #print(indeces)
            #print(len(tem))
            if round(len(tem)/k) >= rTh:
                print("Nem se quer aqui")
                sfs.append(indeces)
                self.hck[xi] = tem
        
        for xi in sfs:
            tem = self.list_neighbor(sfs, xi, cl, k)
            print("passa aqui")
            if xi in self.hck.keys():
                self.hck[xi].append(tem)
            else:
                self.hck[xi] = tem

        #print(list(sfs))
        curld = 0
        for xi in sfs:
            if cl[xi] == 0:
                curld += 1
                self.expand_cluster(sfs, xi, cl, curld) #voltar a partir daqui

        for i in range(curld):
            ci = [x for x in range(len(cl)) if cl[x] == i]
            #ci = [xi for j, xi in sc.iterrows() if cl == i]

            if len(ci) < nTh:
                for j in ci:
                    cl[j] = 0
        
        return cl 
    
    def expand_cluster(self, sfsc, xi, cl, curld):
        sfC = [xi]
        cl[xi] = curld

        # Check if list is empty
        while sfC:
            xj = sfC.__getitem__(-1)
            for xl in self.hck[xj]:
                if xl.cl == 0:
                    xl.cl = curld
                    if xl in sfsc:
                        sfC.append(xi)

            sfC.remove(xj)
        return cl
