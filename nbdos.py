from sklearn.neighbors import NearestNeighbors

class nbdos:

    def __init__(self):
        self.hck = {}

    def list_neighbor(self, sfs, xi, k):
        neigh = NearestNeighbors(n_neighbors=k)
        neigh.fit(sfs)
        return neigh.kneighbors([xi], return_distance=False).tolist()[0]

    def nbdos(self, base, sc, k, rTh, nTh):
        sfs = []

        for i in sc:
            i.cl = 0

        for xi in sc:
            tem = [a for a in self.list_neighbor(base, xi, k) if a.label == xi.label]

            if round(len(tem)/k) >= rTh:
                sfs.append(xi)
                self.hck[xi] = tem
        
        for xi in sfs:
            tem = self.list_neighbor(sfs, xi, k)

            if xi in self.hck.keys():
                self.hck[xi].append(tem)
            else:
                self.hck[xi] = tem

        curld = 0
        for xi in sfs:
            if xi.cl == 0:
                curld += 1
                self.expand_cluster(sfs, xi, curld) #voltar a partir daqui

        for i in range(curld):
            ci = [xi for xi in sc if xi.cl == i]

            if len(ci) < nTh:
                for j in ci:
                    j.cl = 0

    def expand_cluster(self, sfsc, xi, curld):
        sfC = [xi]
        xi.cl = curld

        # Check if list is empty
        while sfC:
            xj = sfC.__getitem__(-1)
            for xl in self.hck[xj]:
                if xl.cl == 0:
                    xl.cl = curld
                    if xl in sfsc:
                        sfC.append(xi)

            sfC.remove(xj)
