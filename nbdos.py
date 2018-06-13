from sklearn.neighbors import NearestNeighbors

class nbdos:

    def list_neighbor(sfs, xi, k):
        neigh = NearestNeighbors(n_neighbors=k)
        neigh.fit(sfs)
        return neigh.kneighbors(xi, return_distance=False)


    def nbdos(self, sc, k, sc_nk, rTh, nTh):
        sfs = []
        cl = 0
        hck = {}
        for xi in sc:
            tem = [c for c in sc_nk if c[-1] == sc[0][-1]]

            if round(len(tem)/k) >= rTh:
                sfs.appen(xi)
                hck[xi] = tem
        
        for xi in sfs:
            tem = self.list_neighbor(sfs, xi, k)
            



    def expand_cluster():
        pass