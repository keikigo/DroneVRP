import random
import numpy as np

from src.make_route import MakeRoute

class LocalSearch(MakeRoute):
    def __init__(self, config, df, a, b):
        super().__init__(config, df, a, b)
        self.C_d = set([u for u in range(1, config['n'] + 1) if df['q'][u] <= config['Q'] and\
            b[0, u] + b[u, config['n'] + 1] <= config['alpha'] - config['beta'] * df['q'][u]])
        self.Nts = [
            self.Nt_1,   #insert
            self.Nt_2,   #swap
            self.Nt_3,   #reverse
            self.Nt_4   #add
            ]
        self.Ntts = [
            self.Ntt_1,  #insert
            self.Ntt_2,  #swap
            # self.Ntt_3, #add
            self.Ntt_4, #swap_all
            self.Ntt_5, #longest
        ] if self.config['n_t']>=2 else []
        self.Nds = [
            self.Nd_1,   #insert
            self.Nd_2   #swap
        ] if self.config['n_d']>=1 else []
        self.Ndds = [
            self.Ndd_1,  #insert
            self.Ndd_2,  #swap
        ] if self.config['n_d']>=2 else []
        self.Ntds = [
            self.Ntd_1,  #insert t -> d
            self.Ntd_2, #insert d -> t
            self.Ntd_3 #swap
        ] if self.config['n_d']>=1 else []
        self.Ns = self.Nts + self.Ntts + self.Nds + self.Ndds + self.Ntds
        self.i1 = self.Ns.index(self.Nt_4)
        
    def reset(self):
        self.routes_t = []
        self.routes_d = [[0 for k in range(self.config['n_l']+1)] for w in range(self.config['n_d'])]
        self.begins_t = []
        self.begins_d = [[0 for k in range(self.config['n_l'] + 1)] for w in range(self.config['n_d'])]
        self.waits_t = []
        self.obj = 0
        self.visit = set(range(1, self.config['n'] + 1))
        self.sol_best = None
        self.obj_best = None
        self.logs = []
        # self.imprv = [0 for _ in self.Ns]
        # self.weights = [1000 for _ in self.Ns]
        random.seed(self.config['seed'])
        
    def choose_nbr(self):
        n_t = len(self.routes_t)
        n_tt = n_t if n_t>=2 else 0
        n_d = sum([1 for v in self.routes_d if len(v)>1+self.config['n_l']])
        n_dd = n_d if n_d>=2 else 0
        n_td = 0.5*(n_t+n_d)
        
        # weights = [1 for _ in self.Ns] 
        weights = [n_t for _ in self.Nts] + [n_tt for _ in self.Ntts] + [n_d for _ in self.Nds] \
            + [n_dd for _ in self.Ndds] + [n_td for _ in self.Ntds]
            
            
        weights[self.i1] = 0 if len(self.routes_t) == self.config['n_t'] else 1
        nbr = random.choices(self.Ns, weights=weights, k=1)[0]
        return nbr
        
    def shuffle(self, routes, i1 = 0, i2 = 0):
        id = [(v, i) for v in range(len(routes)) for i in range(i1, len(routes[v]) - i2)]
        random.shuffle(id)
        return id
    
    def add_t_earliest(self):
        u = int(self.df[self.df.index.isin(self.visit)]['l'].sort_values().index[0])
        route = [0, u, self.config['n'] + 1]
        begin = [0, 0, 0]
        wait = [0, 0, 0]
        cost = self.config['lambda'] * (self.a[0, u] + self.a[u, self.config['n'] + 1])
        for i in [1, 2]:
            begin[i] = max(self.df['e'][route[i]], begin[i - 1] + self.a[route[i - 1], route[i]])
            wait[i] = max(0, self.df['e'][route[i]] - begin[i - 1] - self.a[route[i - 1], route[i]])
        self.routes_t.append(route)
        self.begins_t.append(begin)
        self.waits_t.append(wait)
        self.obj += cost
        self.visit = self.visit - {u}
        return
        
    def insert_all(self):
        criterion1 = dict()
        for u in self.visit:
            costs = np.array([[np.nan for i in range(max(len(route) for route in self.routes_t))] for route in self.routes_t])
            for v in range(len(self.routes_t)):
                route = self.routes_t[v]
                begin = self.begins_t[v]
                wait = self.waits_t[v]
                for p in range(1, len(route)):
                    # r_alt = route[:p] + [u] + route[p:]
                    # alt = self.t_all(r_alt)
                    alt = self.t_insert(route, begin, wait, p, u)
                    if alt != False:
                        costs[v, p] = self.config['lambda'] * (self.a[route[p-1], u] + self.a[u, route[p]] - self.a[route[p-1], route[p]])
            if np.all(np.isnan(costs)):
                return None, None, None, None
            arg = np.unravel_index(np.nanargmin(costs), costs.shape)
            criterion1[u, arg] = costs[arg]
        criterion2 = criterion1.copy()
        for u, arg in criterion2:
            criterion2[u, arg] = self.a[0, u] - criterion1[u, arg]
        u, (v, p) =  min(criterion2, key= criterion2.get)
        r = self.routes_t[v]
        # r_alt = r[:p] + [u] + r[p:]
        delta = self.config['lambda'] * (self.a[r[p-1], u] + self.a[u, r[p]] - self.a[r[p-1], r[p]])
        # alt = self.t_all(r_alt)
        alt = self.t_insert(r, self.begins_t[v], self.waits_t[v], p, u)
        return delta, u, v, alt
    
    
    '''
    [1, i, 2, 3, j, 4] -> [1, 2, 3, i, j, 4]
    '''
    def Nt_1(self):
        id_1 = self.shuffle(self.routes_t, 1, 1)
        for v, i in id_1:
            r = self.routes_t[v]
            id_2 = list(set(range(1, len(r))) - {i, i+1})
            random.shuffle(id_2)
            for j in id_2:
                r_alt = r.copy()
                del r_alt[i]
                r_alt.insert(r_alt.index(r[j]), r[i])
                alt  = self.t_all(r_alt)
                if alt != False:
                    delta = self.config['lambda'] * (
                            self.a[r[i-1], r[i+1]] + self.a[r[j-1], r[i]] + self.a[r[i], r[j]]
                            - self.a[r[i-1], r[i]] - self.a[r[i], r[i+1]] - self.a[r[j-1], r[j]])
                    return delta, [v], [alt]
        return None, None, None
    
    
    '''
    [1, i, 2, 3, j, 4] -> [1, j, 2, 3, i, 4]
    '''
    def Nt_2(self):
        id_1 = self.shuffle(self.routes_t, 1, 1)
        for v, i in id_1:
            r = self.routes_t[v]
            id_2 = list(set(range(1, len(r) - 1)) - {i})
            random.shuffle(id_2)
            for j in id_2:
                r_alt = r.copy()
                r_alt[i], r_alt[j] = r[j], r[i]
                alt  = self.t_all(r_alt)
                if alt != False:
                    if abs(j - i) == 1:
                        p = min(i,j)
                        delta = self.config['lambda'] * (
                            self.a[r[p-1], r[p+1]] + self.a[r[p+1], r[p]] + self.a[r[p], r[p+2]]
                            - self.a[r[p-1], r[p]] - self.a[r[p], r[p+1]] - self.a[r[p+1], r[p+2]])
                    else:
                         delta = self.config['lambda'] * (
                            self.a[r[i-1], r[j]] + self.a[r[j], r[i+1]] + self.a[r[j-1], r[i]] + self.a[r[i], r[j+1]]
                            - self.a[r[i-1], r[i]] - self.a[r[i], r[i+1]] - self.a[r[j-1], r[j]] - self.a[r[j], r[j+1]]) 
                    return delta, [v], [alt]
        return None, None, None
    
    
    '''
    [1, i, 2, 3, j, 4] -> [1, j, 3, 2, i, 4]
    '''
    def Nt_3(self):
        id_1 = self.shuffle(self.routes_t, 1, 1)
        for v, i in id_1:
            r = self.routes_t[v]
            id_2 = list(set(range(1, len(r) - 1)) - {i})
            random.shuffle(id_2)
            for j in id_2:
                p1, p2 = min(i, j), max(i, j)
                r_alt = r[:p1] + r[p1:p2+1][::-1] + r[p2+1:]
                alt  = self.t_all(r_alt)
                if alt != False:
                    delta = self.config['lambda'] * (
                        self.a[r[p1-1], r[p2]] + self.a[r[p1], r[p2+1]] 
                        - self.a[r[p1-1], r[p1]] - self.a[r[p2], r[p2+1]])
                    return delta, [v], [alt]
        return None, None, None
    
    '''
    [1,2,3,i,4,5,6,7] -> [1,2,3,i,7], [1,4,5,6,7]
    '''
    def Nt_4(self):
        v1 = len(self.routes_t)
        id_2 = self.shuffle(self.routes_t, 3, 4)
        for v2, i2 in id_2:
            r2 = self.routes_t[v2]
            r1_alt = r2[:i2+1] + [r2[-1]]
            r2_alt = [r2[0]] + r2[i2+1:]
            alt1 = self.t_all(r1_alt)
            alt2 = self.t_all(r2_alt)
            if alt1 != False and alt2 !=False:
                delta = self.config['lambda'] * (
                    self.a[r2[i2], r2[-1]] + self.a[r2[0], r2[i2+1]] - self.a[r2[i2], r2[i2+1]])
                return delta, [v1, v2], [alt1, alt2]
        return None, None, None
    
    '''
    [1, i, 2, 3, 4] -> [1, 2, 3, 4]
    [11, 12, 13, j, 14] -> [11, 12, 13, i, j, 14]
    '''
    def Ntt_1(self):
        id_1 = self.shuffle(self.routes_t, 1, 1)
        id_2 = self.shuffle(self.routes_t, 1)
        for v1, i1 in id_1:
            r1 = self.routes_t[v1]
            for v2, i2 in id_2:
                if v1 != v2:
                    r2 = self.routes_t[v2]
                    alt1 = self.t_del(r1, self.begins_t[v1], self.waits_t[v1], i1)
                    alt2 = self.t_insert(r2, self.begins_t[v2], self.waits_t[v2], i2, r1[i1])
                    if alt2 != False:
                        delta = self.config['lambda'] * (
                            self.a[r1[i1-1], r1[i1+1]] + self.a[r2[i2-1], r1[i1]] + self.a[r1[i1], r2[i2]]
                            - self.a[r1[i1-1], r1[i1]] - self.a[r1[i1], r1[i1+1]] - self.a[r2[i2-1], r2[i2]])
                        return delta, [v1, v2], [alt1, alt2]
        return None, None, None
    
    
    '''
    [1, i, 2, 3, 4] -> [1, j, 2, 3, 4]
    [11, 12, 13, j, 14] -> [11, 12, 13, i, 14]
    '''
    def Ntt_2(self):
        id_1 = self.shuffle(self.routes_t, 1, 1)
        id_2 = self.shuffle(self.routes_t, 1, 1)
        for v1, i1 in id_1:
            r1 = self.routes_t[v1]
            for v2, i2 in id_2:
                if v1 != v2:
                    r2 = self.routes_t[v2]
                    r1_alt, r2_alt = r1.copy(), r2.copy()
                    r1_alt[i1], r2_alt[i2] = r2[i2], r1[i1]
                    alt1 = self.t_all(r1_alt)
                    alt2 = self.t_all(r2_alt)
                    if alt1 != False and alt2 !=False:
                        delta = self.config['lambda'] * (
                            self.a[r1[i1-1], r2[i2]] + self.a[r2[i2], r1[i1+1]] + self.a[r2[i2-1], r1[i1]] + self.a[r1[i1], r2[i2+1]]
                            -self.a[r1[i1-1], r1[i1]] - self.a[r1[i1], r1[i1+1]] - self.a[r2[i2-1], r2[i2]]- self.a[r2[i2], r2[i2+1]])
                        return delta, [v1, v2], [alt1, alt2]
        return None, None, None
    
    '''
    [1,2,3,i,4,5,6] -> [1,2,3,i,14,15]
    [11,12,13,j,14,15] -> [11,12,13,j,5,6]
    '''
    def Ntt_4(self):
        id_1 = self.shuffle(self.routes_t, 2, 2)
        id_2 = self.shuffle(self.routes_t, 2, 2)
        for v1, i1 in id_1:
            r1 = self.routes_t[v1]
            for v2, i2 in id_2:
                if v1 != v2:
                    r2 = self.routes_t[v2]
                    r1_alt = r1[:i1+1] + r2[i2+1:]
                    r2_alt = r2[:i2+1] + r1[i1+1:]
                    alt1 = self.t_all(r1_alt)
                    alt2 = self.t_all(r2_alt)
                    if alt1 != False and alt2 !=False:
                        delta = self.config['lambda'] * (
                            self.a[r1[i1], r2[i2+1]] + self.a[r2[i2], r1[i1+1]] - self.a[r1[i1], r1[i1+1]]- self.a[r2[i2], r2[i2+1]])
                        return delta, [v1, v2], [alt1, alt2]
        return None, None, None
    
    
    '''
    longest
    '''
    def Ntt_5(self):
        v1 = max(range(len(self.routes_t)), key=lambda i: len(self.routes_t[i]))
        r1 = self.routes_t[v1]
        id_1 = list(set(range(1, len(r1)-1)))
        random.shuffle(id_1)
        id_2 = self.shuffle(self.routes_t, 1)
        for i1 in id_1:
            for v2, i2 in id_2:
                if v1 != v2:
                    r2 = self.routes_t[v2]
                    alt1 = self.t_del(r1, self.begins_t[v1], self.waits_t[v1], i1)
                    alt2 = self.t_insert(r2, self.begins_t[v2], self.waits_t[v2], i2, r1[i1])
                    if alt2 != False:
                        delta = self.config['lambda'] * (
                            self.a[r1[i1-1], r1[i1+1]] + self.a[r2[i2-1], r1[i1]] + self.a[r1[i1], r2[i2]]
                            - self.a[r1[i1-1], r1[i1]] - self.a[r1[i1], r1[i1+1]] - self.a[r2[i2-1], r2[i2]])
                        return delta, [v1, v2], [alt1, alt2]
        return None, None, None
    
    '''
    [1, i, 2, 3, j, 4] -> [1, 2, 3, i, j, 4]
    '''
    def Nd_1(self):
        id_1 = self.shuffle(self.routes_d, 1, 1)
        for v, i in id_1:
            if self.routes_d[v][i] != 0:
                r = self.routes_d[v]
                id_2 = list(set(range(1, len(r))) - {i, i+1})
                random.shuffle(id_2)
                for j in id_2:
                    r_alt = r.copy()
                    del r_alt[i]
                    if i < j:
                        r_alt.insert(j-1, r[i])
                    else:
                        r_alt.insert(j, r[i])
                    alt  = self.d_all(r_alt)
                    if alt != False:
                        delta = (1 - self.config['lambda']) * (
                                self.b[r[i-1], r[i+1]] + self.b[r[j-1], r[i]] + self.b[r[i], r[j]]
                                - self.b[r[i-1], r[i]] - self.b[r[i], r[i+1]] - self.b[r[j-1], r[j]])
                        return delta, [v], [alt]
        return None, None, None
    
    
    '''
    [1, i, 2, 3, j, 4] -> [1, j, 2, 3, i, 4]
    '''
    def Nd_2(self):
        id_1 = self.shuffle(self.routes_d, 1, 1)
        for v, i in id_1:
            if self.routes_d[v][i] != 0:
                r = self.routes_d[v]
                id_2 = list(set(range(1, len(r) - 1)) - {i})
                random.shuffle(id_2)
                for j in id_2:
                    r_alt = r.copy()
                    r_alt[i], r_alt[j] = r[j], r[i]
                    alt  = self.d_all(r_alt)
                    if alt != False:
                        if abs(j - i) == 1:
                            p = min(i,j)
                            delta = (1 - self.config['lambda']) * (
                                self.b[r[p-1], r[p+1]] + self.b[r[p+1], r[p]] + self.b[r[p], r[p+2]]
                                - self.b[r[p-1], r[p]] - self.b[r[p], r[p+1]] - self.b[r[p+1], r[p+2]])
                        else:
                            delta = (1 - self.config['lambda']) * (
                                self.b[r[i-1], r[j]] + self.b[r[j], r[i+1]] + self.b[r[j-1], r[i]] + self.b[r[i], r[j+1]]
                                - self.b[r[i-1], r[i]] - self.b[r[i], r[i+1]] - self.b[r[j-1], r[j]] - self.b[r[j], r[j+1]]) 
                        return delta, [v], [alt]
        return None, None, None
    
    
    '''
    [1, i, 2, 3, 4] -> [1, 2, 3, 4]
    [11, 12, 13, j, 14] -> [11, 12, 13, i, j, 14]
    '''
    def Ndd_1(self):
        id_1 = self.shuffle(self.routes_d, 1, 1)
        id_2 = self.shuffle(self.routes_d, 1)
        for v1, i1 in id_1:
            if self.routes_d[v1][i1] != 0:
                r1 = self.routes_d[v1]
                for v2, i2 in id_2:
                    if v1 != v2 and self.routes_d[v2][i2] != 0:
                        r2 = self.routes_d[v2]
                        r1_alt = r1[:i1] + r1[i1+1:]
                        r2_alt = r2[:i2] + [r1[i1]] + r2[i2:]
                        alt1 = self.d_all(r1_alt)
                        alt2 = self.d_all(r2_alt)
                        if alt1 != False and alt2 != False:
                            delta = (1 - self.config['lambda']) * (
                                self.b[r1[i1-1], r1[i1+1]] + self.b[r2[i2-1], r1[i1]] + self.b[r1[i1], r2[i2]]
                                - self.b[r1[i1-1], r1[i1]] - self.b[r1[i1], r1[i1+1]] - self.b[r2[i2-1], r2[i2]])
                            return delta, [v1, v2], [alt1, alt2]
        return None, None, None
    
    
    '''
    [1, i, 2, 3, 4] -> [1, j, 2, 3, 4]
    [11, 12, 13, j, 14] -> [11, 12, 13, i, 14]
    '''
    def Ndd_2(self):
        id_1 = self.shuffle(self.routes_d, 1, 1)
        id_2 = self.shuffle(self.routes_d, 1, 1)
        for v1, i1 in id_1:
            if self.routes_d[v1][i1] != 0:
                r1 = self.routes_d[v1]
                for v2, i2 in id_2:
                    if v1 != v2 and self.routes_d[v2][i2] != 0:
                        r2 = self.routes_d[v2]
                        r1_alt, r2_alt = r1.copy(), r2.copy()
                        r1_alt[i1], r2_alt[i2] = r2[i2], r1[i1]
                        alt1 = self.d_all(r1_alt)
                        alt2 = self.d_all(r2_alt)
                        if alt1 != False and alt2 !=False:
                            delta = (1 - self.config['lambda']) * (
                                self.b[r1[i1-1], r2[i2]] + self.b[r2[i2], r1[i1+1]] + self.b[r2[i2-1], r1[i1]] + self.b[r1[i1], r2[i2+1]]
                                -self.b[r1[i1-1], r1[i1]] - self.b[r1[i1], r1[i1+1]] - self.b[r2[i2-1], r2[i2]]- self.b[r2[i2], r2[i2+1]])
                            return delta, [v1, v2], [alt1, alt2]
        return None, None, None
    
    
    '''
    t: [1, i, 2, 3, 4] -> [1, 2, 3, 4]
    d: [11, 12, 13, j, 14] -> [11, 12, 13, i, j, 14]
    '''
    def Ntd_1(self):
        id_1 = self.shuffle(self.routes_t, 1, 1)
        id_2 = self.shuffle(self.routes_d, 1)
        for v1, i1 in id_1:
            r1 = self.routes_t[v1]
            if r1[i1] in self.C_d:
                for v2, i2 in id_2:
                    r2 = self.routes_d[v2]
                    r1_alt = r1[:i1] + r1[i1+1:]
                    r2_alt = r2[:i2] + [r1[i1]] + r2[i2:]
                    alt1 = self.t_all(r1_alt)
                    alt2 = self.d_all(r2_alt)
                    if alt1 != False and alt2 != False:
                        delta = self.config['lambda'] * (
                                self.a[r1[i1-1], r1[i1+1]] - self.a[r1[i1-1], r1[i1]] - self.a[r1[i1], r1[i1+1]]) \
                            + (1 - self.config['lambda']) * (
                                self.b[r2[i2-1], r1[i1]] + self.b[r1[i1], r2[i2]] - self.b[r2[i2-1], r2[i2]])
                        return delta, [v1, v2], [alt1, alt2]
        return None, None, None
    
    '''
    d: [1, i, 2, 3, 4] -> [1, 2, 3, 4]
    t: [11, 12, 13, j, 14] -> [11, 12, 13, i, j, 14]
    '''
    def Ntd_2(self):
        id_1 = self.shuffle(self.routes_d, 1, 1)
        id_2 = self.shuffle(self.routes_t, 1)
        for v1, i1 in id_1:
            if self.routes_d[v1][i1] != 0:
                r1 = self.routes_d[v1]
                for v2, i2 in id_2:
                    r2 = self.routes_t[v2]
                    r1_alt = r1[:i1] + r1[i1+1:]
                    # r2_alt = r2[:i2] + [r1[i1]] + r2[i2:]
                    alt1 = self.d_all(r1_alt)
                    # alt2 = self.t_all(r2_alt)
                    alt2 = self.t_insert(r2, self.begins_t[v2], self.waits_t[v2], i2, r1[i1])
                    if alt1 != False and alt2 != False:
                        delta = (1 - self.config['lambda']) * (
                                self.b[r1[i1-1], r1[i1+1]] - self.b[r1[i1-1], r1[i1]] - self.b[r1[i1], r1[i1+1]]) \
                            + self.config['lambda'] * (
                                self.a[r2[i2-1], r1[i1]] + self.a[r1[i1], r2[i2]] - self.a[r2[i2-1], r2[i2]])
                        return delta, [v1, v2], [alt1, alt2]
        return None, None, None
    
    '''
    t: [1, i, 2, 3, 4] -> [1, j, 2, 3, 4]
    d: [11, 12, 13, j, 14] -> [11, 12, 13, i, 14]
    '''
    def Ntd_3(self):
        id_1 = self.shuffle(self.routes_t, 1, 1)
        id_2 = self.shuffle(self.routes_d, 1, 1)
        for v1, i1 in id_1:
            r1 = self.routes_t[v1]
            if r1[i1] in self.C_d:
                for v2, i2 in id_2:
                    if self.routes_d[v2][i2] != 0:
                        r2 = self.routes_d[v2]
                        r1_alt, r2_alt = r1.copy(), r2.copy()
                        r1_alt[i1], r2_alt[i2] = r2[i2], r1[i1]
                        alt1 = self.t_all(r1_alt)
                        alt2 = self.d_all(r2_alt)
                        if alt1 != False and alt2 !=False:
                            delta = self.config['lambda'] * (
                                    self.a[r1[i1-1], r2[i2]] + self.a[r2[i2], r1[i1+1]] - self.a[r1[i1-1], r1[i1]] - self.a[r1[i1], r1[i1+1]]) \
                                + (1 - self.config['lambda']) * (
                                    self.b[r2[i2-1], r1[i1]] + self.b[r1[i1], r2[i2+1]] - self.b[r2[i2-1], r2[i2]]- self.b[r2[i2], r2[i2+1]])
                            return delta, [v1, v2], [alt1, alt2]
        return None, None, None