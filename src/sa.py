import time
import random
import numpy as np

from src.set_data import get_data
from src.set_result import visual, save_result
from src.local_search import LocalSearch

class SA(LocalSearch):
    def __init__(self, config):
        self.df, self.a, self.b = get_data(config)
        super().__init__(config, self.df, self.a, self.b)
        
    def initial(self):
        self.reset()
        self.add_t_earliest()
        while len(self.visit) > 0:
            delta, u, v, alt = self.insert_all()
            if delta is None:
                self.add_t_earliest()
            else:
                self.obj += delta
                self.routes_t[v], self.begins_t[v], self.waits_t[v] = alt
                self.visit -= {u}
        self.sol_best = [self.routes_t, self.routes_d, self.begins_t, self.begins_d, self.waits_t]
        self.obj_best = self.obj
        self.logs.append(self.obj)
        # visual(self.config, self.df, self.sol_best[0], self.sol_best[1], 'init')
        return
        
    def solve(self):
        time_s = time.time()
        self.initial()
        T = self.config['T_0']
        c1 = dict((nbr, []) for nbr in self.Ns)
        c2 = dict((nbr, []) for nbr in self.Ns)
        c3 = dict((nbr, []) for nbr in self.Ns)
        while T > self.config['T_1']:
            for _ in range(self.config['iter']):
                nbr = self.choose_nbr()
                delta, v, alt = nbr()
                c1[nbr].append(delta)
                if delta is not None:
                    c2[nbr].append(delta)
                    if delta < 0 or random.random() < np.exp(- delta / T):
                        c3[nbr].append(delta)
                        self.obj += delta
                        idx = None
                        if nbr == self.Nt_4 and len(self.routes_t) < self.config['n_t']:
                            self.routes_t.append([])
                            self.begins_t.append([])
                            self.waits_t.append([])
                        for num in range(len(v)):
                            if len(alt[num]) == 3:
                                self.routes_t[v[num]], self.begins_t[v[num]], self.waits_t[v[num]] = alt[num]
                                if len(self.routes_t[v[num]]) == 2:
                                    idx = v[num]
                            else:
                                self.routes_d[v[num]], self.begins_d[v[num]] = alt[num]
                        if idx is not None:
                            del self.routes_t[idx], self.begins_t[idx], self.waits_t[idx]
                        if self.obj < self.obj_best:
                            self.sol_best, self.obj_best = [self.routes_t, self.routes_d, self.begins_t, self.begins_d, self.waits_t], self.obj
                self.logs.append(self.obj_best)
                if len(self.logs) > self.config['patience']:
                    if self.config['epsilon'] >= (self.logs[-self.config['patience']] - self.logs[-1]) / self.logs[-1]:
                        T = 0
                        break
            T = T * self.config['r']
        time_e = time.time()
        # if self.config['n'] <= 40:
            # visual(self.config, self.df, self.sol_best[0], self.sol_best[1])
        save_result(self.config, [len(self.logs), time_e-time_s, self.obj_best])
        np.save(f"./logs/{self.config['name']}", np.array(self.logs))
        return c1,c2,c3
        # returns