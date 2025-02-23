import numpy as np
import gurobipy as gp
from gurobipy import GRB, quicksum

from src.set_data import get_data
from src.set_result import get_result, save_result, visual

class Milp:
    def __init__(self, config):
        self.config = config.copy()
        self.df, self.a, self.b = get_data(self.config)
        
    def model(self):
        bigM = max(self.df['e']) +  max(self.df['l']) + np.max(self.a)
        N = set(range(self.config['n'] + 2))
        N0 = N - {self.config['n'] + 1}
        N1 = N - {0}
        C = N0 - {0}
        V = set(range(1, self.config['n_t'] + 1))
        W = set(range(1, self.config['n_d'] + 1))
        K = set(range(1, self.config['n_l'] + 1))
        arc_x = [(v, i, j) for v in V for i in N0 for j in N1 if i != j]
        arc_y = [(w, k, i, j) for w in W for k in K for i in N0 for j in N1 if i != j]
        arc_t1 = [(v, i) for v in V for i in N]
        arc_t2 = [(w, k, i) for w in W for k in K for i in N]
        m = gp.Model()
        x = m.addVars(arc_x, vtype = GRB.BINARY, name = 'x')
        y = m.addVars(arc_y, vtype = GRB.BINARY, name = 'y')
        t1 = m.addVars(arc_t1, lb = 0, vtype = GRB.CONTINUOUS, name = 't1')
        t2 = m.addVars(arc_t2, lb =0, vtype = GRB.CONTINUOUS, name = 't2')
        s = m.addVars(arc_t1, lb =0, vtype = GRB.CONTINUOUS, name = 's')
        m.setObjective(self.config['lambda'] * quicksum(self.a[i, j] * x[v, i, j] for (v, i, j) in arc_x)
                + (1 - self.config['lambda']) * quicksum(self.b[i, j] * y[w, k, i, j] for (w, k, i, j) in arc_y), GRB.MINIMIZE)
        m.addConstrs(((quicksum(x[v, i, j] for i in N0 - {j} for v in V) + quicksum(y[w, k, i, j] for i in N0 - {j} for k in K for w in W)) == 1 for j in C),
            name='all_customer')
        m.addConstrs((quicksum(x[v, 0, j] for j in N1) == 1 for v in V),
            name='t_departure')
        m.addConstrs((quicksum(y[w, k, 0, j] for j in N1) == 1 for k in K for w in W),
            name='d_departure')
        m.addConstrs((quicksum(x[v, i, j] for i in N0 - {j}) - quicksum(x[v, j, i] for i in N1 - {j}) == 0 for j in C for v in V),
            name='t_flow')
        m.addConstrs((quicksum(y[w, k, i, j] for i in N0 - {j}) - quicksum(y[w, k, j, i] for i in N1 - {j}) == 0 for j in C for k in K for w in W),
            name='d_flow')
        m.addConstrs((quicksum(x[v, i, self.config['n'] + 1] for i in N0) == 1 for v in V),
            name='t_arrival')
        m.addConstrs((quicksum(y[w, k, i, self.config['n'] + 1] for i in N0) == 1 for k in K for w in W),
            name='d_arrival')
        
        if self.config['fixed']:
            m.addConstrs((quicksum(self.b[i, j] * y[w, k, i, j] for j in N1 for i in N0 if i != j) <= self.config['tau'] for k in K for w in W),
            name='fixedrange')
        else:
            m.addConstrs((quicksum(self.b[i, j] * y[w, k, i, j] for j in N1 for i in N0 if i != j) <= self.config['alpha'] 
                        - self.config['beta'] * quicksum(self.df['q'][j] * quicksum(y[w, k, i, j] for i in N0 - {j}) for j in C) for k in K for w in W),
            name='consumption')
            
        m.addConstrs((quicksum(self.df['q'][j] * quicksum(y[w, k, i, j] for i in N0 - {j}) for j in C) <= self.config['Q'] for k in K for w in W),
            name='weight')
        m.addConstrs((quicksum(y[w, k, i, j] for j in C for i in N0 if i != j) <= self.config['P'] for k in K for w in W),
            name='capacity')
        m.addConstrs((t1[v, 0] == 0 for v in V),
            name='t_begin0')
        m.addConstrs((s[v, 0] == 0 for v in V),
            name='t_wait0')
        m.addConstrs((t2[w, k - 1, self.config['n'] + 1] - t2[w, k, 0] <= 0 for k in K - {1} for w in W),
            name='d_begin0')
        m.addConstrs((t1[v, i] + self.a[i, j] + s[v, j] - t1[v, j] <= bigM * (1 - x[v, i, j]) for j in N1 for i in N0 for v in V if i != j),
            name='t_time1')
        m.addConstrs((-t1[v, i] - self.a[i, j] - s[v, j] + t1[v, j] <= bigM * (1 - x[v, i, j]) for j in N1 for i in N0 for v in V if i != j),
            name='t_time2')
        m.addConstrs((t2[w, k, i] + self.b[i, j] - t2[w, k, j] <= bigM * (1 - y[w, k, i, j]) for j in N1 for i in N0 for k in K for w in W if i != j),
            name='d_time1')
        m.addConstrs((-t2[w, k, i] - self.b[i, j] + t2[w, k, j] <= bigM * (1 - y[w, k, i, j]) for j in N1 for i in N0 for k in K for w in W if i != j),
            name='d_time2')
        m.addConstrs((self.df['e'][i] - t1[v, i] <= (1 - quicksum(x[v, i, j] for j in N1 if i != j)) * bigM for i in C for v in V),
            name='t_tw1')
        m.addConstrs((self.df['e'][i] - t2[w, k, i] <= (1 - quicksum(y[w, k, i, j] for j in N1 if i != j)) * bigM for i in C for k in K for w in W),
            name='t_tw2')
        m.addConstrs((t1[v, i] - self.df['l'][i] <= (1 - quicksum(x[v, i, j] for j in N1 if i != j)) * bigM for i in C for v in V),
            name='d_tw1')
        m.addConstrs((t2[w, k, i] - self.df['l'][i] <= (1 - quicksum(y[w, k, i, j] for j in N1 if i != j)) * bigM for i in C for k in K for w in W),
            name='d_tw2')
        return m, x, y, t1, t2, s
    
    def time_callback(self, m, where):
        if where == GRB.Callback.MIP:
            lap = m.cbGet(GRB.Callback.RUNTIME)
            if lap > self.config['timelimit']:
                m.terminate()
    
    def solve(self):
        m, x, y, t1, t2, s = self.model()
        # m.setParam('OutputFlag', 0)
        m.setParam('MIPGap', self.config['gap'])
        m.setParam('TimeLimit', self.config['timelimit'])
        m.setParam('Heuristics', 0)
        m.optimize(self.time_callback)
        save_result(self.config, [m.Runtime, m.ObjVal, m.MIPGap])
        # if m.SolCount > 0:
        #     # pass
        routes_t, routes_d, begins_t, begins_d, waits_t = get_result(self.config, x, y, t1, t2, s)
            # if self.config['n'] <= 40:
        visual(self.config, self.df, routes_t, routes_d)
        return m, [routes_t, routes_d, begins_t, begins_d, waits_t]
        #     # return
        # # else: return m, m.Status
        # return