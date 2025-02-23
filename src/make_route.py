from scipy.optimize import linprog

class MakeRoute():
    def __init__(self, config, df, a, b):
        self.config = config.copy()
        self.df = df
        self.a = a
        self.b = b
        
    def t_all(self, route):
        begin = [0 for _ in route]
        wait = [0 for _ in route]
        for i in range(1, len(route)):
            wait[i] = max(0, self.df['e'][route[i]] - begin[i-1] - self.a[route[i-1], route[i]])
            begin[i] = begin[i-1] + self.a[route[i-1], route[i]] + wait[i]
            if begin[i] > self.df['l'][route[i]]:
                return False
        return route, begin, wait
    
    def d_all(self, route):
        depot = [i for i, u in enumerate(route) if u == 0]
        if len(depot) == len(route):
            return route, [0 for _ in route]
        for i in range(len(depot) - 1):
            route_sub = route[depot[i]: depot[i+1]+1]
            if len(route_sub) - 2 > self.config['P']:
                return False
            q = 0
            t_flight = 0
            for j in range(len(route_sub) - 1):
                q += self.df['q'][route_sub[j]]
                t_flight += self.b[route_sub[j], route_sub[j+1]]
            if q > self.config['Q'] or t_flight > self.config['alpha'] - self.config['beta'] * q:
                return False
                
        m = len(route)
        c = [0 for _ in route]
        bounds = [self.df[['e','l']].iloc[u].to_list() for u in route]
        A_ub = [[0 for _ in range(i)] + [1, -1] + [0 for _ in range(m - i - 2)]
                for i in range(m - 1) if i+1 in depot[1:-1]]
        A_eq = [[0 for _ in range(i)] + [1, -1] + [0 for _ in range(m - i - 2)]
                for i in range(m - 1) if i+1 not in depot[1:-1]]
        b_ub = [-self.b[route[i], route[i+1]] for i in range(m - 1) if i+1 in depot[1:-1]]
        b_eq = [-self.b[route[i], route[i+1]] for i in range(m - 1) if i+1 not in depot[1:-1]]
        res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds)
        if res.success:
            return route, res.x
        return False
    
    '''
    [0,11,12,13,14], (p, u) = (3, 5)
    [0,11,12,5,13,14]
    '''
    def t_insert(self, route, begin, wait, p, u):
        b_u = max(self.df['e'][u], begin[p - 1] + self.a[route[p - 1], u])
        w_u = max(0, self.df['e'][u] - begin[p - 1] + self.a[route[p - 1], u])
        if b_u > self.df['l'][u]:
            return False
        b_p_alt = max(self.df['e'][route[p]], b_u + self.a[u, route[p]])
        if b_p_alt > self.df['l'][route[p]]:
            return False
        route_alt = route.copy()
        begin_alt = begin.copy()
        wait_alt = wait.copy()
        push_forward = [0 for _ in route]
        begin_alt[p] = max(self.df['e'][route[p]], b_u + self.a[u, route[p]])
        wait_alt[p] = max(0, self.df['e'][route[p]] - b_u - self.a[u, route[p]])
        push_forward[p] = begin_alt[p] - begin[p]
        for i in range(p, len(route) - 1):
            push_forward[i+1] = max(0, push_forward[i] - wait[i+1])
            begin_alt[i+1] = begin[i+1] + push_forward[i+1]
            if begin_alt[i+1] > self.df['l'][route[i+1]]:
                return False
            wait_alt[i+1] = wait[i+1] - push_forward[i+1]
        route_alt.insert(p, u)
        begin_alt.insert(p, b_u)
        wait_alt.insert(p, w_u)
        return route_alt, begin_alt, wait_alt
    
    '''
    [0,11,12,13,14], p = 3
    [0,11,12,14]
    '''
    def t_del(self, route, begin, wait, p):
        route_alt, begin_alt, wait_alt = route.copy(), begin.copy(), wait.copy()
        del route_alt[p], begin_alt[p], wait_alt[p]
        for i in range(p, len(route_alt) - 1):
            wait_alt[i] = max(0, self.df['e'][route_alt[i]] - begin_alt[i-1] - self.a[route_alt[i-1], route_alt[i]])
            begin_alt[i] = begin_alt[i-1] + self.a[route_alt[i-1], route_alt[i]] + wait_alt[i]
        return route_alt, begin_alt, wait_alt