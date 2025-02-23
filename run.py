from src.milp import Milp
from src.sa import SA

# for n in range(20, 101, 20):
for n in [100]:
    for n_d in [2]:
        # for q in [5,7.5,10]:
        for seed in range(1):
        # for tau in [20]:
            config = {
                'use_heuristic': False,
                'file': 'r101.txt',
                'folder': 'dataset',
                'name': 'r101',
                'dataset': 'solomon',
                'seed_sampling': 1000,
                'x_max': 8000, #m
                'q_max': 7.5, #kg
                't_max': 300, #min
                'v_t': 500, #m/min
                'v_d': 600, #m/min
                
                'fixed': False,
                'lambda': 0.8,
                'n': n,
                'n_d': n_d,
                'n_l': 2,
                'P': 4,
                'Q': 5,    #kg
                'alpha': 30,  #min
                'beta': 2,
                'tau': 20,
                
                
                'gap': 0.0001,
                'timelimit': 20,
                
                'seed':seed,
                'T_0': 1,
                'T_1': 0.01,
                'iter': 100,
                'r': 0.995,
                'patience': 5000,
                'epsilon': 0.001
                }
            config['n_t'] = round(config['n']/10)
            if config['use_heuristic']:
                # config['type'] = "SA"
                config['type'] = "SA_uni"
                # config['type'] = f"SA{int(10*config['q_max'])}"
                config['name'] = f"{config['type']}_{config['n']}_{config['n_d']}_{config['seed']}"
                P = SA(config)
                print(config['name'])
                P.solve()
            else:
                if config['fixed']:
                    # config['type'] = f"MILP{int(10*config['q_max'])}F"
                    config['type'] = f"M{config['tau']}F"
                else:
                    # config['type'] = f"MILP{int(10*config['q_max'])}F"
                    config['type'] = f"MILP"
                config['name'] = f"{config['type']}_{config['n']}_{config['n_d']}"
                P = Milp(config)
                print(config['name'])
                P.solve()