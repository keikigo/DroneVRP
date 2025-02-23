import numpy as np
import pandas as pd

class LoadData:
    def __init__(self, config):
        self.path = f"./{config['folder']}/{config['file']}"
    
    def df_solomon(self):
        df = pd.read_csv(self.path, sep = '\s+', skiprows = 9,
            names = ['customer', 'x', 'y', 'q', 'e', 'l', 's'],
            index_col = 'customer',
            usecols = {'customer', 'x', 'y', 'q', 'e', 'l'})
        return df

def get_data(config):
    if config['dataset'] == 'solomon':
        df = LoadData(config).df_solomon()
        
    x_scale =  config['x_max'] / df[['x', 'y']].max().max()
    q_scale = config['q_max'] / df['q'].max()
    t_scale = config['t_max'] / df['l'].max()
    df['x'] = df['x'] * x_scale
    df['y'] = df['y'] * x_scale
    df['q'] = df['q'] * q_scale
    df['e'] = df['e'] * t_scale
    df['l'] = df['l'] * t_scale
    
    df = pd.concat([df.iloc[0].to_frame().T, 
                    df.iloc[1:].sample(config['n'], random_state = config['seed_sampling']),
                    df.iloc[0].to_frame().T], 
                    ignore_index = True)
    
    n = df.shape[0]
    mtrx = np.zeros((n, n), dtype = 'float')
    for i in range(n):
        for j in range(n):
            mtrx[i,j] = np.sqrt((df['x'][i] - df['x'][j]) ** 2 + (df['y'][i] - df['y'][j]) ** 2)
    return df, mtrx / config['v_t'], mtrx / config['v_d']