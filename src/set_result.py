import os
import csv
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')

def get_result(config, x, y, t1, t2, s):
    routes_t = []
    routes_d = []
    begins_t = []
    begins_d = []
    waits_t = []
    arcs_t = {} 
    for v, i, j in x.keys():
        if x[v, i, j].X > 0.5 and (i, j) != (0, config['n']+1):
            arcs_t[v, i] = j
    arcs_d = {} 
    for w, k, i, j in y.keys():
        if y[w, k, i, j].X > 0.5 and (i, j) != (0, config['n']+1):
            arcs_d[w, k, i] = j
    for v in range(1, config['n_t'] + 1):
        if (v, 0) in arcs_t.keys():
            start = 0
            routes_t.append([0])
            begins_t.append([t1[v, 0].X])
            waits_t.append([s[v, 0].X])
            while start != config['n']+1:
                start = arcs_t[v, start]
                routes_t[-1].append(start)
                begins_t[-1].append(t1[v, start].X)
                waits_t[-1].append(s[v, start].X)
    for w in range(1, config['n_d'] + 1):
        routes_d.append([])
        begins_d.append([])
        for k in range(1, config['n_l'] + 1):
            if (w, k, 0) in arcs_d.keys():
                start = 0
                routes_d[-1].append([0])
                begins_d[-1].append([t2[w, k, 0].X])
                while start != config['n']+1:
                    start = arcs_d[w, k, start]
                    routes_d[-1][-1].append(start)
                    begins_d[-1][-1].append(t2[w, k, start].X)
        if routes_d[-1] == []:
            del routes_d[-1]
            del begins_d[-1]
        else:
            route = []
            begin = []
            for i in range(len(routes_d[-1])):
                route += routes_d[-1][i][:-1]
                begin += begins_d[-1][i][:-1]
            route += [routes_d[-1][i][-1]]
            begin += [begins_d[-1][i][-1]]
            routes_d[-1] = route
            begins_d[-1] = begin
    return routes_t, routes_d, begins_t, begins_d, waits_t

def save_result(config, result):
    os.makedirs('./results', exist_ok=True)
    path = os.path.join('./results', f"{config['type']}.csv")
    if not os.path.exists(path):
        with open(path, mode = 'w', newline = '', encoding = 'utf-8') as file:
            if config['use_heuristic']:
                csv.writer(file).writerow(['dataset', 'seed','n', 'n_d','iter', 'time', 'obj'])
            else:
                csv.writer(file).writerow(['dataset', 'seed', 'n', 'n_d', 'time', 'obj', 'gap'])
    with open(path, mode = 'a', newline = '', encoding = 'utf-8') as file:
            if config['use_heuristic']:
                csv.writer(file).writerow([config['file'], config['seed'], config['n'], config['n_d'],
                    result[0], round(result[1], 3), round(result[2], 3)])
            else:
                csv.writer(file).writerow([config['file'], config['seed'], config['n'], config['n_d'],
                    round(result[0], 3), round(result[1], 3), round(100*result[2], 3)])
    return

def visual(config, df, route_t, route_d, name = ''):
    path = './plots'
    os.makedirs(path, exist_ok=True)
    path += f"/{config['name']}_{name}.png"
    colors = sns.color_palette()
    fig, ax = plt.subplots(figsize = (8,8))
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    index = 0
    for i in range(len(route_t)):
        route = route_t[i]
        if len(route) > 2:
            for j in range(len(route) - 1):
                ax.annotate("", xytext = (df['x'][route[j]], df['y'][route[j]]), 
                            xy = (df['x'][route[j + 1]], df['y'][route[j + 1]]),
                            arrowprops = dict(edgecolor = colors[index], arrowstyle = "-", linestyle = ':'))
            ax.text((df['x'][route[0]] + df['x'][route[1]]) * 0.5,
                    (df['y'][route[0]] + df['y'][route[1]]) * 0.5,
                    'Truck' + str(i + 1), color = colors[index])
            index += 1
    for i in range(len(route_d)):
        route = route_d[i]
        if len(route) > config['n_l'] + 1:
            for j in range(len(route) - 1):
                ax.annotate("", xytext = (df['x'][route[j]], df['y'][route[j]]), 
                            xy = (df['x'][route[j + 1]], df['y'][route[j + 1]]),
                            arrowprops = dict(edgecolor = colors[index], arrowstyle = "-"))
            ax.text((df['x'][route[0]] + df['x'][route[1]]) * 0.5,
                    (df['y'][route[0]] + df['y'][route[1]]) * 0.5,
                    'Drone' + str(i + 1), color = colors[index])
            index += 1
        
    ax.scatter(df['x'][1:-1], df['y'][1:-1], s = 60)
    ax.scatter(df['x'][0], df['y'][0], marker = 's', color = colors[0], s = 60) 
    for i in df.index[:-1]:
        ax.annotate(i, (df['x'][i], df['y'][i]), fontsize = 15)
    ax.set_xticks([])
    ax.set_yticks([])
    # plt.title
    fig.savefig(path, bbox_inches = 'tight', dpi = 200)
    print(path)
    plt.close()
    return