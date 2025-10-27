import pandas as pd
import numpy as np

def zero_adj(data, save_dir = None):
    '''
    Adjust data relative to initial baseline values for balance trainer assessment.
    
    Parameters:
        data (pd.DataFrame): Raw data loaded from CSV.
        save_dir (str, optional): Directory to save the adjusted data CSV. Defaults to None
        
    Returns:
        pd.DataFrame: Zero-adjusted data with calculated deltas and column labels.
    '''
    z_adj = [[0.0] for _ in range(17)] + [[0], [0]]
    tmp_ = data.iloc[14:3014]
    standard = tmp_.iloc[0][0].split('\t')  # ['Time[ms]', 'open_X[mm]', 'open_Y[mm]', 'close_X[mm]', 'close_Y[mm]']
    for i in range(1, 3000):
        tmp = tmp_.iloc[i][0].split('\t')
        z_adj[0].append(float(tmp[0]))
        for j in range(1, 5):
            z_adj[j].append(float(tmp[j]) - float(standard[j]))
            z_adj[j + 4].append(abs(float(z_adj[j][-1]) - float(z_adj[j][-2])))
    for i in range(1, len(z_adj[0])):
        z_adj[9].append(np.sqrt(z_adj[5][i]**2 + z_adj[6][i]**2))
        z_adj[10].append(np.sqrt(z_adj[7][i]**2 + z_adj[8][i]**2))
    tmp_ = data.iloc[3016:6016]
    standard = tmp_.iloc[0][0].split('\t')  # ['open_LU_weight[kg]', ...]
    for i in range(1, 3000):
        tmp = tmp_.iloc[i][0].split('\t')
        for j in range(4):
            z_adj[j + 11].append(float(tmp[j]) - float(standard[j]))
    tmp_ = data.iloc[6017:9017]
    standard = tmp_.iloc[0][0].split('\t')  # ['close_LU_weight[kg]', ...]
    for i in range(1, 3000):
        tmp = tmp_.iloc[i][0].split('\t')
        for j in range(4):
            z_adj[j + 15].append(float(tmp[j]) - float(standard[j]))
    z_adj = pd.DataFrame(z_adj).T
    z_adj.columns = ['Time[ms]', 'open_X[mm]', 'open_Y[mm]', 'close_X[mm]', 'close_Y[mm]',
                    'open_delta_X[mm]', 'open_delta_Y[mm]',
                    'close_delta_X[mm]', 'close_delta_Y[mm]',
                    'open_delta_distance', 'close_delta_distance',
                    'open_LU_weight[kg]', 'open_LD_weight[kg]',
                    'open_RU_weight[kg]', 'open_RD_weight[kg]',
                    'close_LU_weight[kg]', 'close_LD_weight[kg]',
                    'close_RU_weight[kg]', 'close_RD_weight[kg]']
    
    if save_dir:
        z_adj.to_csv(save_dir, index=False)
        
    return z_adj
