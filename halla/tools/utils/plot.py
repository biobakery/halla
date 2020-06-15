import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def generate_hallagram(significant_blocks, x_features, y_features, clust_x_idx, clust_y_idx, sim_table):
    '''Plot hallagram given args:
    - significant blocks: a list of significant blocks using feature names, e.g.,
                          [[['X2'], ['Y0']], [['X0', 'X1'], ['Y1']]] --> two blocks
    - x_features        : feature names of x
    - y_features        : feature names of y
    - sim_table         : similarity table
    '''
    clust_x_idx, clust_y_idx = np.asarray(clust_x_idx), np.asarray(clust_y_idx)
    # shuffle similarity table
    clust_sim_table = np.asarray(sim_table)[clust_x_idx,:][:,clust_y_idx]
    # shuffle features
    clust_x_features = np.asarray(x_features)[clust_x_idx]
    clust_y_features = np.asarray(y_features)[clust_y_idx]
    
    # create a dict to ease indices conversion
    x_ori2clust_idx = { idx: i for i, idx in enumerate(clust_x_idx) }
    y_ori2clust_idx = { idx: i for i, idx in enumerate(clust_y_idx) }

    # begin plotting
    fig = plt.figure()
    ax = sns.heatmap(clust_sim_table, cmap='RdBu_r', xticklabels=clust_y_features, yticklabels=clust_x_features)
    
    for block in significant_blocks:
        x_block, y_block = block[0], block[1]
        clust_x_block = [x_ori2clust_idx[idx] for idx in x_block]
        clust_y_block = [y_ori2clust_idx[idx] for idx in y_block]
        ax.vlines([min(clust_y_block), max(clust_y_block)+1], min(clust_x_block), max(clust_x_block)+1)
        ax.hlines([min(clust_x_block), max(clust_x_block)+1], min(clust_y_block), max(clust_y_block)+1)
    plt.show()