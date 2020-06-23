import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy.cluster.hierarchy as sch

def get_indices_map_dict(new_indices):
    return({ idx: i for i, idx in enumerate(new_indices) })

def generate_hallagram(significant_blocks, x_features, y_features, clust_x_idx, clust_y_idx, sim_table, cmap='RdBu_r', **kwargs):
    '''Plot hallagram given args:
    - significant blocks: a list of significant blocks in the original indices, e.g.,
                          [[[2], [0]], [[0,1], [1]]] --> two blocks
    - {x,y}_features    : feature names of {x,y}
    - clust_{x,y}_idx   : the indices of {x,y} in clustered form 
    - sim_table         : similarity table
    - cmap              : color map
    - kwargs            : other keyword arguments to be passed to seaborn's heatmap()
    '''
    clust_x_idx, clust_y_idx = np.asarray(clust_x_idx), np.asarray(clust_y_idx)
    # shuffle similarity table
    clust_sim_table = np.asarray(sim_table)[clust_x_idx,:][:,clust_y_idx]
    # shuffle features
    clust_x_features = np.asarray(x_features)[clust_x_idx]
    clust_y_features = np.asarray(y_features)[clust_y_idx]
    
    # create a dict to ease indices conversion
    x_ori2clust_idx = get_indices_map_dict(clust_x_idx)
    y_ori2clust_idx = get_indices_map_dict(clust_y_idx)

    vmax = np.abs(np.max(sim_table))
    vmin = -vmax

    # begin plotting
    fig = plt.figure()
    ax = sns.heatmap(clust_sim_table, xticklabels=clust_y_features, yticklabels=clust_x_features, cmap=cmap, vmin=vmin, vmax=vmax, **kwargs)
    
    for block in significant_blocks:
        x_block, y_block = block[0], block[1]
        clust_x_block = [x_ori2clust_idx[idx] for idx in x_block]
        clust_y_block = [y_ori2clust_idx[idx] for idx in y_block]
        ax.vlines([min(clust_y_block), max(clust_y_block)+1], min(clust_x_block), max(clust_x_block)+1)
        ax.hlines([min(clust_x_block), max(clust_x_block)+1], min(clust_y_block), max(clust_y_block)+1)
    plt.show()

def generate_clustermap(significant_blocks, x_features, y_features, x_linkage, y_linkage, sim_table, cmap='RdBu_r', **kwargs):
    '''Plot a clustermap given args:
    - significant blocks: a list of significant blocks in the original indices, e.g.,
                          [[[2], [0]], [[0,1], [1]]] --> two blocks
    - {x,y}_features    : feature names of {x,y}
    - {x,y}_linkage     : precomputed linkage matrix for {x,y}
    - sim_table         : similarity table
    - cmap              : color map
    - kwargs            : other keyword arguments to be passed to seaborn's clustermap()
    '''
    vmax = np.abs(np.max(sim_table))
    vmin = -vmax
    clustermap = sns.clustermap(sim_table, row_linkage=x_linkage, col_linkage=y_linkage, cmap='RdBu_r',
                        xticklabels=y_features, yticklabels=x_features, vmin=vmin, vmax=vmax, **kwargs)
    ax = clustermap.ax_heatmap
    x_ori2clust_idx = get_indices_map_dict(np.asarray(sch.to_tree(x_linkage).pre_order()))
    y_ori2clust_idx = get_indices_map_dict(np.asarray(sch.to_tree(y_linkage).pre_order()))

    for block in significant_blocks:
        x_block, y_block = block[0], block[1]
        clust_x_block = [x_ori2clust_idx[idx] for idx in x_block]
        clust_y_block = [y_ori2clust_idx[idx] for idx in y_block]
        ax.vlines([min(clust_y_block), max(clust_y_block)+1], min(clust_x_block), max(clust_x_block)+1)
        ax.hlines([min(clust_x_block), max(clust_x_block)+1], min(clust_y_block), max(clust_y_block)+1)
    plt.show()