import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy.cluster.hierarchy as sch
import pandas as pd
from os.path import join

def get_indices_map_dict(new_indices):
    return({ idx: i for i, idx in enumerate(new_indices) })

def generate_hallagram(significant_blocks, x_features, y_features, clust_x_idx, clust_y_idx, sim_table, cmap='RdBu_r', **kwargs):
    '''Plot hallagram given args:
    - significant blocks: a list of significant blocks in the original indices, e.g.,
                          [[[2], [0]], [[0,1], [1]]] --> two blocks
    - {x,y}_features    : feature names of {x,y}
    - clust_{x,y}_idx   : the indices of {x,y} in clustered form 
    - sim_table         : similarity table with size [len(x_features), len(y_features)]
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
    ax = sns.heatmap(clust_sim_table, xticklabels=clust_y_features, yticklabels=clust_x_features,
                        cmap=cmap, vmin=vmin, vmax=vmax, **kwargs)
    
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
    - sim_table         : similarity table with size [len(x_features), len(y_features)]
    - cmap              : color map
    - kwargs            : other keyword arguments to be passed to seaborn's clustermap()
    '''
    vmax = np.abs(np.max(sim_table))
    vmin = -vmax
    clustermap = sns.clustermap(sim_table, row_linkage=x_linkage, col_linkage=y_linkage, cmap=cmap,
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

def report_all_associations(dir_name, x_features, y_features, sim_table, pval_table, qval_table, output_file='all_associations.txt'):
    '''Store the association between each feature in X and Y along with p-values and q-values, given:
    - dir_name          : output directory name
    - {x,y}_features    : feature names of {x,y}
    - sim_table         : similarity table with size [len(x_features), len(y_features)]
    - pval_table        : pvalue table with size [len(x_features), len(y_features)]
    - qval_table        : qvalue table with size [len(x_features), len(y_features)]
    - output_file       : the output file name

    Store a .txt file that contains a table with column titles:
        'X_features', 'Y_features', 'association', 'p-values', 'q-values'
    '''
    filepath = join(dir_name, output_file)
    # initiate arrays for generating a pandas DataFrame later
    list_x_features, list_y_features = [], []
    list_association = []
    list_pvals, list_qvals = [], []
    for i in range(len(x_features)):
        for j in range(len(y_features)):
            list_x_features.append(x_features[i])
            list_y_features.append(y_features[j])
            list_association.append(sim_table[i][j])
            list_pvals.append(pval_table[i][j])
            list_qvals.append(qval_table[i][j])
    # create a pandas DataFrame
    df = pd.DataFrame(data={
        'X_features' : list_x_features,
        'Y_features' : list_y_features,
        'association': list_association,
        'p-values'   : list_pvals,
        'q-values'   : list_qvals,
    })
    # store into a file
    df.to_csv(filepath, sep='\t', index=False)

def report_significant_clusters(dir_name, significant_blocks, x_features, y_features, output_file='sig_clusters.txt'):
    '''Store only the significant clusters, given:
    - dir_name          : output directory name
    - significant_blocks: a list of significant blocks in the original indices, e.g.,
                          [[[2], [0]], [[0,1], [1]]] --> two blocks
    - {x,y}_features    : feature names of {x,y}
    - output_file       : the output file name

    # TODO: what are the scores to be stored?
    Store a .txt file that contains a table with column titles:
        'cluster_X', 'cluster_Y'
    '''
    filepath = join(dir_name, output_file)
    # initiate arrays for generating a pandas DataFrame later
    list_x_clust, list_y_clust = [], []
    for block in significant_blocks:
        list_x_clust.append(';'.join([x_features[idx] for idx in block[0]]))
        list_y_clust.append(';'.join([y_features[idx] for idx in block[1]]))
    # create a pandas DataFrame
    df = pd.DataFrame(data={
        'cluster_X': list_x_clust,
        'cluster_Y': list_y_clust
    })
    # store into a file
    df.to_csv(filepath, sep='\t', index=False)