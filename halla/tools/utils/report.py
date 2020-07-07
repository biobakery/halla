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

def generate_lattice_plot(x_data, y_data, x_features, y_features, x_types, y_types, title, output_file,
                            figsize=(12, 12)):
    '''Generate and store lattice plot for each associationn, given:
    - {x,y}_data    : the data for features in {x,y} involved in the association in numpy
    - {x,y}_features: the names of the features in {x,y} involved in the association
    - output_file   : the output file name
    - title         : the plot title
    - {x,y}_types   : data types in {x,y}'s features, which determines scatterplot/boxplot/confusion matrix
    '''
    if (len(x_data) != len(x_features) and len(x_data) != len(x_types)) or \
        (len(y_data) != len(y_features) and len(y_data) != len(y_types)):
        raise ValueError('{x,y}_data should have the same length as {x,y}_features and {x,y}_types!')
    row_num = len(x_data) + len(y_data)
    fig, axs = plt.subplots(row_num, row_num, figsize=figsize)
    # combine all data
    all_data = np.concatenate((x_data, y_data), axis=0)
    all_features = list(x_features) + list(y_features)
    all_types = list(x_types) + list(y_types)
    for i in range(row_num):
        for j in range(row_num):
            if i < j:
                axs[i,j].axis('off')
                continue
            if i == j:
                # 1) plot a histogram if i == j
                sns.distplot(all_data[i], kde=False, ax=axs[i,j])
            elif all_types[i] == all_types[j]:
                if all_types[i] == object:
                    # 2) plot confusion matrix if both are categorical
                    conf_mat = np.zeros((max(all_data[i])+1, max(all_data[j])+1))
                    for k in range(all_data[i]):
                        conf_mat[all_data[i,k], all_data[j,k]] += 1
                    sns.heatmap(conf_mat, ax=axs[i,j], annot=True, cbar=False)
                else:
                    # 3) plot scatterplot if both are continuous
                    sns.scatterplot(x=all_data[j], y=all_data[i], ax=axs[i,j])
            else:
                # 4) plot boxplot if the data are mixed
                if all_types[j] == float:
                    sns.boxplot(x=all_data[j], y=all_data[i], orient='h', ax=axs[i,j])
                else:
                    sns.boxplot(x=all_data[j], y=all_data[i], ax=axs[i,j])
            # remove ticks from inner plots & add labels to side plots
            if j != 0:
                axs[i,j].set_yticks([])
            else:
                axs[i,j].set_ylabel(all_features[i], fontdict=dict(weight='bold'))
            if i != row_num - 1:
                axs[i,j].set_xticks([])
            else:
                axs[i,j].set_xlabel(all_features[j], fontdict=dict(weight='bold'))
    fig.suptitle(title)    
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(output_file)
    plt.close()