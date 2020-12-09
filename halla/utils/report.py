import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import seaborn as sns
import numpy as np
import scipy.cluster.hierarchy as sch
import pandas as pd
from os.path import join
import itertools
from mpl_toolkits.axes_grid1 import make_axes_locatable
from copy import deepcopy

def get_indices_map_dict(new_indices):
    return({ idx: i for i, idx in enumerate(new_indices) })

def get_included_features(significant_blocks, num_x_features, num_y_features, trim=True, forced_x_idx=None, forced_y_idx=None):
    '''if trim is True, returns only the included features in X and Y. Optionally force certain features to be included.
    '''
    if trim:
        included_x_features, included_y_features = [], []
        for block in significant_blocks:
            included_x_features = included_x_features + block[0]
            included_y_features = included_y_features + block[1]
        for forced_x in forced_x_idx:
            included_x_features = included_x_features + forced_x_idx
        for forced_y in forced_y_idx:
            included_y_features = included_y_features + forced_y_idx
        included_x_features = sorted(list(set(included_x_features)))
        included_y_features = sorted(list(set(included_y_features)))
    else:
        included_x_features = [idx for idx in range(num_x_features)]
        included_y_features = [idx for idx in range(num_y_features)]
    return(included_x_features, included_y_features)

def remove_unshown_features(significant_blocks, shown_x, shown_y):
    '''
    Given a set of significant blocks and features that will be shown, go through each block and remove features that won't be shown

    '''
    shown_x_set = set(shown_x)
    shown_y_set = set(shown_y)
    signif_blocks = deepcopy(significant_blocks)
    has_deleted = np.zeros([len(signif_blocks),2],
                           dtype = bool)
    for block in range(len(signif_blocks)):
        if (not set(signif_blocks[block][0]).issubset(shown_x_set)):
            has_deleted[block,0] = True
        if (not set(signif_blocks[block][1]).issubset(shown_y_set)):
            has_deleted[block,1] = True
        signif_blocks[block][0] = list(set(signif_blocks[block][0]).intersection(shown_x_set))
        signif_blocks[block][1] = list(set(signif_blocks[block][1]).intersection(shown_y_set))

    to_delete = []
    for i in range(len(signif_blocks)):
        if (not signif_blocks[i][0] or not signif_blocks[i][1]) or (len(signif_blocks[i][0]) == 1 and len(signif_blocks[i][1]) == 1):
            to_delete.append(i)

    for i in sorted(to_delete, reverse=True):
        del signif_blocks[i]
        np.delete(has_deleted, i, axis = 0)

    return(signif_blocks, has_deleted)

def generate_hallagram(significant_blocks, x_features, y_features, clust_x_idx, clust_y_idx, sim_table, fdr_reject_table,
                        x_dataset_label='', y_dataset_label='', mask=False, trim=True,
                        signif_dots=True, block_num=30, show_lower=True, force_x_ft=None, force_y_ft=None, dpi=100,
                        figsize=None, cmap='RdBu_r', cbar_label='', text_scale=10, block_border_width=1.65, output_file='out.eps', **kwargs):
    '''Plot hallagram given args:
    - significant blocks: a list of *ranked* significant blocks in the original indices, e.g.,
                          [[[2], [0]], [[0,1], [1]]] --> two blocks
    - {x,y}_features     : feature names of {x,y}
    - clust_{x,y}_idx    : the indices of {x,y} in clustered form
    - sim_table          : similarity table with size [len(x_features), len(y_features)]
    - fdr_reject_table   : boolean association table with size [len(x_features), len(y_features)]
    - {x,y}_dataset_label: axis label
    - mask               : if True, mask all cells not included in significant blocks
    - trim               : if True, trim all features that are not significant
    - signif_dots        : if True, show dots on significant pairwise associations
    - block_num          : number indicating the top N blocks to include
    - show_lower         : if True, put a grey box around blocks ranked below the block_num threshold that show up on the hallagram anyway
    - figsize            : figure size
    - cmap               : color map
    - text_scale         : how much the rank text size should be scaled
    - block_border_width : the border width for all blocks
    - output_file        : file path to store the hallagram
    - kwargs             : other keyword arguments to be passed to seaborn's heatmap()
    '''
    #---data preparation---#

    if trim:
        top_blocks = significant_blocks[:block_num] # these are the first N blocks that MUST be shown & highlighted
    else:
        top_blocks = significant_blocks

    if len(top_blocks) == 0:
        print('The length of significant blocks is 0, no hallagram can be generated...')
        return

    if force_x_ft is None:
        force_x_ft = []
    if force_y_ft is None:
        force_y_ft = []

    if force_x_ft:
        forced_x_idx = [x_features.tolist().index(i) for i in force_x_ft]
    else:
        forced_x_idx = []
    if force_y_ft:
        forced_y_idx = [y_features.tolist().index(i) for i in force_y_ft]
    else:
        forced_y_idx = []

    included_x_feat, included_y_feat = get_included_features(top_blocks,
                                                             len(x_features),
                                                             len(y_features),
                                                             trim,
                                                             forced_x_idx,
                                                             forced_y_idx)
    if show_lower:
        lower_blocks, has_deleted = remove_unshown_features(significant_blocks[block_num:],
                                                            included_x_feat,
                                                            included_y_feat)

    # filter the indices with the included features
    clust_x_idx = np.asarray([i for i in clust_x_idx if i in included_x_feat])
    clust_y_idx = np.asarray([i for i in clust_y_idx if i in included_y_feat])

    # if mask, replace all insignificant cells to NAs
    clust_sim_table = np.copy(sim_table)
    clust_fdr_reject_table = np.copy(fdr_reject_table)
    if mask:
        dummy_table = np.full(clust_sim_table.shape, np.nan)
        dummy_fdr = np.full(clust_sim_table.shape, np.nan)
        for block in top_blocks:
            for x, y in itertools.product(block[0], block[1]):
                dummy_table[x,y] = clust_sim_table[x,y]
                dummy_fdr[x,y] = clust_fdr_reject_table[x,y]
        clust_sim_table = dummy_table
        clust_fdr_reject_table = dummy_fdr

    # shuffle similarity table and features accordingly
    clust_sim_table = clust_sim_table[clust_x_idx,:][:,clust_y_idx]
    clust_fdr_reject_table = clust_fdr_reject_table[clust_x_idx,:][:,clust_y_idx]
    clust_x_features = np.asarray(x_features)[clust_x_idx]
    clust_y_features = np.asarray(y_features)[clust_y_idx]

    # create a dict to ease indices conversion
    x_ori2clust_idx = get_indices_map_dict(clust_x_idx)
    y_ori2clust_idx = get_indices_map_dict(clust_y_idx)

    #---begin plotting---#
    # set up colormap anchor
    vmax, vmin = np.max(sim_table), np.min(sim_table)
    if vmin < 0 and vmax > 0:
        vmax = max(abs(vmin), vmax)
        vmin = -vmax

    sns.set_style('whitegrid')
    if figsize is None:
        # set each cell as ~0.3, then add margins
        figsize = (max(5, 0.3*clust_sim_table.shape[1]+4), max(5, 0.3*clust_sim_table.shape[0]+4))

    _, ax = plt.subplots(figsize=figsize)
    # add colorbar axes
    divider = make_axes_locatable(ax)
    cbar_ax = divider.append_axes('left', size=str(100/(figsize[0]*2.5))+'%', pad=1/(figsize[0]*6))
    cbar_kws = { 'label': cbar_label, 'ticklocation': 'left' }
    sns.heatmap(clust_sim_table, xticklabels = clust_y_features,
                        cmap=cmap, vmin=vmin, vmax=vmax, square=True,
                        zorder=3, cbar_ax=cbar_ax, ax=ax, cbar_kws=cbar_kws, **kwargs)
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position('right')
    ax.tick_params(axis = 'y', length=0)
    ax.set_yticklabels(clust_x_features, rotation=0, ha='left')
    if mask:
        # minor ticks
        ax.set_xticks(np.arange(0, clust_sim_table.shape[1], 1), minor=True)
        ax.set_yticks(np.arange(0, clust_sim_table.shape[0], 1), minor=True)
        ax.grid(which='minor', color='xkcd:light grey', zorder=0)
    ax.set_xlabel(y_dataset_label, fontweight='bold')
    ax.set_ylabel(x_dataset_label, fontweight='bold')

    if signif_dots:
        for i in range(len(clust_y_features)):
            for j in range(len(clust_x_features)):
                if clust_fdr_reject_table[j,i]:
                    ax.scatter(x = i+.5, y =j + .5, c = 'black', marker = "o", zorder = 3, s=25)
                    ax.scatter(x = i+.5, y =j + .5, c = 'white', marker = "o", zorder = 3, s=10)

    if show_lower:
        block_i = 0
        for rank, block in enumerate(lower_blocks):
            x_block, y_block = block[0], block[1]
            clust_x_block = [x_ori2clust_idx[idx] for idx in x_block]
            clust_y_block = [y_ori2clust_idx[idx] for idx in y_block]
            if has_deleted[block_i,0]:
                ax.vlines([min(clust_y_block)], min(clust_x_block), max(clust_x_block)+1, color='0.2', linewidths=block_border_width, zorder=4, alpha = .5, capstyle="projecting")
                ax.vlines([max(clust_y_block)+1], min(clust_x_block), max(clust_x_block)+1, color='0.2', linewidths=block_border_width, zorder=4, linestyles='dotted', alpha = .5, capstyle="projecting")
            else:
                ax.vlines([min(clust_y_block), max(clust_y_block)+1], min(clust_x_block), max(clust_x_block)+1, color='0.2', linewidths=block_border_width, zorder=4, alpha = .5, capstyle="projecting")

            if has_deleted[block_i,1]:
                ax.hlines([min(clust_x_block)], min(clust_y_block), max(clust_y_block)+1, color='0.2', linewidths=block_border_width, zorder=4, alpha = .5, capstyle="projecting")
                ax.hlines([max(clust_x_block)+1], min(clust_y_block), max(clust_y_block)+1, color='0.2', linewidths=block_border_width, zorder=4, linestyles='dotted', alpha = .5, capstyle="projecting")
            else:
                ax.hlines([min(clust_x_block), max(clust_x_block)+1], min(clust_y_block), max(clust_y_block)+1, color='0.2', linewidths=block_border_width, zorder=4, alpha = .5, capstyle="projecting")
            block_i += 1

    for rank, block in enumerate(top_blocks):
        x_block, y_block = block[0], block[1]
        clust_x_block = [x_ori2clust_idx[idx] for idx in x_block]
        clust_y_block = [y_ori2clust_idx[idx] for idx in y_block]
        ax.vlines([min(clust_y_block), max(clust_y_block)+1], min(clust_x_block), max(clust_x_block)+1, color='black', linewidths=block_border_width, zorder=4, capstyle="projecting")
        ax.hlines([min(clust_x_block), max(clust_x_block)+1], min(clust_y_block), max(clust_y_block)+1, color='black', linewidths=block_border_width, zorder=4, capstyle="projecting")
        # add rank text
        text_content = str(rank + 1)
        text_size = (min(max(clust_y_block) - min(clust_y_block),
                         max(clust_x_block) - min(clust_x_block)) + 1)*text_scale
        text = ax.text(
            np.mean(clust_y_block) + 0.5, np.mean(clust_x_block) + 0.5,
            text_content, size=text_size, color='white', ha='center', va='center', weight='bold')
        text.set_path_effects([
            path_effects.Stroke(linewidth=3, foreground='black'),
            path_effects.Normal(),
        ])

    plt.subplots_adjust(wspace=1/(figsize[0]*6), hspace=0)
    plt.savefig(output_file, format=output_file.split('.')[-1].lower(), bbox_inches='tight', dpi=dpi)

def generate_clustermap(significant_blocks, x_features, y_features, x_linkage, y_linkage, sim_table, fdr_reject_table,
                        x_dataset_label='', y_dataset_label='', signif_dots=True, figsize=None, cmap='RdBu_r', text_scale=10,
                        dendrogram_ratio=None, cbar_label='',
                        block_border_width=1.5, mask=False, output_file='out.png', **kwargs):
    '''Plot a clustermap given args:
    - significant blocks: a list of *ranked* significant blocks in the original indices, e.g.,
                          [[[2], [0]], [[0,1], [1]]] --> two blocks
    - {x,y}_features     : feature names of {x,y}
    - {x,y}_linkage      : precomputed linkage matrix for {x,y}
    - sim_table          : similarity table with size [len(x_features), len(y_features)]
    - fdr_reject_table   : boolean association table with size [len(x_features), len(y_features)]
    - {x,y}_dataset_label: axis label
    - signif_dots        : if True, show dots on significant pairwise associations
    - figsize            : figure size
    - cmap               : color map
    - text_scale         : how much the rank text size should be scaled
    - block_border_width : the border width for all blocks
    - mask               : if True, mask all cells not included in significant blocks
    - output_file        : file path to store the hallagram
    - kwargs             : other keyword arguments to be passed to seaborn's clustermap()
    '''
    vmax, vmin = np.max(sim_table), np.min(sim_table)
    if vmin < 0 and vmax > 0:
        vmax = max(abs(vmin), vmax)
        vmin = -vmax
    mask_ar = None
    if mask:
        sns.set_style('white')
        mask_ar = np.full(sim_table.shape, True, dtype=bool)
        for block in significant_blocks:
            for i, j in itertools.product(block[0], block[1]):
                mask_ar[i][j] = False
    if figsize is None:
        figsize = (max(5, 0.3*sim_table.shape[1]+2.5), max(5, 0.3*sim_table.shape[0]+4))
    if dendrogram_ratio is None:
        dendrogram_ratio = (0.8/figsize[0], 0.8/figsize[1])
    cbar_pos = (0, 0.8, 0.1/figsize[0], 0.18)
    cbar_kws = { 'label': cbar_label, 'ticklocation': 'left' }
    clustermap = sns.clustermap(sim_table, row_linkage=x_linkage, col_linkage=y_linkage, cmap=cmap, mask=mask_ar, zorder=2,
                        xticklabels=y_features, yticklabels=x_features, vmin=vmin, vmax=vmax, cbar_pos=cbar_pos, cbar_kws=cbar_kws,
                        figsize=figsize, dendrogram_ratio=dendrogram_ratio, **kwargs)
    ax = clustermap.ax_heatmap
    ax.set_xlabel(y_dataset_label, fontweight='bold')
    ax.set_ylabel(x_dataset_label, fontweight='bold')
    if mask:
        # minor ticks
        ax.set_xticks(np.arange(0, sim_table.shape[1], 1), minor=True)
        ax.set_yticks(np.arange(0, sim_table.shape[0], 1), minor=True)
        ax.grid(which='minor', color='xkcd:light grey', zorder=0)
    x_ori2clust_idx = get_indices_map_dict(np.asarray(sch.to_tree(x_linkage).pre_order()))
    y_ori2clust_idx = get_indices_map_dict(np.asarray(sch.to_tree(y_linkage).pre_order()))

    dot_order_x = np.asarray(clustermap.dendrogram_row.reordered_ind)
    dot_order_y = clustermap.dendrogram_col.reordered_ind

    if signif_dots:
        for i in range(len(x_features)):
            for j in range(len(y_features)):
                if fdr_reject_table[dot_order_x[i],dot_order_y[j]]:
                    ax.scatter(y = i + .5, x = j + .5, c = 'black', marker = "o", zorder = 3, s = 25)
                    ax.scatter(y = i + .5, x = j + .5, c = 'white', marker = "o", zorder = 3, s = 10)

    for rank, block in enumerate(significant_blocks):
        x_block, y_block = block[0], block[1]
        clust_x_block = [x_ori2clust_idx[idx] for idx in x_block]
        clust_y_block = [y_ori2clust_idx[idx] for idx in y_block]
        ax.vlines([min(clust_y_block), max(clust_y_block)+1], min(clust_x_block), max(clust_x_block)+1, color='black', linewidths=block_border_width, zorder=3)
        ax.hlines([min(clust_x_block), max(clust_x_block)+1], min(clust_y_block), max(clust_y_block)+1, color='black', linewidths=block_border_width, zorder=3)
        # add rank text
        text_content = str(rank + 1)
        text_size = (min(max(clust_y_block) - min(clust_y_block),
                         max(clust_x_block) - min(clust_x_block)) + 1)*text_scale
        text = ax.text(
            np.mean(clust_y_block) + 0.5, np.mean(clust_x_block) + 0.5,
            text_content, size=text_size, color='white', ha='center', va='center', weight='bold')
        text.set_path_effects([
            path_effects.Stroke(linewidth=3, foreground='black'),
            path_effects.Normal(),
        ])
    plt.savefig(output_file, format=output_file.split('.')[-1].lower(), bbox_inches='tight')

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

def report_significant_clusters(dir_name, significant_blocks, scores, x_features, y_features,
                                score_label='best_adjusted_pvalue', output_file='sig_clusters.txt'):
    '''Store only the significant clusters, given:
    - dir_name          : output directory name
    - significant_blocks: a list of significant blocks in the original indices, e.g.,
                          [[[2], [0]], [[0,1], [1]]] --> two blocks
    - scores            : a list of scores for each significant blocks
    - {x,y}_features    : feature names of {x,y}
    - output_file       : the output file name
    Store a .txt file that contains a table with column titles:
        'cluster_X', 'cluster_Y'
    '''
    score_label = score_label.replace(' ', '_')
    filepath = join(dir_name, output_file)
    # initiate arrays for generating a pandas DataFrame later
    list_x_clust, list_y_clust = [], []
    for block in significant_blocks:
        list_x_clust.append(';'.join([x_features[idx] for idx in block[0]]))
        list_y_clust.append(';'.join([y_features[idx] for idx in block[1]]))
    # create a pandas DataFrame
    df = pd.DataFrame(data={
        'cluster_rank': [i+1 for i in range(len(significant_blocks))],
        'cluster_X'   : list_x_clust,
        'cluster_Y'   : list_y_clust,
        score_label   : scores
    })
    # store into a file
    df.to_csv(filepath, sep='\t', index=False)

def generate_lattice_plot(x_data, y_data, x_ori_data, y_ori_data, x_features, y_features, x_types, y_types,
                            title, output_file, axis_stretch=1e-5, plot_size=4):
    '''Generate and store lattice plot for each associationn, given:
    - {x,y}_data    : the data for features in {x,y} involved in the association in numpy (the discretized data if discretized)
    - {x,y}_ori_data: the original data for features in {x,y} involved in the association in numpy
    - {x,y}_features: the names of the features in {x,y} involved in the association
    - {x,y}_types   : data types in {x,y}'s features, which determines scatterplot/boxplot/confusion matrix
    - title         : the plot title
    - output_file   : the output file name
    - axis_stretch  : stretch both axes of continuous data by the value, e.g., x --> [x_min - axis_stretch, x_max + axis_stretch]
    - plot_size     : the size of each plot - the figsize would be (# features*plot_size, # features*plot_size)
    '''
    if (len(x_data) != len(x_features) and len(x_data) != len(x_types)) or \
        (len(y_data) != len(y_features) and len(y_data) != len(y_types)):
        raise ValueError('{x,y}_data should have the same length as {x,y}_features and {x,y}_types!')
    row_num = len(x_data) + len(y_data)
    sns.set_style('white')
    fig, axs = plt.subplots(row_num, row_num, figsize=(row_num*plot_size, row_num*plot_size))
    # combine all data
    all_data = np.concatenate((x_data, y_data), axis=0)
    all_ori_data = np.concatenate((x_ori_data, y_ori_data), axis=0)
    all_features = list(x_features) + list(y_features)
    all_types = list(x_types) + list(y_types)
    for i in range(row_num):
        for j in range(row_num):
            if i < j:
                axs[i,j].axis('off')
                continue
            if i == j:
                # 1) plot a histogram if i == j
                if all_types[i] == float:
                    # continuous data: show CDF of the original data
                    # - if discretized, show borders between categories
                    y_min, y_max = 0.0, 1.1
                    x_min, x_max = np.nanmin(all_ori_data[i]) - axis_stretch, np.nanmax(all_ori_data[i]) + axis_stretch
                    sorted_data = np.sort(np.concatenate(([x_min], np.unique(all_ori_data[i]), [x_max])))
                    cdf_line = [(all_ori_data[i] <= val).sum()/len(all_ori_data[i]) for val in sorted_data]
                    sns.lineplot(x=sorted_data, y=cdf_line, ax=axs[i,j], zorder=1)
                    try:
                        np.testing.assert_equal(all_ori_data[i], all_data[i])
                    except: # discretized
                        ori_data, disc_data = np.array(all_ori_data[i]), np.array(all_data[i])
                        border_x = [np.nanmax(ori_data[disc_data ==  x]) for x in range(int(disc_data.min()), int(disc_data.max()))]
                        border_y = [len(ori_data[disc_data <=  x])*1.0/len(ori_data) for x in range(int(disc_data.min()), int(disc_data.max()))]
                        # normalize border_x
                        border_x_norm = [(x - x_min) / (x_max - x_min) for x in border_x]
                        border_y_norm = [(y - y_min) / (y_max - y_min) for y in border_y]
                        for k in range(len(border_x)):
                            axs[i,j].axvline(x=border_x[k], ymax=border_y_norm[k], color='k', ls='--', zorder=2)
                            axs[i,j].axhline(y=border_y[k], xmin=border_x_norm[k], color='k', ls='--', zorder=2)
                    axs[i,j].set_xlim(x_min, x_max)
                    axs[i,j].set_ylim(y_min, y_max)
                else:
                    # categorical data: bins should not be set by default
                    sns.countplot(x=all_data[i], ax=axs[i,j])
            elif all_types[i] == all_types[j]:
                if all_types[i] == object:
                    # 2) plot confusion matrix if both are categorical
                    conf_mat = np.zeros((max(all_data[i])+1, max(all_data[j])+1))
                    for k in range(len(all_data[i])):
                        conf_mat[all_data[i,k], all_data[j,k]] += 1
                    sns.heatmap(conf_mat, ax=axs[i,j], annot=True, cbar=False, cmap='Blues', linewidths=0.1, linecolor='gray')
                else:
                    # 3) plot scatterplot if both are continuous
                    sns.scatterplot(x=all_ori_data[j], y=all_ori_data[i], ax=axs[i,j])
                    axs[i,j].set_xlim(np.nanmin(all_ori_data[j]) - axis_stretch, np.nanmax(all_ori_data[j]) + axis_stretch)
                    axs[i,j].set_ylim(np.nanmin(all_ori_data[i]) - axis_stretch, np.nanmax(all_ori_data[i]) + axis_stretch)
            else:
                # 4) plot boxplot if the data are mixed
                if all_types[j] == float:
                    sns.boxplot(x=all_ori_data[j], y=all_data[i], orient='h', color='w', ax=axs[i,j])
                    sns.stripplot(x=all_ori_data[j], y=all_data[i], jitter=0.25, orient='h', ax=axs[i,j])
                    axs[i,j].set_xlim(np.nanmin(all_ori_data[j]) - axis_stretch, np.nanmax(all_ori_data[j]) + axis_stretch)
                else:
                    sns.boxplot(x=all_data[j], y=all_ori_data[i], color='w', ax=axs[i,j])
                    sns.stripplot(x=all_data[j], y=all_ori_data[i], jitter=0.25, ax=axs[i,j])
                    axs[i,j].set_ylim(np.nanmin(all_ori_data[i]) - axis_stretch, np.nanmax(all_ori_data[i]) + axis_stretch)
            # add y-ticks on the right side on histogram plots
            if i == j:
                axs[i,j].yaxis.tick_right()
                if i != row_num - 1: axs[i,j].set_xticks([])
                axs[i,j].set_ylabel('')
            else:
                # remove ticks from inner plots
                if j != 0: axs[i,j].set_yticks([])
                if i != row_num - 1: axs[i,j].set_xticks([])
            # add labels to outer  plots
            if j == 0: axs[i,j].set_ylabel(all_features[i], fontdict=dict(weight='bold'))
            if i == row_num - 1: axs[i,j].set_xlabel(all_features[j], fontdict=dict(weight='bold'))
            for spine in axs[i,j].spines:
                axs[i,j].spines[spine].set_visible(True)
    # align labels
    fig.align_xlabels()
    fig.align_ylabels()
    fig.suptitle(title)
    plt.subplots_adjust(wspace=.05, hspace=.05)
    plt.savefig(output_file)
    plt.close()
