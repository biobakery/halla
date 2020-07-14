import numpy as np
import itertools
import functools

def is_densely_associated(block, fnr_thresh=0.1):
    '''Check if the significance block (boolean array) is densely associated, given the
       false negative rate threshold
    '''
    # ensure that block is a numpy boolean array
    if type(block) is not np.ndarray or block.dtype != 'bool':
        raise ValueError('block argument should be a boolean Numpy array!')
    return(np.sum(block) / block.size >= 1 - fnr_thresh)

def calc_gini_impurity(ar):
    '''Calculate gini impurity: 1 - proportion(True)^2 - proportion(False)^2
    Arg:
    - ar: a binary array (can be any dimension)
    '''
    # ensure that ar is a boolean array
    ar = np.asarray(ar)
    if ar.dtype != 'bool':
        raise ValueError('ar argument should be a boolean array!')
    prop_true  = np.sum(ar) / ar.size
    prop_false = 1 - prop_true
    return(1 - prop_true**2 - prop_false**2)

def calc_weighted_gini_impurity(lists):
    '''Calculate weighted gini impurity given an array of lists
    Arg:
    - lists: a list of binary arrays
    '''
    total_size = sum([len(list_i) for list_i in lists])
    weighted_scores = [len(list_i)*1.0/total_size * calc_gini_impurity(list_i) for list_i in lists]
    return(sum(weighted_scores))

def bifurcate(tree):
    '''Bifurcate a tree to its two branches
    - Arg: tree (ClusterNode object)
    Returns an array with length <=2 containing ClusterNode branches
      If tree is a leaf, it should return an empty array
    '''
    return([branch for branch in [tree.get_left(), tree.get_right()] if branch is not None])

def bifurcate_one(x_tree, y_tree, fdr_reject_table):
    '''Given two trees, bifurcate only one:
    - 1) if both are leaves, return (None, None)
    - 2) if one of them is a leaf, return the bifurcated non-leaf tree
    - 3) if both are non-leaves, bifurcate the one that produces lower gini impurity
    Args:
    - x_tree, y_tree   = ClusterNode objects
    '''
    x_branches = bifurcate(x_tree)
    y_branches = bifurcate(y_tree)
    # 1) if both are leaves
    if len(x_branches) == 0 and len(y_branches) == 0: return(None, None)
    # 2) if one of them is a leaf
    if len(x_branches) == 0:
        return([x_tree], y_branches)
    if len(y_branches) == 0:
        return(x_branches, [y_tree])
    # 3) if both are non-leaves
    x_features = x_tree.pre_order()
    y_features = y_tree.pre_order()
    # split1 = split x_tree
    split1_blocks = [[fdr_reject_table[feature_x, feature_y] for feature_x, feature_y in itertools.product(branch.pre_order(), y_features)]
                                                             for branch in x_branches]
    split1_gini = calc_weighted_gini_impurity(split1_blocks)
    # split2 = split y_tree
    split2_blocks = [[fdr_reject_table[feature_x, feature_y] for feature_x, feature_y in itertools.product(x_features, branch.pre_order())]
                                                             for branch in y_branches]
    split2_gini = calc_weighted_gini_impurity(split2_blocks)
    return((x_branches, [y_tree]) if split1_gini < split2_gini else ([x_tree], y_branches)) 

def compare_and_find_dense_block(X, Y, fdr_reject_table, fnr_thresh=0.1):
    '''Given another HierarchicalTree object Y, compare and find
    densely-associated block from the top of the hierarchy;

    Densely-associated block = (1 - FNR)% of pairwise association are
      FDR significant
    Args:
    - X               : X hierarchical tree (ClusterNode object)
    - Y               : Y hierarchical tree (ClusterNode object)
    - fdr_reject_table: a boolean table where True = reject H0
    - fnr_thresh      : false negative rate threshold
    '''

    def _check_iter_block(x_tree, y_tree):
        '''Check block iteratively until:
        - a densely-associated block is found
        - x_tree and y_tree are leaves
        Append densely-associated blocks to final_blocks array

        Terminate when:
        1) report block
        2) can no longer bifurcate
        '''
        X_features = x_tree.pre_order()
        Y_features = y_tree.pre_order()
        block_fdr_reject = fdr_reject_table[X_features,:][:,Y_features]
        if is_densely_associated(block_fdr_reject, fnr_thresh):
            # 1) terminate when the block is reported
            final_blocks.append([X_features, Y_features])
            return
        x_branches, y_branches = bifurcate_one(x_tree, y_tree, fdr_reject_table)
        if x_branches is None and y_branches is None:
            # 2) terminate when both trees can no longer bifurcate
            return
        for x_branch, y_branch in itertools.product(x_branches, y_branches):
            _check_iter_block(x_branch, y_branch)

    final_blocks = []
    _check_iter_block(X, Y)
    return(final_blocks)

def trim_block(block, fdr_reject_table):
    '''Trim the sides if all points on the side are insignificant
    # TODO: set a threshold?
    '''
    x_features, y_features = np.array(block[0]), np.array(block[1])
    x_start, x_end = 0, len(x_features)-1
    y_start, y_end = 0, len(y_features)-1
    while x_start < x_end:
        if fdr_reject_table[x_features[x_start],:][y_features].sum() > 0: break
        x_start += 1
    while x_end > x_start:
        if fdr_reject_table[x_features[x_end],:][y_features].sum() > 0: break
        x_end -= 1
    while y_start < y_end:
        if fdr_reject_table[:, y_features[y_start]][x_features].sum() > 0: break
        y_start += 1
    while y_end > y_start:
        if fdr_reject_table[:, y_features[y_end]][x_features].sum() > 0: break
        y_end -= 1
    return([[x_features[x] for x in range(x_start, x_end+1)], [y_features[y] for y in range(y_start, y_end+1)]])