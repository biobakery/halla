from .config_loader import config, update_config
from .hierarchy import HierarchicalTree
from .logger import HAllALogger
from .utils.data import preprocess, eval_type, is_all_cont
from .utils.similarity import get_similarity_function
from .utils.stats import get_pvalue_table, pvalues2qvalues, test_pvalue_run_time
from .utils.tree import compare_and_find_dense_block, trim_block
from .utils.report import generate_hallagram, generate_clustermap, \
                          report_all_associations, report_significant_clusters, \
                          generate_lattice_plot
from .utils.filesystem import reset_dir

import pandas as pd
import numpy as np
import scipy.spatial.distance as spd
from os.path import join
import time

########
# AllA
########
class AllA(object):
    def __init__(self, max_freq_thresh=config.preprocess['max_freq_thresh'],
                 transform_data_funcs=config.preprocess['transform_funcs'],
                 discretize_bypass_if_possible=config.preprocess['discretize_bypass_if_possible'],
                 discretize_func=config.preprocess['discretize_func'], discretize_num_bins=config.preprocess['discretize_num_bins'],
                 pdist_metric=config.association['pdist_metric'],
                 permute_func=config.permute['func'], permute_iters=config.permute['iters'], permute_speedup=config.permute['speedup'],
                 fdr_alpha=config.stats['fdr_alpha'], fdr_method=config.stats['fdr_method'],
                 out_dir=config.output['dir'], verbose=config.output['verbose'], seed=0):
        # update AllA config setting
        update_config('output', dir=out_dir, verbose=verbose)
        update_config('preprocess', max_freq_thresh=max_freq_thresh,
                                    transform_funcs=transform_data_funcs,
                                    discretize_bypass_if_possible=discretize_bypass_if_possible,
                                    discretize_func=discretize_func, discretize_num_bins=discretize_num_bins)
        update_config('association', pdist_metric=pdist_metric)
        update_config('permute', func=permute_func, iters=permute_iters, speedup=permute_speedup)
        update_config('stats', fdr_alpha=fdr_alpha, fdr_method=fdr_method)
        self._reset_attributes()
        self.seed = seed
        if not hasattr(self, 'name'):
            self.name = 'AllA'
            self.logger = HAllALogger(name=self.name, config=config)

    '''Private functions
    '''
    def _reset_attributes(self):
        self.X, self.Y = None, None
        self.X_types, self.Y_types = None, None
        self.similarity_table = None
        self.pvalue_table, self.qvalue_table = None, None
        self.fdr_reject_table = None
        self.significant_blocks = None
        self.significant_blocks_qvalues = None
        self.has_loaded = False
        self.has_run = False

    def _compute_pairwise_similarities(self):
        dist_metric = config.association['pdist_metric']

        self.logger.log_step_start('Step 1: Computing pairwise similarities, p-values, and q-values', sub=True)
        start_time = time.time()

        X, Y = self.X.to_numpy(), self.Y.to_numpy()

        # obtain similarity matrix
        self.logger.log_message('Generating the similarity table...')
        self.similarity_table = spd.cdist(X, Y, metric=get_similarity_function(dist_metric))

        # obtain p-values
        self.logger.log_message('Generating the p-value table...')
        confp = config.permute
        extrapolated_time, timing_message = test_pvalue_run_time(X, Y, pdist_metric=dist_metric,
                                                   permute_func=confp['func'], permute_iters=confp['iters'],
                                                   permute_speedup=confp['speedup'],
                                                   alpha=config.stats['fdr_alpha'], seed=self.seed)
        if extrapolated_time > 10:
            self.logger.log_message(timing_message)
        self.pvalue_table = get_pvalue_table(X, Y, pdist_metric=dist_metric,
                                                   permute_func=confp['func'], permute_iters=confp['iters'],
                                                   permute_speedup=confp['speedup'],
                                                   alpha=config.stats['fdr_alpha'], seed=self.seed)

        # obtain q-values
        self.logger.log_message('Generating the q-value table...')
        self.fdr_reject_table, self.qvalue_table = pvalues2qvalues(self.pvalue_table.flatten(), config.stats['fdr_method'], config.stats['fdr_alpha'])
        self.qvalue_table = self.qvalue_table.reshape(self.pvalue_table.shape)
        self.fdr_reject_table = self.fdr_reject_table.reshape(self.pvalue_table.shape)

        end_time = time.time()
        self.logger.log_result('Number of significant associations', self.fdr_reject_table.sum())
        self.logger.log_step_end('Computing pairwise similarities, p-values, q-values', end_time - start_time, sub=True)

    def _find_dense_associated_blocks(self):
        '''Find significant cells based on FDR reject table
        '''
        def compare_qvalue(x):
            return(self.qvalue_table[x[0][0], x[1][0]])

        self.logger.log_step_start('Step 2: Finding densely associated blocks', sub=True)
        start_time = time.time()

        n, m = self.X.shape[0], self.Y.shape[0]
        self.significant_blocks = [[[x], [y]] for x in range(n) for y in range(m) if self.fdr_reject_table[x][y]]
        # sort by the p-values in ascending order
        self.significant_blocks.sort(key=compare_qvalue)
        self.significant_blocks_qvalues = [self.qvalue_table[x[0][0]][x[1][0]] for x in self.significant_blocks]

        end_time = time.time()
        self.logger.log_result('Number of significant clusters', len(self.significant_blocks))
        self.logger.log_step_end('Finding densely associated blocks', end_time - start_time, sub=True)

    def _generate_reports(self):
        '''Generate reports and store in config.output['dir'] directory:
        1) all_associations.txt: stores the associations between each feature in X and Y along with its
                                p-values and q-values in a table
        2) sig_clusters.txt    : stores only the significant clusters
        '''
        self.logger.log_step_start('Generating reports')

        # create directory
        dir_name = config.output['dir']
        reset_dir(dir_name, verbose=config.output['verbose'])

        # generate performance.txt
        self.logger.write_performance_log(dir_name, config)

        # generate all_associations.txt
        report_all_associations(dir_name,
                                self.X.index.to_numpy(),
                                self.Y.index.to_numpy(),
                                self.similarity_table,
                                self.pvalue_table,
                                self.qvalue_table)

        # generate sig_clusters.txt
        report_significant_clusters(dir_name,
                                    self.significant_blocks,
                                    self.significant_blocks_qvalues,
                                    self.X.index.to_numpy(),
                                    self.Y.index.to_numpy())

        # print datasets (original and discretized)
        self.X.to_csv(join(dir_name, 'X.tsv'), sep='\t')
        self.Y.to_csv(join(dir_name, 'Y.tsv'), sep='\t')
        self.X_ori.to_csv(join(dir_name, 'X_original.tsv'), sep='\t')
        self.Y_ori.to_csv(join(dir_name, 'Y_original.tsv'), sep='\t')

    '''Public functions
    '''
    def load(self, X_file, Y_file=None):
        def _read_and_drop_duplicated_indices(filepath):
            # drop duplicates and keep the first row
            df = pd.read_table(filepath, index_col=0)
            df = df[~df.index.duplicated(keep='first')]
            return(df)

        self.logger.log_step_start('Loading and preprocessing data')
        confp = config.preprocess

        start_time = time.time()

        X, self.X_types = eval_type(_read_and_drop_duplicated_indices(X_file))
        Y, self.Y_types = eval_type(_read_and_drop_duplicated_indices(Y_file)) if Y_file \
            else (X.copy(deep=True), np.copy(self.X_types))

        # if not all types are continuous but pdist_metric is only for continuous types
        if not (is_all_cont(self.X_types) and is_all_cont(self.Y_types)) and not (config.association['pdist_metric'].lower() in ['nmi','xicor']):
            raise ValueError('pdist_metric should be nmi or xicor if not all features are continuous...')
        # if pdist_metric is nmi but no discretization method is specified, assign to equal frequency (quantile)
        if config.association['pdist_metric'].lower() == 'nmi' and confp['discretize_func'] is None:
            self.logger.log_message('Discretization function is None; assigning to equal frequency (quantile) given metric = NMI...')
            update_config('preprocess', discretize_func='quantile')
        if config.association['pdist_metric'].lower() == 'xicor' and not (is_all_cont(self.X_types) and is_all_cont(self.Y_types)) and confp['discretize_func'] is None:
            self.logger.log_message('Discretization function is None but pdist_metric = XICOR and data contains categorical variables; assigning discretization function to equal frequency (quantile)...')
            update_config('preprocess', discretize_func='quantile')
        # if all features are continuous and distance metric != nmi, discretization can be bypassed
        if is_all_cont(self.X_types) and is_all_cont(self.X_types) and confp['discretize_func'] is not None and \
            config.association['pdist_metric'].lower() != 'nmi' and confp['discretize_bypass_if_possible']:
            self.logger.log_message('All features are continuous and bypassing discretization is enabled; bypassing discretization...')
            update_config('preprocess', discretize_func=None)

        # filter tables by intersect columns
        intersect_cols = [col for col in X.columns if col in Y.columns]
        X, Y = X[intersect_cols], Y[intersect_cols]

        # clean and preprocess data
        func_args = {
            'transform_funcs'    : confp['transform_funcs'],
            'max_freq_thresh'    : confp['max_freq_thresh'],
            'discretize_func'    : confp['discretize_func'],
            'discretize_num_bins': confp['discretize_num_bins']
        }
        self.X, self.X_ori, self.X_types = preprocess(X, self.X_types, **func_args)
        self.Y, self.Y_ori, self.Y_types = preprocess(Y, self.Y_types, **func_args)

        self.has_loaded = True
        end_time = time.time()

        self.logger.log_message('Preprocessing step completed:')
        self.logger.log_result('X shape (# features, # size)', self.X.shape)
        self.logger.log_result('Y shape (# features, # size)', self.Y.shape)
        self.logger.log_step_end('Loading and preprocessing data', end_time - start_time)

    def run(self):
        '''Run AllA:
        1) compute pairwise similarity matrix and p-values
        2) find significantly-associated cells
        '''
        if self.has_loaded == False:
            raise RuntimeError('load function has not been called!')

        self.logger.log_step_start('Performing %s' % self.name)

        # step 1: computing pairwise similarity matrix
        self._compute_pairwise_similarities()

        # step 2: find significantly-associated cells
        self._find_dense_associated_blocks()

        # generate reports
        self._generate_reports()

    def generate_hallagram(self, block_num=30, x_dataset_label='', y_dataset_label='',
                            cmap=None, cbar_label='', figsize=None, text_scale=10,
                            output_file='hallagram.png', mask=False, signif_dots=True, **kwargs):
        '''Generate a hallagram showing the top [block_num] significant blocks
        '''
        if cmap is None:
            cmap = 'YlGnBu' if config.association['pdist_metric'] in ['nmi', 'dcor', 'xicor'] else 'RdBu_r'
        file_name = join(config.output['dir'], output_file)
        if block_num is None:
            block_num = len(self.significant_blocks)
        else:
            block_num = min(block_num, len(self.significant_blocks))
        generate_hallagram(self.significant_blocks,
                           self.X.index.to_numpy(),
                           self.Y.index.to_numpy(),
                           [idx for idx in range(self.X.shape[0])],
                           [idx for idx in range(self.Y.shape[0])],
                           self.similarity_table,
                           self.fdr_reject_table,
                           x_dataset_label=x_dataset_label,
                           y_dataset_label=y_dataset_label,
                           figsize=figsize,
                           text_scale=text_scale,
                           output_file=file_name,
                           cmap=cmap, cbar_label=cbar_label,
                           mask=mask, signif_dots = signif_dots, **kwargs)

########
# HAllA
########
class HAllA(AllA):
    def __init__(self, max_freq_thresh=config.preprocess['max_freq_thresh'],
                 transform_data_funcs=config.preprocess['transform_funcs'],
                 discretize_bypass_if_possible=config.preprocess['discretize_bypass_if_possible'],
                 discretize_func=config.preprocess['discretize_func'], discretize_num_bins=config.preprocess['discretize_num_bins'],
                 pdist_metric=config.association['pdist_metric'], linkage_method=config.hierarchy['linkage_method'],
                 sim2dist_set_abs=config.hierarchy['sim2dist_set_abs'], sim2dist_func=config.hierarchy['sim2dist_func'],
                 permute_func=config.permute['func'], permute_iters=config.permute['iters'], permute_speedup=config.permute['speedup'],
                 fdr_alpha=config.stats['fdr_alpha'], fdr_method=config.stats['fdr_method'],
                 fnr_thresh=config.stats['fnr_thresh'], rank_cluster=config.stats['rank_cluster'],
                 out_dir=config.output['dir'], verbose=config.output['verbose'], seed=0):
        # TODO: add restrictions on the input - ensure the methods specified are available
        self.name = 'HAllA'
        # retrieve AllA variables
        alla_vars = vars()
        for key in ['linkage_method', 'fnr_thresh', 'rank_cluster', 'sim2dist_set_abs', 'sim2dist_func']:
            del alla_vars[key]
        # call AllA init function
        AllA.__init__(**alla_vars)

        # update HAllA config settings
        update_config('stats', fnr_thresh=fnr_thresh, rank_cluster=rank_cluster)
        update_config('hierarchy', linkage_method=linkage_method,
                                   sim2dist_set_abs=sim2dist_set_abs,
                                   sim2dist_func=sim2dist_func)
        self.logger = HAllALogger(self.name, config=config)

    '''Private functions
    '''
    def _reset_attributes(self):
        self.X, self.Y = None, None
        self.X_types, self.Y_types = None, None
        self.X_hierarchy, self.Y_hierarchy = None, None
        self.similarity_table = None
        self.pvalue_table, self.qvalue_table = None, None
        self.fdr_reject_table = None
        self.significant_blocks = None
        self.significant_blocks_qvalues = None
        self.has_loaded = False
        self.has_run = False

    def _run_clustering(self):
        self.logger.log_step_start('Step 2: Performing hierarchical clustering', sub=True)
        start_time = time.time()
        func_args = {
            'pdist_metric'    : config.association['pdist_metric'],
            'linkage_method'  : config.hierarchy['linkage_method'],
            'sim2dist_set_abs': config.hierarchy['sim2dist_set_abs'],
            'sim2dist_func'   : config.hierarchy['sim2dist_func']
        }
        self.X_hierarchy = HierarchicalTree(self.X, **func_args)
        self.Y_hierarchy = HierarchicalTree(self.Y, **func_args)
        end_time = time.time()
        self.logger.log_step_end('Performing hierarchical clustering', end_time - start_time, sub=True)

    def _find_dense_associated_blocks(self):
        def sort_by_best_qvalue(x):
            qvalue_table = self.qvalue_table[x[0],:][:,x[1]]
            return(qvalue_table.min())
        def sort_by_avg_qvalue(x):
            qvalue_table = self.qvalue_table[x[0],:][:,x[1]]
            return(qvalue_table.mean())

        self.logger.log_step_start('Step 3: Finding densely associated blocks', sub=True)
        start_time = time.time()
        self.significant_blocks = compare_and_find_dense_block(self.X_hierarchy.tree, self.Y_hierarchy.tree,
                                     self.fdr_reject_table, fnr_thresh=config.stats['fnr_thresh'])
        # sort significant blocks by the rank_cluster method
        sort_func = sort_by_best_qvalue if config.stats['rank_cluster'] == 'best' else sort_by_avg_qvalue
        self.significant_blocks.sort(key=sort_func)
        self.significant_blocks_qvalues = [sort_func(x) for x in self.significant_blocks]
        end_time = time.time()
        self.logger.log_result('Number of significant clusters', len(self.significant_blocks))
        self.logger.log_step_end('Finding densely associated blocks', end_time - start_time, sub=True)

    def _generate_reports(self):
        '''Generate reports and store in config.output['dir'] directory
        '''
        AllA._generate_reports(self)

        # store the linkages in .npy files
        dir_name = config.output['dir']
        np.save(join(dir_name, 'X_linkage.npy'), self.X_hierarchy.linkage, allow_pickle=False)
        np.save(join(dir_name, 'Y_linkage.npy'), self.Y_hierarchy.linkage, allow_pickle=False)

    '''Public functions
    '''
    def run(self):
        '''Run all 3 steps:
        1) compute pairwise similarity matrix
        2) cluster hierarchically
        3) find densely-associated blocks iteratively
        '''
        if self.has_loaded == False:
            raise RuntimeError('load function has not been called!')

        self.logger.log_step_start('Performing %s' % self.name)

        # step 1: computing pairwise similarity matrix
        self._compute_pairwise_similarities()

        # step 2: hierarchical clustering
        self._run_clustering()
        # step 3: iteratively finding densely-associated blocks
        self._find_dense_associated_blocks()

        # generate reports
        self._generate_reports()

    def generate_hallagram(self, block_num=30, x_dataset_label='', y_dataset_label='',
                            cmap=None, cbar_label='', figsize=None, text_scale=10,
                            output_file='hallagram.png', mask=False, signif_dots=True, **kwargs):
        '''Generate a hallagram showing the top [block_num] significant blocks
        '''
        if cmap is None:
            cmap = 'YlGnBu' if config.association['pdist_metric'] in ['nmi', 'dcor', 'xicor'] else 'RdBu_r'
        file_name = join(config.output['dir'], output_file)
        if block_num is None:
            block_num = len(self.significant_blocks)
        else:
            block_num = min(block_num, len(self.significant_blocks))
        generate_hallagram(self.significant_blocks,
                           self.X.index.to_numpy(),
                           self.Y.index.to_numpy(),
                           self.X_hierarchy.tree.pre_order(),
                           self.Y_hierarchy.tree.pre_order(),
                           self.similarity_table,
                           fdr_reject_table=self.fdr_reject_table,
                           x_dataset_label=x_dataset_label,
                           y_dataset_label=y_dataset_label,
                           figsize=figsize,
                           text_scale=text_scale,
                           output_file=file_name,
                           cmap=cmap, cbar_label=cbar_label,
                           mask=mask, signif_dots=signif_dots, block_num=block_num, **kwargs)

    def generate_clustermap(self, x_dataset_label='', y_dataset_label='',
                            cmap=None, cbar_label='', figsize=None, text_scale=10,
                            output_file='clustermap.png', mask=False, signif_dots=True, **kwargs):
        '''Generate a clustermap (hallagram + dendrogram)
        '''
        # if the dimension is too large, generate a hallagram instead
        if max(self.similarity_table.shape) > 500:
            print('The dimension is too large - please generate a hallagram instead.')
            return
        if cmap is None:
            cmap = 'YlGnBu' if config.association['pdist_metric'] in ['nmi', 'dcor', 'xicor'] else 'RdBu_r'

        file_name = join(config.output['dir'], output_file)
        generate_clustermap(self.significant_blocks,
                            self.X.index.to_numpy(),
                            self.Y.index.to_numpy(),
                            self.X_hierarchy.linkage,
                            self.Y_hierarchy.linkage,
                            self.similarity_table,
                            fdr_reject_table=self.fdr_reject_table,
                            x_dataset_label=x_dataset_label,
                            y_dataset_label=y_dataset_label,
                            figsize=figsize,
                            text_scale=text_scale,
                            cmap=cmap, cbar_label=cbar_label,
                            output_file=file_name,
                            mask=mask,
                            signif_dots=signif_dots,
                            **kwargs)

    def generate_diagnostic_plot(self, block_num=30, plot_dir='diagnostic', axis_stretch=1e-5, plot_size=4):
        '''Generate a lattice plot for each significant association;
        save all plots in the plot_dir folder under config.output['dir']
        '''
        # create the diagnostic directory under config.output['dir']
        reset_dir(join(config.output['dir'], plot_dir))
        if block_num is None:
            block_num = len(self.significant_blocks)
        else:
            block_num = min(block_num, len(self.significant_blocks))
        for i, block in enumerate(self.significant_blocks[:block_num]):
            title = 'Association %d' % (i+1)
            out_file = join(config.output['dir'], plot_dir, 'association_%d' % (i+1))
            x_data = self.X.to_numpy()[block[0],:]
            y_data = self.Y.to_numpy()[block[1],:]
            x_ori_data = self.X_ori.to_numpy()[block[0],:]
            y_ori_data = self.Y_ori.to_numpy()[block[1],:]
            x_features = self.X.index.to_numpy()[block[0]]
            y_features = self.Y.index.to_numpy()[block[1]]
            x_types = np.array(self.X_types)[block[0]]
            y_types = np.array(self.Y_types)[block[1]]
            generate_lattice_plot(x_data, y_data, x_ori_data, y_ori_data,
                                    x_features, y_features, x_types, y_types, title,
                                    out_file, axis_stretch=axis_stretch, plot_size=plot_size)
