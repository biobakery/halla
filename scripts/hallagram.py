'''Generate hallagram on console environment
'''

import argparse
import sys
import numpy as np
from os.path import join
import pkg_resources

from halla.utils.report import generate_hallagram, generate_clustermap
from .loader import HAllAPartialLoader

def parse_argument(args):
    parser = argparse.ArgumentParser(
        description='HAllA clustermap/hallagram generator')

    # --load parameters--
    parser.add_argument(
        '-i', '--input',
        help='Path to the output directory',
        required=True)
    parser.add_argument(
        '-c', '--clustermap',
        help='Generate a clustermap instead of a hallagram',
        action='store_true', required=False)
    parser.add_argument(
        '--x_dataset_label',
        help='Label for X dataset',
        default='', required=False)
    parser.add_argument(
        '--y_dataset_label',
        help='Label for Y dataset',
        default='', required=False)
    parser.add_argument(
        '--cbar_label',
        help='Label for the colorbar',
        default='', required=False)
    parser.add_argument(
        '--cmap',
        help='Colormap',
        default='RdBu_r', required=False)
    parser.add_argument(
        '-n', '--block_num',
        help='Number of top clusters to show (for hallagram only); if -1, show all clusters',
        default=30, type=int, required=False)
    parser.add_argument(
        '--mask',
        help='Mask the hallagram/clustermap',
        action='store_true', required=False)
    parser.add_argument(
        '--trim',
        help='Trim hallagram to features containing at least one significant block',
        default=True,
        type=bool, required=False)
    parser.add_argument(
        '--text_scale',
        help='Significant cluster text scale',
        default=10, type=float, required=False)
    parser.add_argument(
        '--block_border_width',
        help='Significant cluster border width',
        default=1.5, type=float, required=False)
    parser.add_argument(
        '-o', '--output',
        help='Path to output file under the HAllA/AllA result directory; default: hallagram.png or clustermap.png',
        default='', required=False)
    parser.add_argument(
        '--fdr_alpha',
        help='FDR threshold',
        default=0.05, type=float, required=False)
    parser.add_argument(
        '--show_lower',
        help='Show multi-member blocks ranked below block_num as grey outlined boxes',
        default=True, type=bool, required=False)
    parser.add_argument(
        '--force_x_ft',
        nargs = '+',
        help='Force specific x features to be included, irrespective of trim setting.',
        default=None,
        required=False)
    parser.add_argument(
        '--force_y_ft',
        nargs = '+',
        help='Force specific y features to be included, irrespective of trim setting.',
        default=None,
        required=False)
    parser.add_argument(
        '--dpi',
        help='Figure DPI',
        default=100, type=float, required=False)

    params = parser.parse_args()
    if params.block_num == -1: params.block_num = None
    if params.output == '':
        params.output = 'clustermap.png' if params.clustermap else 'hallagram.png'
    return(params)

def main():
    params = parse_argument(sys.argv)
    input_dir = params.input
    output_file = join(input_dir, params.output)
    loader = HAllAPartialLoader(input_dir)
    if params.clustermap:
        if loader.name == 'AllA':
            raise ValueError('Cannot generate a clustermap from AllA result.')
        if max(loader.sim_table.shape) > 500:
            raise ValueError('The dimension is too large - please generate a hallagram instead.')
        generate_clustermap(loader.significant_blocks,
                            loader.X_features,
                            loader.Y_features,
                            loader.X_linkage,
                            loader.Y_linkage,
                            loader.sim_table,
                            x_dataset_label=params.x_dataset_label,
                            y_dataset_label=params.y_dataset_label,
                            fdr_reject_table = loader.fdr_reject_table,
                            cbar_label=params.cbar_label,
                            text_scale=params.text_scale,
                            block_border_width=params.block_border_width,
                            mask=params.mask,
                            cmap=params.cmap,
                            output_file=output_file)
    else:
        block_num = params.block_num
        if block_num is None:
            block_num = len(loader.significant_blocks)
        else:
            block_num = min(block_num, len(loader.significant_blocks))
        if block_num > 500:
            raise ValueError('The number of blocks to show is too large, please input block # < 300')
        generate_hallagram(loader.significant_blocks,
                           loader.X_features,
                           loader.Y_features,
                           loader.X_tree.pre_order() if loader.name == 'HAllA' else [idx for idx in range(loader.X.shape[0])],
                           loader.Y_tree.pre_order() if loader.name == 'HAllA' else [idx for idx in range(loader.Y.shape[0])],
                           loader.sim_table,
                           fdr_reject_table = loader.fdr_reject_table,
                           block_num = params.block_num,
                           x_dataset_label=params.x_dataset_label,
                           y_dataset_label=params.y_dataset_label,
                           cbar_label=params.cbar_label,
                           text_scale=params.text_scale,
                           block_border_width=params.block_border_width,
                           mask=params.mask,
                           trim=params.trim,
                           force_x_ft=params.force_x_ft,
                           force_y_ft=params.force_y_ft,
                           dpi=params.dpi,
                           cmap=params.cmap,
                           output_file=output_file)

if __name__ == "__main__":
    main()
