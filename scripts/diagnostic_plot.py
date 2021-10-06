'''Generate diagnostic plot on console environment
'''

import argparse
import sys
import numpy as np
from os.path import join, isfile, exists
import pkg_resources

from halla.utils.filesystem import reset_dir
from halla.utils.report import generate_lattice_plot
from .loader import HAllAPartialLoader

def parse_argument(args):
    parser = argparse.ArgumentParser(
        description='HAllA diagnostic plot generator')

    # --load parameters--
    parser.add_argument(
        '-i', '--input',
        help='Path to the output directory',
        required=True)
    parser.add_argument(
        '-n', '--block_num',
        help='Number of top clusters for generating the lattice plots; if -1, show all clusters',
        default=50, type=int, required=False)
    parser.add_argument(
        '--axis_stretch',
        help='Adding gaps to both ends of axis (for continuous data)',
        default=1e-5, type=float, required=False)
    parser.add_argument(
        '--plot_size',
        help='The size of each plot',
        default=4, type=float, required=False)
    parser.add_argument(
        '--file_type',
        help="The file type of the plots",
        default="pdf",
        required=False
    )
    parser.add_argument(
        '-o', '--output_dir',
        help='Directory name to store all the lattice plots under the HAllA/AllA result directory',
        default='diagnostic', required=False)
    parser.add_argument(
        '--dont_skip_large_blocks',
        required=False,
        dest='dont_skip',
        default=False,
        action='store_true')
    parser.add_argument(
        '--large_diagnostic_subset',
        help = "Subset the feature pairs plotted in large block (>15, <45) diagnostic plots.",
        required=False,
        dest='large_diagnostic_subset',
        default=105
    )
    return(parser.parse_args())

def main():
    params = parse_argument(sys.argv)
    input_dir = params.input
    loader = HAllAPartialLoader(input_dir)

    reset_dir(join(input_dir, params.output_dir))
    block_num = params.block_num
    if block_num == -1:
        block_num = len(loader.significant_blocks)
    else:
        block_num = min(block_num, len(loader.significant_blocks))
    for i, block in enumerate(loader.significant_blocks[:block_num]):
        title = 'Association %d' % (i+1)
        out_file = join(input_dir, params.output_dir, 'association_%d.' % (i+1) + params.file_type)
        warn_file = join(input_dir, params.output_dir, 'warnings.txt')
        x_data = loader.X.to_numpy()[block[0],:]
        y_data = loader.Y.to_numpy()[block[1],:]
        x_ori_data = loader.X_ori.to_numpy()[block[0],:]
        y_ori_data = loader.Y_ori.to_numpy()[block[1],:]
        x_features = loader.X_features[block[0]]
        y_features = loader.Y_features[block[1]]
        x_types = np.array(loader.X_types)[block[0]]
        y_types = np.array(loader.Y_types)[block[1]]
        if (x_data.shape[0] + y_data.shape[0]) > 15 and (x_data.shape[0] + y_data.shape[0]) <= 45:
            warn_string = "Over 15 features included in association %d. Only a subset of features will be shown in the diagnostic plot. Increase --large_diagnostic_subset beyond 105 to show more." % (i+1)
            if exists(warn_file):
                append_write = 'a'
            else:
                append_write = 'w'
            warn_file_write = open(warn_file, append_write)
            warn_file_write.write(warn_string + '\n')
            warn_file_write.close()
            print(warn_string, file = sys.stderr)
            generate_lattice_plot(x_data, y_data, x_ori_data, y_ori_data,
                                x_features, y_features, x_types, y_types, title,
                                out_file, axis_stretch=params.axis_stretch, plot_size=params.plot_size, n_pairs_to_show = params.large_diagnostic_subset)
            continue
        if (x_data.shape[0] + y_data.shape[0]) > 45 and not params.dont_skip:
            warn_string = "Skipping association %d because there are too many included features. Add --dont_skip_large_blocks to disable this behavior." % (i+1)
            if exists(warn_file):
                append_write = 'a'
            else:
                append_write = 'w'
            warn_file_write = open(warn_file, append_write)
            warn_file_write.write(warn_string + '\n')
            warn_file_write.close()
            print(warn_string, file = sys.stderr)
            continue
        generate_lattice_plot(x_data, y_data, x_ori_data, y_ori_data,
                                x_features, y_features, x_types, y_types, title,
                                out_file, axis_stretch=params.axis_stretch, plot_size=params.plot_size, n_pairs_to_show = (x_data.shape[0] + y_data.shape[0])**2)

if __name__ == "__main__":
    main()