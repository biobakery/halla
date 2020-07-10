from os.path import join, isdir
import os
import sys

import shutil

def create_dir(dir_name):
    # remove any existing directory with the same name
    if isdir(dir_name):
        try:
            print('Creating the directory', dir_name)
            shutil.rmtree(dir_name)
        except EnvironmentError:
            sys.exit('Unable to remove directory %s' % dir_name)
    # create a new directory
    try:
        os.mkdir(dir_name)
    except EnvironmentError:
        sys.exit('Unable to create directory %s' % dir_name)