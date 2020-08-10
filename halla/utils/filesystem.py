from os.path import join, isdir
import os
import sys

import shutil

def create_dir(dir_name, verbose=True):
    '''Creating a new directory; if the directory already exists, just let it be~
    '''
    if isdir(dir_name): return
    try:
        if verbose: print('Creating a new directory %s' % dir_name)
        os.mkdir(dir_name)
    except EnvironmentError:
        sys.exit('Unable to create directory %s' % dir_name)

def reset_dir(dir_name, verbose=True):
    '''Remove any existing directory with the same name and create a new one
    '''
    if isdir(dir_name):
        try:
            if verbose: print('Directory %s exists; deleting...' % dir_name)
            shutil.rmtree(dir_name)
        except EnvironmentError:
            sys.exit('Unable to remove directory %s' % dir_name)
    # create a new directory
    create_dir(dir_name, verbose)