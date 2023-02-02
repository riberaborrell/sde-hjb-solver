import os
import shutil
import sys

import numpy as np

from sde_hjb_solver.config import PROJECT_ROOT_DIR, DATA_ROOT_DIR

def get_project_dir():
    ''' returns the absolute path of the repository's directory
    '''
    return PROJECT_ROOT_DIR

def get_data_dir():
    ''' returns the absolute path of the repository's data directory
    '''
    return DATA_ROOT_DIR

def make_dir_path(dir_path):
    ''' Create directories of the given path if they do not already exist
    '''
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)

def empty_dir(dir_path):
    ''' Remove all files in the directory from the given path
    '''
    if os.path.isdir(dir_path):
        for filename in os.listdir(dir_path):
            file_path = os.path.join(dir_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete {}. Reason: {}'.format((file_path, e)))

def save_data(data_dict, rel_dir_path):

    dir_path = os.path.join(get_data_dir(), rel_dir_path)

    # create directoreis of the given path if it does not exist
    make_dir_path(dir_path)

    file_path = os.path.join(dir_path, 'hjb-solution.npz')
    np.savez(file_path, **data_dict)

def load_data(rel_dir_path):
    try:
        file_path = os.path.join(get_data_dir(), rel_dir_path, 'hjb-solution.npz')
        data = dict(np.load(file_path, allow_pickle=True))
        for file_name in data.keys():
            if data[file_name].ndim == 0:
                data[file_name] = data[file_name].item()
        return data
    except FileNotFoundError as e:
        print(e)
        sys.exit()

