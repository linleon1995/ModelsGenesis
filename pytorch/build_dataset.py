
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt

from utils import generate_pair
# from data.data_utils import get_files
# from data.data_transformer import ImageDataTransformer


class GeneralDataset():
    def __init__(self, input_path_list, target_path_list, input_load_func, target_load_func, data_transformer=None):
        self.input_load_func = input_load_func
        self.target_load_func = target_load_func
        self.input_path_list = input_path_list
        self.target_path_list = target_path_list
        self.data_transformer = data_transformer
    
    def __len__(self):
        return len(self.input_path_list)

    def __getitem__(self, idx):
        input_data, target = self.input_load_func(self.input_path_list[idx]), self.target_load_func(self.target_path_list[idx])
        input_data, target = np.swapaxes(np.swapaxes(input_data, 0, 1), 1, 2), np.swapaxes(np.swapaxes(target, 0, 1), 1, 2)
        if self.data_transformer is not None:
            input_data, target = self.data_transformer(input_data, target)
        input_data = input_data / 255
        input_data, target = input_data[np.newaxis], target[np.newaxis]
        # for s in range(32):
        #     if np.sum(target[...,s]) > 0:
        #         plt.imshow(input_data[0,...,s], 'gray')
        #         plt.imshow(target[0,...,s], alpha=0.2)
        #         plt.show()
        # TODO: related issue: https://discuss.pytorch.org/t/torch-from-numpy-not-support-negative-strides/3663/12
        # Re-assign array memory beacause of the flipping operation
        input_data, target = input_data.copy(), target.copy()
        return input_data, target


def flip(x, y):
    dim = len(x.shape)
    for axis in range(dim):
    # for axis in range(1,2):
        if np.random.rand() > 0.5:
            x = np.flip(x, axis=axis)
            y = np.flip(y, axis=axis)
    return x, y


def build_dataloader(input_roots, target_roots, train_cases, valid_cases=None, train_batch_size=1, pin_memory=True, 
                     num_workers=0, transform_config=None):
    input_load_func = target_load_func = np.load
    transformer = flip

    def get_samples(roots, cases):
        samples = []
        for root in roots:
            samples.extend(get_files(root, keys=cases))
        return samples

        
    train_input_samples = get_samples(input_roots, train_cases)   
    train_target_samples = get_samples(target_roots, train_cases)   

    # TODO: catch only foreground (for experiment)
    for input_path, target_path in zip(train_input_samples, train_target_samples):
        if np.sum(np.load(target_path)) <= 0:
            train_input_samples.remove(input_path)
            train_target_samples.remove(target_path)
            
    train_dataset = GeneralDataset(
        train_input_samples, train_target_samples, input_load_func, target_load_func, data_transformer=transformer)
    train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, pin_memory=pin_memory, num_workers=num_workers)

    valid_dataloader = None
    if valid_cases is not None:
        valid_input_samples = get_samples(input_roots, valid_cases, 'npy')   
        valid_target_samples = get_samples(target_roots, valid_cases, 'npy')   
        # TODO: Temporally solution because of slowing validation
        valid_input_samples, valid_target_samples = valid_input_samples[:500], valid_target_samples[:500]
        valid_dataset = GeneralDataset(valid_input_samples, valid_target_samples, input_load_func, target_load_func)
        valid_dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=False, pin_memory=pin_memory, num_workers=num_workers)
    
    return train_dataloader, valid_dataloader



def get_files(path, keys=[], return_fullpath=True, sort=True, sorting_key=None, recursive=True, get_dirs=False, ignore_suffix=False):
    """Get all the file name under the given path with assigned keys
    Args:
        path: (str)
        keys: (list, str)
        return_fullpath: (bool)
        sort: (bool)
        sorting_key: (func)
        recursive: The flag for searching path recursively or not(bool)
    Return:
        file_list: (list)
    """
    file_list = []
    assert isinstance(keys, (list, str))
    if isinstance(keys, str): keys = [keys]
    # Rmove repeated keys
    keys = list(set(keys))

    def push_back_filelist(root, f, file_list, is_fullpath):
        f = f[:-4] if ignore_suffix else f
        if is_fullpath:
            file_list.append(os.path.join(root, f))
        else:
            file_list.append(f)

    for i, (root, dirs, files) in enumerate(os.walk(path)):
        # print(root, dirs, files)
        if not recursive:
            if i > 0: break

        if get_dirs:
            files = dirs
            
        for j, f in enumerate(files):
            if keys:
                for key in keys:
                    if key in f:
                        push_back_filelist(root, f, file_list, return_fullpath)
            else:
                push_back_filelist(root, f, file_list, return_fullpath)

    if file_list:
        if sort: file_list.sort(key=sorting_key)
    else:
        f = 'dir' if get_dirs else 'file'
        if keys: 
            logging.warning(f'No {f} exist with key {keys}.') 
        else: 
            logging.warning(f'No {f} exist.') 
    return file_list

