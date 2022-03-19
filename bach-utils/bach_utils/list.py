import random
from copy import deepcopy
import bach_utils.sorting as utsrt
from bach_utils.filtering import filter_population 
import numpy as np

from gym.utils import seeding
seed_value = 3
np_random, seed_value = seeding.np_random(seed_value)

# TODO: Check whether do we still need return_count or not?

def get_startswith(source_list, startswith):
    target_list = [l for l in source_list if l.startswith(startswith)]
    return target_list

def get_sorted(source_list, sorting_function, return_count=False, population_idx=None):
    filtered_source_list = source_list
    if(population_idx is not None):
        filtered_source_list = filter_population(source_list, population_idx)

    source_list_sorted = sorting_function(filtered_source_list)
    if(return_count):
        return source_list_sorted, len(source_list_sorted)
    return source_list_sorted

# def get_latest(source_list, return_count=False):
#     source_list_sorted = get_sorted(source_list, utsrt.sort_steps, return_count)
#     if(return_count):
#         return [source_list_sorted[-1]], len(source_list_sorted)
#     return [source_list_sorted[-1]]

def get_latest(source_list, return_count=False, population_idx=None):
    source_list_sorted = get_sorted(source_list, utsrt.sort_steps, return_count, population_idx)    
    return_list = [source_list_sorted[-1]]
    if(return_count):
        return return_list, len(source_list_sorted)
    return return_list

def get_first(source_list, return_count=False):
    source_list_sorted = get_sorted(source_list, utsrt.sort_steps, return_count)
    if(return_count):
        return [source_list_sorted[0]], len(source_list_sorted)

    return [source_list_sorted[0]]

def get_random_from(full_list, seed=1):
    # random.randint is uniform
    # https://stackoverflow.com/questions/41100287/randint-doesnt-always-follow-uniform-distribution
    random_idx = np_random.randint(0, len(full_list))#random.randint(0, len(full_list)-1) #np.random.randint(0, len(full_list))#
    # print(random_idx, len(full_list))
    return [full_list[random_idx]]

def get_random(source_list, seed=1, return_count=False):
    return_files = get_random_from(deepcopy(source_list), seed=seed)
    if(return_count):
        return return_files, len(source_list)

    return return_files

def get_idx(source_list, idx, sorting_function=utsrt.sort_steps):
    source_list_sorted = get_sorted(source_list, sorting_function)
    if(idx >= len(source_list_sorted)):
        raise ValueError("Index for the file is greater than the length of the available files")
    return [source_list_sorted[idx]]
