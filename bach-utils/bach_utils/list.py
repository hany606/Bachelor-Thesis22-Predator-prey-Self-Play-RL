import random
from copy import deepcopy
import bach_utils.sorting as utsrt


def get_sorted(source_list, sorting_function, return_count=False):
    source_list_sorted = sorting_function(source_list)
    if(return_count):
        return source_list_sorted, len(source_list_sorted)
    return source_list_sorted

def get_latest(source_list, return_count=False):
    source_list_sorted = get_sorted(source_list, utsrt.sort_steps, return_count)
    if(return_count):
        return [source_list_sorted[-1]], len(source_list_sorted)
    return [source_list_sorted[-1]]

def get_first(source_list, return_count=False):
    source_list_sorted = get_sorted(source_list, utsrt.sort_steps, return_count)
    if(return_count):
        return [source_list_sorted[0]], len(source_list_sorted)

    return [source_list_sorted[0]]

def get_random_from(full_list, seed=1):
    random_idx = random.randint(0, len(full_list)-1)
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

def sample_set(source_list, num):
    sample = []
    for i in range(num):
        idx = i % len(source_list) # if num < len(list) then it will just add them, if not then it will be a circular buffer
        sample.append(source_list[idx])
    return sample