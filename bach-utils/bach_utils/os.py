import os
import random
import bach_utils.sorting as utsrt
import bach_utils.list as utlst
    
def get_all(log_dir):
    return os.listdir(log_dir)

def get_startswith(log_dir, startswith):
    file_list = [f for f in os.listdir(log_dir) if f.startswith(startswith)]
    return file_list

def get_sorted(log_dir, startswith, sorting_function, return_count=False):
    file_list = get_startswith(log_dir, startswith)
    return utlst.get_sort(file_list, sorting_function, return_count)
    sorting_function(file_list)
    if(return_count):
        return file_list, len(file_list)
    return file_list

def get_latest(log_dir, startswith, return_count=False):
    file_list = get_startswith(log_dir, startswith)
    return utlst.get_latest(file_list, return_count)

    file_list = get_sorted(log_dir, startswith, utsrt.sort_steps, return_count)
    if(return_count):
        return [file_list[-1]], len(file_list)
    return [file_list[-1]]

def get_first(log_dir, startswith, return_count=False):
    file_list = get_startswith(log_dir, startswith)
    return utlst.get_first(file_list, return_count)

    file_list = get_sorted(log_dir, startswith, sort_steps, return_count)
    if(return_count):
        return [file_list[0]], len(file_list)

    return [file_list[0]]

# def get_random_from(full_list, seed=1):
#     random_idx = random.randint(0, len(full_list)-1)
#     return [full_list[random_idx]]

def get_random(log_dir, startswith, seed=1, return_count=False):
    file_list = get_startswith(log_dir, startswith)
    return utlst.get_random(file_list, seed, return_count)

    return_files = get_random_from(file_list, seed=seed)
    if(return_count):
        return return_files, len(file_list)

    return return_files


def get_idx(log_dir, startswith, idx, sorting_function=utsrt.sort_steps):
    file_list = get_startswith(log_dir, startswith)
    return utlst.get_idx(file_list, idx, sorting_function)

    file_list = get_sorted(log_dir, startswith, sorting_function)
    if(idx >= len(file_list)):
        raise ValueError("Index for the file is greater than the length of the available files")
    return [file_list[idx]]
