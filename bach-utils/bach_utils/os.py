import os
import random

# -------------------------------------------------------------------------------------------
# Sourec: https://stackoverflow.com/questions/4623446/how-do-you-sort-files-numerically
import re

def tryint(s):
    try:
        return int(s)
    except:
        return s

def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [ tryint(c) for c in re.split('([0-9]+)', s) ]

def sort_nicely(l):
    """ Sort the given list in the way that humans expect.
    """
    l.sort(key=alphanum_key)
# -------------------------------------------------------------------------------------------
    
def get_all(log_dir):
    return os.listdir(log_dir)

def get_startswith(log_dir, startswith):
    file_list = [f for f in os.listdir(log_dir) if f.startswith(startswith)]
    return file_list

def get_sorted(log_dir, startswith, return_count=False):
    file_list = get_startswith(log_dir, startswith)
    # file_list.sort()
    sort_nicely(file_list)
    if(return_count):
        return file_list, len(file_list)
    return file_list

def get_latest(log_dir, startswith, return_count=False):

    if(return_count):
        file_list = get_sorted(log_dir, startswith, return_count)
        return [file_list[-1]], len(file_list)

    file_list = get_sorted(log_dir, startswith, return_count)
    return file_list[-1]

def get_first(log_dir, startswith, return_count=False):
    if(return_count):
        file_list = get_sorted(log_dir, startswith, return_count)
        return [file_list[0]], len(file_list)

    file_list = get_sorted(log_dir, startswith, return_count)
    return file_list[0]

def get_random(log_dir, startswith, seed=1, return_count=False):
    random.seed(seed)
    file_list = get_startswith(log_dir, startswith)
    if(return_count):
        return [file_list[random.randint(0, len(file_list)-1)]], len(file_list)

    return [file_list[random.randint(0, len(file_list)-1)]]

def get_idx(log_dir, startswith, idx):
    file_list = get_sorted(log_dir, startswith)
    if(idx >= len(file_list)):
        raise ValueError("Index for the file is greater than the length of the available files")
    return [file_list[idx]]