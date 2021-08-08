import os
import random

# -------------------------------------------------------------------------------------------
# Based on: https://stackoverflow.com/questions/4623446/how-do-you-sort-files-numerically
import re

def trynum(s):
    try:
        return int(s)
    except:
        try:
            return float(s)
        except:
            return s

# Example: l = ["history_1_winrate_m_0.53_s_565", "history_2_winrate_m_0.59_s_562", "history_4_winrate_m_0.56_s_563", "history_3_winrate_m_0.15_s_567"]

# Sort the given list in the way that humans expect.
def sort_nicely(l):
    def alphanum_key(s):
        """ Turn a string into a list of string and number chunks.
            "z23a" -> ["z", 23, "a"]
        """
        return [ trynum(c) for c in re.split('([0-9]+)', s) ]
    # e.g. ['history_1_winrate_m_0.53_s_565', 'history_2_winrate_m_0.59_s_562', 'history_3_winrate_m_0.15_s_567', 'history_4_winrate_m_0.56_s_563']
    l.sort(key=alphanum_key)

# Sort based on the steps in the file name 
# Search for _s_ in the path and order using after it till the end(_s_<....>)
def sort_steps(l):
    def steps_key(s):
        return [trynum(c) for c in re.split('(_s_)', s)][-1]
    # e.g. ['history_2_winrate_m_0.59_s_562', 'history_4_winrate_m_0.56_s_563', 'history_1_winrate_m_0.53_s_565', 'history_3_winrate_m_0.15_s_567']
    l.sort(key=steps_key)

# Sort based on th metric in the file name
# Search for _m_ in the path and order using after it till the othe _ (_m_<....>_)
def sort_metric(l):
    def metric_key(s):
        #return [tryint(c) for c in re.split('(_s_)', re.split('(_m_)', s)[-1])][0]
	    return [trynum(c) for c in re.split('(_m_(.*)_s_)', s)][-2]
    # e.g. ['history_3_winrate_m_0.15_s_567', 'history_1_winrate_m_0.53_s_565', 'history_4_winrate_m_0.56_s_563', 'history_2_winrate_m_0.59_s_562']
    l.sort(key=metric_key)
# -------------------------------------------------------------------------------------------
    
def get_all(log_dir):
    return os.listdir(log_dir)

def get_startswith(log_dir, startswith):
    file_list = [f for f in os.listdir(log_dir) if f.startswith(startswith)]
    return file_list

def get_sorted(log_dir, startswith, sorting_function, return_count=False):
    file_list = get_startswith(log_dir, startswith)
    sorting_function(file_list)
    if(return_count):
        return file_list, len(file_list)
    return file_list

def get_latest(log_dir, startswith, return_count=False):
    file_list = get_sorted(log_dir, startswith, sort_steps, return_count)
    if(return_count):
        return [file_list[-1]], len(file_list)

    return [file_list[-1]]

def get_first(log_dir, startswith, return_count=False):
    file_list = get_sorted(log_dir, startswith, sort_steps, return_count)
    if(return_count):
        return [file_list[0]], len(file_list)

    return [file_list[0]]

def get_random_from(full_list, seed=1):
    random.seed(seed)
    return [full_list[random.randint(0, len(full_list)-1)]]

def get_random(log_dir, startswith, seed=1, return_count=False):
    random.seed(seed)
    file_list = get_startswith(log_dir, startswith)
    return_files = get_random_from(file_list, seed=seed)
    if(return_count):
        return return_files, len(file_list)

    return return_files


def get_idx(log_dir, startswith, idx, sorting_function=sort_steps):
    file_list = get_sorted(log_dir, startswith, sorting_function)
    if(idx >= len(file_list)):
        raise ValueError("Index for the file is greater than the length of the available files")
    return [file_list[idx]]