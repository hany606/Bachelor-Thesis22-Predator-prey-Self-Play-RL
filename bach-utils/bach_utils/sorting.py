# Based on: https://stackoverflow.com/questions/4623446/how-do-you-sort-files-numerically
import re
from copy import deepcopy

def trynum(s):
    try:
        return int(s)
    except:
        try:
            return float(s)
        except:
            return s

# Example: l = ["history_1_winrate_m_0.53_s_565", "history_2_winrate_m_0.59_s_562", "history_4_winrate_m_0.56_s_563", "history_3_winrate_m_0.15_s_567"]
# --------------------------------------------------------------------------------
def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [ trynum(c) for c in re.split('([0-9]+)', s) ]

def steps_key(s):
    return [trynum(c) for c in re.split('(_s_)', s)][-1]

def metric_key(s):
    #return [tryint(c) for c in re.split('(_s_)', re.split('(_m_)', s)[-1])][0]
    return [trynum(c) for c in re.split('(_m_(.*)_s_)', s)][-2]

def _sort(l, sorting_key):
    ll = deepcopy(l)
    ll.sort(key=sorting_key)
    return ll

# This producing a sorted list and works in O(n) assuming the list is already sorted: just insert in the correct place (TODO: check aggregation complexity analysis or something with similar name, I don't really remember)
def _insertion_sort(l, e, sorting_key):
    ll = deepcopy(l)
    k_e = sorting_key(e)
    for i in range(len(ll)):
        k = sorting_key(ll[i])
        if(k_e < k):
            ll.insert(i, e)
            break
    if(len(ll) == len(l)):
        ll.append(e)
    return ll
# --------------------------------------------------------------------------------

# --------------------------------------------------------------------------------
# Sort the given list in the way that humans expect.
# e.g. ['history_1_winrate_m_0.53_s_565', 'history_2_winrate_m_0.59_s_562', 'history_3_winrate_m_0.15_s_567', 'history_4_winrate_m_0.56_s_563']
def sort_nicely(l):
    return _sort(l, alphanum_key)
def insertion_sorted_steps(l, e):
    return _insertion_sort(l, e, alphanum_key)
# --------------------------------------------------------------------------------

# --------------------------------------------------------------------------------
# Sort based on the steps in the file name 
# Search for _s_ in the path and order using after it till the end(_s_<....>)
# e.g. ['history_2_winrate_m_0.59_s_562', 'history_4_winrate_m_0.56_s_563', 'history_1_winrate_m_0.53_s_565', 'history_3_winrate_m_0.15_s_567']
def sort_steps(l):    
    return _sort(l, steps_key)
def insertion_sorted_steps(l, e):
    return _insertion_sort(l, e, steps_key)
# --------------------------------------------------------------------------------


# --------------------------------------------------------------------------------
# Sort based on th metric in the file name
# Search for _m_ in the path and order using after it till the othe _ (_m_<....>_)
# e.g. ['history_3_winrate_m_0.15_s_567', 'history_1_winrate_m_0.53_s_565', 'history_4_winrate_m_0.56_s_563', 'history_2_winrate_m_0.59_s_562']
def sort_metric(l):
    return _sort(l, metric_key)
def insertion_sorted_metric(l, e):
    return _insertion_sort(l, e, metric_key)
# --------------------------------------------------------------------------------


