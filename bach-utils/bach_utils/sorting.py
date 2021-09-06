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

# Sort the given list in the way that humans expect.
def sort_nicely(l):
    def alphanum_key(s):
        """ Turn a string into a list of string and number chunks.
            "z23a" -> ["z", 23, "a"]
        """
        return [ trynum(c) for c in re.split('([0-9]+)', s) ]
    ll = deepcopy(l)
    # e.g. ['history_1_winrate_m_0.53_s_565', 'history_2_winrate_m_0.59_s_562', 'history_3_winrate_m_0.15_s_567', 'history_4_winrate_m_0.56_s_563']
    ll.sort(key=alphanum_key)
    return ll

# Sort based on the steps in the file name 
# Search for _s_ in the path and order using after it till the end(_s_<....>)
def sort_steps(l):
    def steps_key(s):
        return [trynum(c) for c in re.split('(_s_)', s)][-1]
    
    ll = deepcopy(l)
    # e.g. ['history_2_winrate_m_0.59_s_562', 'history_4_winrate_m_0.56_s_563', 'history_1_winrate_m_0.53_s_565', 'history_3_winrate_m_0.15_s_567']
    ll.sort(key=steps_key)
    return ll

# Sort based on th metric in the file name
# Search for _m_ in the path and order using after it till the othe _ (_m_<....>_)
def sort_metric(l):
    def metric_key(s):
        #return [tryint(c) for c in re.split('(_s_)', re.split('(_m_)', s)[-1])][0]
	    return [trynum(c) for c in re.split('(_m_(.*)_s_)', s)][-2]
    
    ll = deepcopy(l)
    # e.g. ['history_3_winrate_m_0.15_s_567', 'history_1_winrate_m_0.53_s_565', 'history_4_winrate_m_0.56_s_563', 'history_2_winrate_m_0.59_s_562']
    ll.sort(key=metric_key)
    return ll
