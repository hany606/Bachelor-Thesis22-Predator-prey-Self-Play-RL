from bach_utils.sorting import population_key

from copy import deepcopy
import re


def _filter(l, key_func, filter_func):
    ll = deepcopy(l)
    keys = list(map(lambda x: key_func(x), ll))
    keys = dict(zip(ll,keys))
    filtered_list = list(filter(lambda x: filter_func(keys[x]), ll))
    return filtered_list

def filter_population(l, population_idx):
    filter_func = lambda x: x==population_idx
    return _filter(l, population_key, filter_func)

if __name__ == '__main__':
    l = ['history_0_winrate_m_1.0_s_500_p_1', 'history_0_winrate_m_1.0_s_500_p_0', 'history_0_winrate_m_1.0_s_1000_p_1', 'history_0_winrate_m_1.0_s_1000_p_0', 'history_0_winrate_m_1.0_s_1500_p_0', 'history_0_winrate_m_1.0_s_1500_p_1', 'history_0_winrate_m_1.0_s_2000_p_0', 'history_0_winrate_m_1.0_s_2000_p_1', 'history_1_winrate_m_1.0_s_2500_p_1', 'history_1_winrate_m_1.0_s_2500_p_0', 'history_1_winrate_m_1.0_s_3000_p_0', 'history_1_winrate_m_1.0_s_3000_p_1', 'history_1_winrate_m_1.0_s_3500_p_0', 'history_1_winrate_m_1.0_s_3500_p_1', 'history_1_winrate_m_1.0_s_4000_p_0', 'history_1_winrate_m_1.0_s_4000_p_1', 'history_2_winrate_m_1.0_s_4500_p_1', 'history_2_winrate_m_1.0_s_4500_p_0', 'history_2_winrate_m_1.0_s_5000_p_0', 'history_2_winrate_m_1.0_s_5000_p_1', 'history_2_winrate_m_1.0_s_5500_p_0', 'history_2_winrate_m_1.0_s_5500_p_1', 'history_2_winrate_m_1.0_s_6000_p_0', 'history_2_winrate_m_1.0_s_6000_p_1', 'history_3_winrate_m_0.5_s_6500_p_1', 'history_3_winrate_m_1.0_s_6500_p_0', 'history_3_winrate_m_0.5_s_7000_p_1', 'history_3_winrate_m_1.0_s_7000_p_0', 'history_3_winrate_m_0.5_s_7500_p_1', 'history_3_winrate_m_1.0_s_7500_p_0', 'history_3_winrate_m_0.5_s_8000_p_1', 'history_3_winrate_m_1.0_s_8000_p_0', 'history_4_winrate_m_0.5_s_8500_p_0', 'history_4_winrate_m_1.0_s_8500_p_1', 'history_4_winrate_m_0.0_s_9000_p_0', 'history_4_winrate_m_1.0_s_9000_p_1', 'history_4_winrate_m_0.0_s_9500_p_0', 'history_4_winrate_m_1.0_s_9500_p_1', 'history_4_winrate_m_1.0_s_10000_p_1', 'history_4_winrate_m_0.0_s_10000_p_0', 'history_5_winrate_m_1.0_s_10500_p_1', 'history_5_winrate_m_0.0_s_10500_p_0', 'history_5_winrate_m_0.0_s_11000_p_0', 'history_5_winrate_m_1.0_s_11000_p_1', 'history_5_winrate_m_1.0_s_11500_p_1', 'history_5_winrate_m_0.0_s_11500_p_0', 'history_5_winrate_m_0.0_s_12000_p_0', 'history_5_winrate_m_1.0_s_12000_p_1', 'history_6_winrate_m_1.0_s_12500_p_0', 'history_6_winrate_m_1.0_s_12500_p_1', 'history_6_winrate_m_1.0_s_13000_p_1', 'history_6_winrate_m_0.5_s_13000_p_0', 'history_6_winrate_m_1.0_s_13500_p_1', 'history_6_winrate_m_0.5_s_13500_p_0', 'history_6_winrate_m_1.0_s_14000_p_1', 'history_6_winrate_m_0.5_s_14000_p_0']#["history_1_winrate_m_0.53_s_565_p_1", "history_1_winrate_m_0.8_s_562_p_0", "history_2_winrate_m_0.59_s_562_p_2", "history_4_winrate_m_0.56_s_563_p_1", "history_3_winrate_m_0.15_s_567_p_1"]
    print("Filtering by the population number")
    print(filter_population(l, population_idx=1))