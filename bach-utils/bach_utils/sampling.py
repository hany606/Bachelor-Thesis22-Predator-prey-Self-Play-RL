from bach_utils.logger import get_logger
clilog = get_logger()

import bach_utils.os as utos
import bach_utils.list as utlst
import bach_utils.sorting as utsrt
from copy import deepcopy
import os
from datetime import datetime
import random


def sample_set(source_list, num):
    sample = []
    for i in range(num):
        idx = i % len(source_list) # if num < len(list) then it will just add them, if not then it will be a circular buffer
        sample.append(source_list[idx])
    return sample

# Return names of the sampled models
# TODO: add the arhive in the parameters to be used to extract the model information via its name
def sample_opponents(files_list, num_sampled_opponents, selection, sorted=False, randomly_reseed=True, delta=0, idx=None):
    if(randomly_reseed):
        random_seed = datetime.now().microsecond//1000
        clilog.debug(f"Reseed opponent sampler: {random_seed}")
        random.seed(random_seed)
        
    sampled_opponents_filenames = []
    if(len(files_list) == 0):
        sampled_opponents_filenames = [None for _ in range(num_sampled_opponents)]
    else:
        if(selection == "random"):
            # print("_______________________________")
            # print(files_list)
            file_list_copy = deepcopy(files_list)
            # random.shuffle(file_list_copy)
            sampled_opponents_filenames = [utlst.get_random_from(file_list_copy)[0] for _ in range(num_sampled_opponents)]  # TODO: Take care about pigonhole principle -> if the set is too small, then the likelihood for selection each element in the list will be relatively large
            # print(file_list_copy)
            # print(sampled_opponents_filenames)
            # print("_______________________________")

        elif(selection == "latest"):
            # The list is not sorted by its nature -> 
            # We add to the archive list less frequent than we sample, thus it is more optimized if we sort after adding to the archive not with each sampling
            # However, take into consideration that we may sample with different metrics in the training and the evaluation, thus we need to pass the correct sorted archive
            # We may only sort the keys for the dictionary not the whole dictionary
            if(not sorted):
                files_list = utsrt.sort_steps(files_list)
            latest = files_list[-1]
            sampled_opponents_filenames = [latest for _ in range(num_sampled_opponents)]

        elif(selection == "cyclic"):
            sampled_opponents_filenames = [files_list[idx%len(files_list)] for _ in range(num_sampled_opponents)]  # TODO: Take care about pigonhole principle -> if the set is too small, then the likelihood for selection each element in the list will be relatively large

        elif(selection == "reverse-cyclic"):
            sampled_opponents_filenames = [files_list[-((idx+1)%len(files_list))] for _ in range(num_sampled_opponents)]  # TODO: Take care about pigonhole principle -> if the set is too small, then the likelihood for selection each element in the list will be relatively large

        elif(selection == "delta-latest"):
            # The list is not sorted by its nature -> 
            # We add to the archive list less frequent than we sample, thus it is more optimized if we sort after adding to the archive not with each sampling
            # However, take into consideration that we may sample with different metrics in the training and the evaluation, thus we need to pass the correct sorted archive
            # We may only sort the keys for the dictionary not the whole dictionary
            if(not sorted):
                files_list = utsrt.sort_steps(files_list)
            delta_latest = files_list[-delta:] # get the latest delta models
            # print("--------------- Debug ---------------")
            # print(files_list)
            # print(delta_latest)
            # print("-------------------------------------")
            sampled_opponents_filenames = [utlst.get_random_from(delta_latest)[0] for _ in range(num_sampled_opponents)]  # TODO: Take care about pigonhole principle -> if the set is too small, then the likelihood for selection each element in the list will be relatively large
            
        elif(selection == "gauss"):
            # TODO: discretized normal distribution with mean in the middle and variance delta
            # import matplotlib.pyplot as plt
            # import numpy as np

            # def normal(x, mu, sigma):
            #     return 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (x - mu)**2 / (2 * sigma**2))

            # mu, sigma = 0, 0.1
            # s = np.random.normal(mu, sigma, 1000)
            # count, bins, ignored = plt.hist(s, 30, density=True)
            # plt.plot(bins, normal(bins, mu, sigma), linewidth=2, color='r')

            # plt.show()

            # print("_______________________________")
            # print(files_list)
            file_list_copy = deepcopy(files_list)
            # random.shuffle(file_list_copy)
            sampled_opponents_filenames = [utlst.get_random_from(file_list_copy)[0] for _ in range(num_sampled_opponents)]  # TODO: Take care about pigonhole principle -> if the set is too small, then the likelihood for selection each element in the list will be relatively large

        

        elif(selection == "latest-set"):
            if(not sorted):
                files_list = utsrt.sort_metric(files_list)
            reversed_files_list = deepcopy(files_list)
            reversed_files_list.reverse()
            sampled_opponents_filenames = sample_set(reversed_files_list, num_sampled_opponents)

        elif(selection == "highest"):
            if(not sorted):
                files_list = utsrt.sort_metric(files_list)
            target = files_list[-1]
            sampled_opponents_filenames = [target for _ in range(num_sampled_opponents)]

        elif(selection == "highest-set"):
            if(not sorted):
                files_list = utsrt.sort_metric(files_list)
            reversed_files_list = deepcopy(files_list)
            reversed_files_list.reverse()
            sampled_opponents_filenames = sample_set(reversed_files_list, num_sampled_opponents)

        elif(selection == "lowest"):
            if(not sorted):
                files_list = utsrt.sort_metric(files_list)
            target = files_list[0]
            sampled_opponents_filenames = [target for _ in range(num_sampled_opponents)]

        elif(selection == "lowest-set"):
            if(not sorted):
                files_list = utsrt.sort_metric(files_list)
            sampled_opponents_filenames = sample_set(files_list, num_sampled_opponents)

    return sampled_opponents_filenames

# Return paths of the sampled models
def sample_opponents_os(sample_path, startswith_keyword, num_sampled_opponents, selection, sorted=False, randomly_reseed=True, delta=0, idx=None):
    files_list = utos.get_startswith(sample_path, startswith_keyword)
    sampled_opponents_filenames = sample_opponents(files_list, num_sampled_opponents, selection, sorted, randomly_reseed=randomly_reseed, delta=delta, idx=idx)
    sampled_opponents_filenames = [os.path.join(sample_path, f) if f is not None else None for f in sampled_opponents_filenames]
    return sampled_opponents_filenames

# TODO: create sampler, that is possible to sample a set of idxs at the begining of the training and so on