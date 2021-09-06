import bach_utils.sorting as utsrt
# In this Archive class, there are the following archives (dict or lists datastructers):
# - archive_dict -> Ditionary such that the key is the name of the policy (indictes the round, step, metric) and the policy itself
# - sorted_archive_keys_dict: Dictionary that has sorted lists according to the required sorting metrics
class Archive:
    def __init__(self, sorting_keys=["random"], sorting=True):
        self.archive_dict = {}  # Store the names as keys and policies as values for the dictionary
        self.sorting_flag = sorting
        # Dictionary to store the functions for the sorting related with the keys
        self.sorting_functions = {"random": ["no_sort", lambda x: x],
                                  "latest": ["sort_steps", utsrt.insertion_sorted_steps],
                                  "latest-set": ["sort_steps", utsrt.insertion_sorted_steps],
                                  "highest": ["sort_metric", utsrt.insertion_sorted_metric],
                                  "highest-set": ["sort_metric", utsrt.insertion_sorted_metric],
                                  "lowest": ["sort_metric", utsrt.insertion_sorted_metric],
                                  "lowest-set": ["sort_metric", utsrt.insertion_sorted_metric],
                                  }
        self.sorting_keys = list(set(sorting_keys))
        for sk in self.sorting_keys:
            if(sk not in self.sorting_functions):
                raise ValueError("Incorrect values for sorting_keys in Archive object sorting_keys")
            
        self.sorted_archive_keys_dict = {self.sorting_functions[k][0]:[] for k in self.sorting_keys}

        self.get_sorted_dict = {"no_sort": lambda : [list(self.archive_dict.keys()), list(self.archive_dict.values())] ,
                                "sort_steps": lambda : self._get_sorted("sort_steps"),
                                "sort_metric": lambda : self._get_sorted("sort_metric"),
                                }

    # Add in the dictionary, and added them to the required sorted lists and sort them
    def add(self, name, policy):
        self.archive_dict[name] = policy

        if(self.sorting_flag):
            for key in self.sorting_keys:
                k = self.sorting_functions[key][0]
                sorting_function = self.sorting_functions[k][1]

                self.sorted_archive_keys_dict[k].append(name)   # just append the name
                self.sorted_archive_keys_dict[k] = sorting_function(self.sorted_archive_keys_dict[k])

    # Here possible values for the sorting_key: no_sort, sort_steps, sort_metric    (internal)
    def _get_sorted(self, sorting_key):
        if(not self.sorting_flag):
            raise ValueError("Sorted flag is False, it is not possible to return sorted list from the Archive")

        sorted_names = self.sorted_archive_keys_dict[sorting_key]
        sorted_policies = [self.archive_dict[s] for s in sorted_names]
        return sorted_names, sorted_policies

    # Here possible values for the sorting_key: random, latest, highest, lowest, ... that the user using them in the training parameters
    def get_sorted(self, sorting_key):
        return self.get_sorted_dict[self.sorting_functions[sorting_key][0]]()

    def load(self, name):
        return self.archive_dict[name]
    