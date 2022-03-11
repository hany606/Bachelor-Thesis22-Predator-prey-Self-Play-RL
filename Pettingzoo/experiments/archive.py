# TODO: integrate the archive with sampling methods
# TODO: integrate the archive with getting specific round (To be indexable with the number of round or specific id)
# TODO: integrate the archive to be constructed using a path of files

import bach_utils.sorting as utsrt
from stable_baselines3.common.utils import check_for_correct_spaces
from stable_baselines3.common.save_util import recursive_getattr, recursive_setattr, load_from_zip_file, save_to_zip_file
import os
from random import randint
from copy import deepcopy

def random_no_sort(l, e):
    l.append(e)
    return l

# TODO: create Class called Entity/Version that have: model policy, model name, frequency, elo-score and others -> archive_dict as dictionary

# Approximatly the model will be ~ 0.000190734863 GB in the memory which means it will fit only for ~ 62915 model in 10GB RAM

# In this Archive class, there are the following archives (dict or lists datastructers):
# - archive_dict -> Ditionary such that the key is the name of the policy (indictes the round, step, metric) and the policy itself
# - sorted_archive_keys_dict: Dictionary that has sorted lists according to the required sorting metrics
class ArchiveSB3:
    def __init__(self, sorting_keys=["random"], sorting=True, moving_least_freq_flag=False, save_path=None):
        self.archive_dict = {}  # Store the names as keys and policies as values for the dictionary
        self.sorting_flag = sorting
        # Dictionary to store the functions for the sorting related with the keys
        self.sorting_functions = {"random": ["no_sort", random_no_sort],
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

        self.num_models = 0
        self.moving_least_freq_flag = moving_least_freq_flag    # This flag to activate of optimization to remove the least frequent enteries in the archive and save them on the disk instead
        self.save_path = save_path  # This is used in order to save the least frequent from RAM to disk
        self.moving_threshold = 500
        self.random_id = randint(1,100000000000)
    
    # Based on: https://github.com/DLR-RM/stable-baselines3/blob/f3a35aa786ee41ffff599b99fa1607c067e89074/stable_baselines3/common/base_class.py#L728
    def _get_model_parameters(self, model):
        # Copy parameter list so we don't mutate the original dict
        data = model.__dict__.copy()
        exclude = set(model._excluded_save_params())
        state_dicts_names, torch_variable_names = deepcopy(model._get_torch_save_params())
        all_pytorch_variables = state_dicts_names + torch_variable_names
        for torch_var in all_pytorch_variables:
            # We need to get only the name of the top most module as we'll remove that
            var_name = torch_var.split(".")[0]
            # Any params that are in the save vars must not be saved by data
            exclude.add(var_name)
        # Remove parameter entries of parameters which are to be excluded
        for param_name in exclude:
            data.pop(param_name, None)

        # Build dict of torch variables
        pytorch_variables = None
        if torch_variable_names is not None:
            pytorch_variables = {}
            for name in torch_variable_names:
                attr = recursive_getattr(model, name)
                pytorch_variables[name] = attr

        # Build dict of state_dicts
        params_to_save = deepcopy(model.get_parameters())
        return {"data": data, "params": params_to_save, "pytorch": pytorch_variables, "device": model.device}

    # Add in the dictionary, and added them to the required sorted lists and sort them
    def add(self, name, model):
        # 1. Get the network parameter from the model policy
        model_parameters = self._get_model_parameters(model)
        # TODO: Think about if the length of the key will differ that much in getting and adding in the directory or not
        
        # 2. Add them to the archive
        # The model will never be updated under the same name, instead if it is trained again, it will be saved with a different name
        # Logically, it is not reasonable that it will save with the same name, but there is a case
        # When the evaluation was not made -> no change in the name but the model changed -> So, it is better if it is saving to evaluate the saved model before saving it
        # However, it might be possible that after another evaluation -> the same performance
        # if(name in self.archive_dict):
        #     raise ValueError("Adding a model with the same name of an existing model")
        self.archive_dict[name] = {"model": model_parameters, "last_time": self.num_models, "state": "RAM"}
        self.num_models += 1

        if(self.sorting_flag):
            for key in self.sorting_keys:
                k = self.sorting_functions[key][0]
                sorting_function = self.sorting_functions[key][1]

                # self.sorted_archive_keys_dict[k].append(name)   # just append the name
                self.sorted_archive_keys_dict[k] = sorting_function(self.sorted_archive_keys_dict[k], name)
                # print(f"Sort res{len(self.sorted_archive_keys_dict[k])}, -> key: {k}")
        # No problems will happen to sorting as it is with the keys -> name and we are keeping it
        if(self.num_models % self.moving_threshold*3 == 0 and self.moving_least_freq_flag):
            for key in self.archive_dict[name].keys():
                if(self.archive_dict[key]["last_time"] - self.num_models >= self.moving_threshold):
                    # model = self.archive_dict[name]["model"]
                    # By default for now it is automatically saved in disk, so here we just need to remove it from the RAM
                    self.archive_dict[key]["state"] = "disk"
                    self.archive_dict[key]["device"] = self.archive_dict[key]["model"]["device"]
                    del self.archive_dict[key]["model"]
                    
    # Here possible values for the sorting_key: no_sort, sort_steps, sort_metric    (internal)
    def _get_sorted(self, sorting_key):
        if(not self.sorting_flag):
            raise ValueError("Sorted flag is False, it is not possible to return sorted list from the Archive")

        sorted_names = self.sorted_archive_keys_dict[sorting_key]
        sorted_policies = [self.archive_dict[s] for s in sorted_names]
        # print(f"Get sorted: current length of archive: {self.num_models} returned names{len(sorted_names)}")
        return [sorted_names, sorted_policies]

    # sorting_key: random, latest, ....etc
    # Here possible values for the sorting_key: random, latest, highest, lowest, ... that the user using them in the training parameters
    def get_sorted(self, sorting_key):
        # print(self.sorting_functions[sorting_key][0])
        # print(len(self.get_sorted_dict[self.sorting_functions[sorting_key][0]]()[0]))
        return self.get_sorted_dict[self.sorting_functions[sorting_key][0]]()

    # Based on: https://github.com/DLR-RM/stable-baselines3/blob/f3a35aa786ee41ffff599b99fa1607c067e89074/stable_baselines3/common/base_class.py#L627
    def load(self, algorithm_class, name, env, **kwargs):
        # Load the model
        if(self.archive_dict[name]["state"] == "disk"):
            data, params, pytorch_variables = load_from_zip_file(os.path.join(self.save_path, name), device=self.archive_dict[name]["device"])
            self.archive_dict[name]["model"] = {"data": data, "params": params, "pytorch": pytorch_variables, "device": self.archive_dict[name]["device"]}
        
        self.archive_dict[name]["last_time"] = self.num_models
        model_parameters = self.archive_dict[name]["model"]
        data, params, pytorch_variables, device = model_parameters["data"], model_parameters["params"], model_parameters["pytorch"], model_parameters["device"],
        # Remove stored device information and replace with ours
        if "policy_kwargs" in data:
            if "device" in data["policy_kwargs"]:
                del data["policy_kwargs"]["device"]

        if "policy_kwargs" in kwargs and kwargs["policy_kwargs"] != data["policy_kwargs"]:
            raise ValueError(
                f"The specified policy kwargs do not equal the stored policy kwargs."
                f"Stored kwargs: {data['policy_kwargs']}, specified kwargs: {kwargs['policy_kwargs']}"
            )

        if "observation_space" not in data or "action_space" not in data:
            raise KeyError("The observation_space and action_space were not given, can't verify new environments")

        if env is not None:
            # Wrap first if needed
            env = algorithm_class._wrap_env(env, data["verbose"])
            # Check if given env is valid
            check_for_correct_spaces(env, data["observation_space"], data["action_space"])
        else:
            # Use stored env, if one exists. If not, continue as is (can be used for predict)
            if "env" in data:
                env = data["env"]

        # noinspection PyArgumentList
        model = algorithm_class(  # pytype: disable=not-instantiable,wrong-keyword-args
            policy=data["policy_class"],
            env=env,
            device=device,
            _init_setup_model=False,  # pytype: disable=not-instantiable,wrong-keyword-args
        )

        # load parameters
        model.__dict__.update(data)
        model.__dict__.update(kwargs)
        model._setup_model()

        # put state_dicts back in place
        model.set_parameters(params, exact_match=True, device=device)

        # put other pytorch variables back in place
        if pytorch_variables is not None:
            for name in pytorch_variables:
                # Skip if PyTorch variable was not defined (to ensure backward compatibility).
                # This happens when using SAC/TQC.
                # SAC has an entropy coefficient which can be fixed or optimized.
                # If it is optimized, an additional PyTorch variable `log_ent_coef` is defined,
                # otherwise it is initialized to `None`.
                if pytorch_variables[name] is None:
                    continue
                # Set the data attribute directly to avoid issue when using optimizers
                # See https://github.com/DLR-RM/stable-baselines3/issues/391
                recursive_setattr(model, name + ".data", pytorch_variables[name].data)

        # Sample gSDE exploration matrix, so it uses the right device
        # see issue #44 on stable-baselines3
        if model.use_sde:
            model.policy.reset_noise()  # pytype: disable=attribute-error
        self.archive_dict[name]["last_time"] = self.num_models

        return model
        
    def change_archive_core(self, archive):
        self.archive_dict = deepcopy(archive.archive_dict)
        self.num_models = archive.num_models
        for k in self.sorting_keys:
            key = self.sorting_functions[k][0]
            self.sorted_archive_keys_dict[key] = deepcopy(archive.sorted_archive_keys_dict[key])
        