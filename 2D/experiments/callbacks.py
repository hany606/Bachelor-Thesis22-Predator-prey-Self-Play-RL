# Generation and round are being used as the same meaning
from random import sample
from stable_baselines3.common.callbacks import EvalCallback, EventCallback
import os
import bach_utils.sorting as utsrt
import bach_utils.os as utos
import bach_utils.list as utlst
import bach_utils.sampling as utsmpl

import numpy as np

from stable_baselines3.common.vec_env import sync_envs_normalization
import wandb
from shared import *

OS = False #True   # This flag just for testing now in order not to break the compatibility and the working of the code

# Select opponent for training
class TrainingOpponentSelectionCallback(EventCallback):
    def __init__(self, *args, **kwargs):
        self.sample_path = kwargs["sample_path"]
        self.startswith_keyword = "history"
        self.env = kwargs["env"]
        self.opponent_selection = kwargs["opponent_selection"]
        self.sample_after_rollout = kwargs["sample_after_rollout"]
        self.num_sampled_per_round = kwargs["num_sampled_per_round"]
        self.archive = kwargs["archive"]
        self.OS = OS    # Global flag

        del kwargs["sample_path"]
        del kwargs["env"]
        del kwargs["opponent_selection"]
        del kwargs["num_sampled_per_round"]
        del kwargs["sample_after_rollout"]
        # del kwargs["archive"]
        new_kwargs = {}
        for k in kwargs.keys():
            if(k == "archive"):
                continue
            new_kwargs[k] = kwargs[k]



        self.sampled_per_round = []
        self.sampled_idx = 0

        super(TrainingOpponentSelectionCallback, self).__init__(*args, **new_kwargs)

    # Only once the training starts of the round
    def _on_training_start(self):
        # if(not self.sampled_per_round):
        print("training started")
        opponent = None
        if(not self.OS):
            # print("Not OS")
            archive = self.archive.get_sorted(self.opponent_selection)
            models_names = archive[0]
            self.sampled_per_round = utsmpl.sample_opponents(models_names, self.num_sampled_per_round, selection=self.opponent_selection, sorted=True)
        if(self.OS):
            self.sampled_per_round = utsmpl.sample_opponents_os(self.sample_path, self.startswith_keyword, self.num_sampled_per_round, selection=self.opponent_selection)
        # If it is not updated with every rollout, only updated at the begining
        if(self.num_sampled_per_round == 1):
            self.env.set_target_opponent_policy_filename(self.sampled_per_round[0])
        super(TrainingOpponentSelectionCallback, self)._on_training_start()

    # def _on_training_end(self):
    #     self.sampled_per_round = []
    #     super(TrainingOpponentSelectionCallback, self)._on_training_start()

    # With every rollout
    def _on_rollout_start(self):
        print("Rollout")
        if(self.sample_after_rollout):
            opponent = None
            if(not self.OS):
                archive = self.archive.get_sorted(self.opponent_selection)
                models_names = archive[0]
                opponent = utsmpl.sample_opponents(models_names, self.num_sampled_per_round, selection=self.opponent_selection, sorted=True)[0]
            if(self.OS):
                opponent = utsmpl.sample_opponents_os(self.sample_path, self.startswith_keyword, self.num_sampled_per_round, selection=self.opponent_selection)[0]
            self.env.set_target_opponent_policy_filename(opponent)
        
        if(self.num_sampled_per_round > 1):
            print("Change sampled agent")
            self.env.set_target_opponent_policy_filename(self.sampled_per_round[self.sampled_idx % self.num_sampled_per_round]) # as a circular buffer
            self.sampled_idx += 1
        super(TrainingOpponentSelectionCallback, self)._on_rollout_start()


# Based on: https://github.com/hardmaru/slimevolleygym/blob/master/training_scripts/train_ppo_selfplay.py
# Based on: https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/callbacks.py
class EvalSaveCallback(EvalCallback):
    def __init__(self, *args, **kwargs):
        self.save_path = kwargs["save_path"]
        self.eval_metric = kwargs["eval_metric"]
        self.eval_opponent_selection = kwargs["eval_opponent_selection"]
        self.eval_sample_path = kwargs["eval_sample_path"]
        self.save_freq = kwargs["save_freq"]
        self.archive = kwargs["archive"]["self"]  # pass it by reference
        self.opponent_archive = kwargs["archive"]["opponent"]  # pass it by reference
        self.agent_name = kwargs["agent_name"]
        self.num_rounds = kwargs["num_rounds"]
        self.name_prefix = None
        self.startswith_keyword = "history"
        self.OS = OS
        # self.OS = True

        self.win_rate = None
        self.best_mean_reward = None
        self.last_mean_reward = None

        del kwargs["save_path"]
        del kwargs["eval_metric"]
        del kwargs["eval_opponent_selection"]
        del kwargs["eval_sample_path"]
        del kwargs["save_freq"]
        del kwargs["agent_name"]
        del kwargs["num_rounds"]
        # del kwargs["archive"]
        new_kwargs = {}
        for k in kwargs.keys():
            if(k == "archive"):
                continue
            new_kwargs[k] = kwargs[k]

        super(EvalSaveCallback, self).__init__(*args, **new_kwargs)
        if not isinstance(self.eval_env, DummyVecEnvSelfPlay):
            self.eval_env.__class__ = DummyVecEnvSelfPlay   # This works fine, the other solution is commented



        # self.opponent_archive = self.eval_env.archive

        self.evaluation_matrix = np.zeros((self.num_rounds, self.num_rounds))
        # Just for debugging, TODO: remove it
        # for i in range(self.num_rounds):
        #     for j in range(self.num_rounds):
        #         self.evaluation_matrix[i,j] = -1

        # if isinstance(self.eval_env, DummyVecEnv):
        #     eval_env = DummyVecEnvMod([lambda: eval_env])

    def set_name_prefix(self, name_prefix):
        self.name_prefix = name_prefix


    def _evaluate(self, model, n_eval_episodes, deterministic, sampled_opponents):
        # Sync training and eval env if there is VecNormalize
        sync_envs_normalization(self.training_env, self.eval_env)

        # Reset success rate buffer
        self._is_success_buffer = []
        return evaluate_policy(
                                model,
                                self.eval_env,
                                n_eval_episodes=n_eval_episodes,
                                render=self.render,
                                deterministic=deterministic,
                                return_episode_rewards=True,
                                warn=self.warn,
                                callback=self._log_success_callback,
                                sampled_opponents=sampled_opponents
                              )

    def _evaluate_policy_core(self, logger_prefix, n_eval_episodes, deterministic, sampled_opponents, override=False) -> bool:
        episode_rewards, episode_lengths, win_rate, std_win_rate = self._evaluate(self.model, n_eval_episodes, deterministic, sampled_opponents)

        if self.log_path is not None:
            self.evaluations_timesteps.append(self.num_timesteps)
            self.evaluations_results.append(episode_rewards)
            self.evaluations_length.append(episode_lengths)

            kwargs = {}
            # Save success log if present
            if len(self._is_success_buffer) > 0:
                self.evaluations_successes.append(self._is_success_buffer)
                kwargs = dict(successes=self.evaluations_successes)

            np.savez(
                self.log_path,
                timesteps=self.evaluations_timesteps,
                results=self.evaluations_results,
                ep_lengths=self.evaluations_length,
                **kwargs,
            )

        mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
        self.mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
        self.last_mean_reward = mean_reward
        self.win_rate, std_win_rate = win_rate, std_win_rate # ratio [0,1]

        if self.verbose > 0:
            print(f"Eval num_timesteps={self.num_timesteps}, " f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}")
            print(f"Episode length: {self.mean_ep_length:.2f} +/- {std_ep_length:.2f}")
        # Add to current Logger
        self.logger.record(f"{logger_prefix}/mean_reward", float(mean_reward))
        self.logger.record(f"{logger_prefix}/mean_ep_length", self.mean_ep_length)
        self.logger.record(f"{logger_prefix}/record_timesteps", self.num_timesteps)
        if self.verbose > 0:
            print(f"Win rate: {100 * win_rate:.2f}% +/- {std_win_rate:.2f}")
        self.logger.record(f"{logger_prefix}/win_rate", win_rate)

        if len(self._is_success_buffer) > 0:
            success_rate = np.mean(self._is_success_buffer)
            if self.verbose > 0:
                print(f"Success rate: {100 * success_rate:.2f}%")
            self.logger.record(f"{logger_prefix}/success_rate", success_rate)

        # Dump log so the evaluation results are printed with the correct timestep
        # self.logger.record(f"{logger_prefix}/time/total_timesteps", self.num_timesteps, exclude="tensorboard")
        # self.logger.dump(self.num_timesteps)
        
        if mean_reward > self.best_mean_reward:
            if self.verbose > 0:
                print("New best mean reward!")
            if self.best_model_save_path is not None:
                self.model.save(os.path.join(self.best_model_save_path, "best_model"))
            self.best_mean_reward = mean_reward
            # Trigger callback if needed
            if self.callback is not None:
                return self._on_event()
        return True

    # In order to keep the oringinal callback functions, this is just a wrapper for the parameterized _evaluate_policy() -> _evaluate_policy_core()
    def _evaluate_policy(self, force_evaluation=False) -> bool:
        if (force_evaluation or (self.eval_freq > 0 and self.n_calls % self.eval_freq == 0)):
            sampled_opponents = None
            if(not self.OS):
                # print("Sample models for evaluation")
                # print(self.opponent_archive.archive_dict.keys())
                archive = self.opponent_archive.get_sorted(self.eval_opponent_selection)
                models_names = archive[0]
                # print(len(models_names))
                sampled_opponents = utsmpl.sample_opponents(models_names, self.n_eval_episodes, selection=self.eval_opponent_selection, sorted=True)
            if(self.OS):
                sampled_opponents = utsmpl.sample_opponents_os(self.eval_sample_path, self.startswith_keyword, self.n_eval_episodes, selection=self.eval_opponent_selection)
            return self._evaluate_policy_core(logger_prefix="eval", 
                                            n_eval_episodes=self.n_eval_episodes,
                                            deterministic=self.deterministic,
                                            sampled_opponents=sampled_opponents)
        return True

    def _save_model_core(self):
        metric_value = None
        if(self.eval_metric == "steps"):
            metric_value = self.mean_ep_length
        elif(self.eval_metric == "bestreward"):
            metric_value = self.best_mean_reward 
        elif(self.eval_metric == "lastreward"):
            metric_value = self.last_mean_reward
        elif(self.eval_metric == "winrate"):
            metric_value = self.win_rate
        # history_<num-round>_<reward/points/winrate>_m_<value>_s_<num-step>
        name = f"{self.name_prefix}_{self.eval_metric}_m_{metric_value}_s_{self.num_timesteps}"
        path = os.path.join(self.save_path, name)
        self.model.save(path)
        if(not self.OS):
            self.archive.add(name, self.model) # Add the model to the archive
        if self.verbose > 0:
            print(f"Saving model checkpoint to {path}")
        return name

    def _save_model(self, force_saving=False):
        if (force_saving or (self.save_freq > 0 and self.n_calls % self.save_freq == 0)):
            name = self._save_model_core()
            return name
        return None

    def _on_step(self) -> bool:
        # 1. evaluate the policy
        result = self._evaluate_policy()#super(EvalSaveCallback, self)._on_step()
        # 2. Save the results
        name = self._save_model()
        return result

    # This doesn't work -> make save_freq=NUM_TIMESTEPS and eval_freq=NUM_TIMESTEPS will work like this
    def _on_training_end(self) -> None:
        if(self.save_freq == 0 and self.eval_freq == 0):
            self.eval_freq = self.n_calls   # There is a problem when I do not set it, thus, I have made this setting (The plots are not being reported in wandb)
            print("Evaluating the model according to the metric and save it")
            result = self._evaluate_policy(force_evaluation=True)
            name = self._save_model(force_saving=True)
            self.eval_freq = 0   # There is a problem when this line is not
        super(EvalSaveCallback, self)._on_training_end()

    # TODO: Add a feature that it will use the correct sorted from the archive if the metric for the archive is steps!
    def compute_eval_matrix_aggregate(self, prefix, round_num, opponents_path=None, agents_path=None, n_eval_rep=5, deterministic=None, algorithm_class=None):
        models_names = None
        deterministic = self.deterministic if deterministic is None else deterministic  # https://stackoverflow.com/questions/66455636/what-does-deterministic-true-in-stable-baselines3-library-means
        if(self.OS and (opponents_path is None or agents_path is None)):
            raise ValueError("Wrong value for opponent/agent path")

        if(not self.OS):
            opponent_archive = self.opponent_archive.get_sorted("random")
            opponent_models_names = opponent_archive[0] # 0 index to get just the names
            archive = self.archive.get_sorted("random")
            models_names = archive[0]

        else:
            self.eval_env.set_attr("OS", True)

        # 1. evaluate for the current round of agent 1 against all previous (including the current round) of agent 2
        i = round_num
        startswith_keyword = f"{prefix}{i}_"
        agent_model = None
        sampled_agent = None
        # Get 1st agent
        if(not self.OS):
            # Get the agents that has models' names starts with "history_<i>_"
            sampled_agent_startswith = utlst.get_startswith(models_names, startswith=startswith_keyword)
            # Get the latest agent in this round/generation
            sampled_agent = utlst.get_latest(sampled_agent_startswith)[0]
            # Load the model
            agent_model = self.archive.load(name=sampled_agent, env=self.eval_env, algorithm_class=algorithm_class)
        # The same but for OS stored
        else:
            # sampled_agent_startswith = utos.get_startswith(self.save_path, startswith=startswith_keyword)
            sampled_agent = os.path.join(agents_path, utos.get_latest(self.save_path, startswith=startswith_keyword)[0])  # Join it with the agent path
            agent_model = algorithm_class.load(sampled_agent, env=self.eval_env)

        # TODO: Make it in one loop (Easy)
        for j in range(round_num+1):
            print("---------------")
            print(f"Round: {i} vs {j}")
            opponent_startswith_keyword = f"{prefix}{j}_"
            sampled_opponent = None
            # Get 2nd agent
            if(not self.OS):
                # if(len(models_names) == 0): Not possible as we are evaluating after training
                sampled_opponent_startswith = utlst.get_startswith(opponent_models_names, startswith=opponent_startswith_keyword)
                sampled_opponent = utlst.get_latest(sampled_opponent_startswith)[0]

            else:
                # sampled_opponent_startswith = utos.get_startswith(self.eval_sample_path, startswith=opponent_startswith_keyword)
                sampled_opponent = os.path.join(opponents_path, utos.get_latest(self.eval_sample_path, startswith=opponent_startswith_keyword)[0])

            print(f"Model {sampled_agent} vs {sampled_opponent}")
            # Run evaluation n_eval_rep for each opponent
            eval_model_list = [sampled_opponent]
            # The current model vs the iterated model from the opponent (last opponent in each generation/round)
            # TODO: it is possible to change this agent_model to self.model as they are the same in this loop
            _, _, win_rate, _ = self._evaluate(agent_model, n_eval_episodes=n_eval_rep,
                                            deterministic=deterministic,
                                            sampled_opponents=eval_model_list)
            # Save the result to a matrix (nxm) -> n -agent, m -opponents -> Index by round number
            # Add this matrix to __init__
            # It will be redundent to have 2 matrices but it is fine
            self.evaluation_matrix[i,j] = win_rate
            print(f"win rate: {win_rate}")

        # 2. evaluate for the previous rounds of agent 1 against the current round of agent 2
        j = round_num
        opponent_startswith_keyword = f"{prefix}{j}_"
        sampled_opponent = None
        # Get 2nd agent
        if(not self.OS):
            # if(len(models_names) == 0): Not possible as we are evaluating after training
            sampled_opponent_startswith = utlst.get_startswith(opponent_models_names, startswith=opponent_startswith_keyword)
            sampled_opponent = utlst.get_latest(sampled_opponent_startswith)[0]

        else:
            # sampled_opponent_startswith = utos.get_startswith(self.eval_sample_path, startswith=opponent_startswith_keyword)
            sampled_opponent = os.path.join(opponents_path, utos.get_latest(self.eval_sample_path, startswith=opponent_startswith_keyword)[0])

        eval_model_list = [sampled_opponent]

        for i in range(round_num):
            print("---------------")
            print(f"Round: {i} vs {j}")
            startswith_keyword = f"{prefix}{i}_"
            agent_model = None
            
            # Get 1st agent
            if(not self.OS):
                # Get the agents that has models' names starts with "history_<i>_"
                sampled_agent_startswith = utlst.get_startswith(models_names, startswith=startswith_keyword)
                # Get the latest agent in this round/generation
                sampled_agent = utlst.get_latest(sampled_agent_startswith)[0]
                # Load the model
                agent_model = self.archive.load(name=sampled_agent, env=self.eval_env, algorithm_class=algorithm_class)
            # The same but for OS stored
            else:
                # sampled_agent_startswith = utos.get_startswith(self.save_path, startswith=startswith_keyword)
                sampled_agent = os.path.join(agents_path, utos.get_latest(self.save_path, startswith=startswith_keyword)[0])  # Join it with the agent path
                agent_model = algorithm_class.load(sampled_agent, env=self.eval_env)

            print(f"Model {sampled_agent} vs {sampled_opponent}")
            # Run evaluation n_eval_rep for each opponent
            # The current model vs the iterated model from the opponent (last opponent in each generation/round)
            _, _, win_rate, _ = self._evaluate(agent_model, n_eval_episodes=n_eval_rep,
                                            deterministic=deterministic,
                                            sampled_opponents=eval_model_list)
            # Save the result to a matrix (nxm) -> n -agent, m -opponents -> Index by round number
            self.evaluation_matrix[i,j] = win_rate
            print(f"win rate: {win_rate}")
            

    # Evaluate the whole matrix
    def compute_eval_matrix(self, prefix, num_rounds, opponents_path=None, agents_path=None, n_eval_rep=5, deterministic=False, algorithm_class=None):
        models_names = None

        # self.evaluation_matrix.append([])
        # Evaluate the model and save it (+ Regular evaluation)
        # self._save_model()
        # Evaluate the current model vs all the previous opponents
        # Get the list of all the previous opponents
        if(self.OS and (opponents_path is None or agents_path is None)):
            raise ValueError("Wrong value for opponent/agent path")

        if(not self.OS):
            opponent_archive = self.opponent_archive.get_sorted("random")
            opponent_models_names = opponent_archive[0] # 0 index to get just the names
            archive = self.archive.get_sorted("random")
            models_names = archive[0]

        else:
            self.eval_env.set_attr("OS", True)


        self.evaluation_matrix = np.zeros((num_rounds, num_rounds))
        for i in range(num_rounds):
            startswith_keyword = f"{prefix}{i}_"
            agent_model = None

            # Get 1st agent
            if(not self.OS):
                # Get the agents that has models' names starts with "history_<i>_"
                sampled_agent_startswith = utlst.get_startswith(models_names, startswith=startswith_keyword)
                # Get the latest agent in this round/generation
                sampled_agent = utlst.get_latest(sampled_agent_startswith)[0]
                # Load the model
                agent_model = self.archive.load(name=sampled_agent, env=self.eval_env, algorithm_class=algorithm_class)
            # The same but for OS stored
            else:
                # sampled_agent_startswith = utos.get_startswith(self.save_path, startswith=startswith_keyword)
                sampled_agent = os.path.join(agents_path, utos.get_latest(self.save_path, startswith=startswith_keyword)[0])  # Join it with the agent path
                agent_model = algorithm_class.load(sampled_agent, env=self.eval_env)


            for j in range(num_rounds):
                print(f"Round: {i} vs {j}")
                opponent_startswith_keyword = f"{prefix}{j}_"
                sampled_opponent = None
                # Get 2nd agent
                if(not self.OS):
                    # if(len(models_names) == 0): Not possible as we are evaluating after training
                    sampled_opponent_startswith = utlst.get_startswith(opponent_models_names, startswith=opponent_startswith_keyword)
                    sampled_opponent = utlst.get_latest(sampled_opponent_startswith)[0]

                else:
                    # sampled_opponent_startswith = utos.get_startswith(self.eval_sample_path, startswith=opponent_startswith_keyword)
                    sampled_opponent = os.path.join(opponents_path, utos.get_latest(self.eval_sample_path, startswith=opponent_startswith_keyword)[0])

                # Run evaluation n_eval_rep for each opponent
                eval_model_list = [sampled_opponent]
                # The current model vs the iterated model from the opponent (last opponent in each generation/round)
                _, _, win_rate, _ = self._evaluate(agent_model, n_eval_episodes=n_eval_rep,
                                                deterministic=deterministic,
                                                sampled_opponents=eval_model_list)
                # Save the result to a matrix (nxm) -> n -agent, m -opponents -> Index by round number
                # Add this matrix to __init__
                # It will be redundent to have 2 matrices but it is fine
                self.evaluation_matrix[i,j] = win_rate
        
        # For each generation/round previous to the current generation/round and including this round
        # for i in range(round_num+1):    # round_num+1 In order to include itself round
        #     startswith_keyword = f"{prefix}{round_num}"
        #     sampled_opponents = None            
        #     if(not self.OS):
        #         # if(len(models_names) == 0): Not possible as we are evaluating after training
        #         sampled_opponents_startswith = utlst.get_startswith(models_names, startswith=startswith_keyword)
        #         sampled_opponents = utlst.get_latest(sampled_opponents_startswith)[0]
        #     else:
        #         sampled_opponents_startswith = utos.get_startswith(self.eval_sample_path, startswith=startswith_keyword)
        #         sampled_opponents = utos.get_latest(sampled_opponents_startswith)[0]

        #     # Run evaluation n_eval_rep for each opponent
        #     eval_model_list = [sampled_opponents]
        #     # The current model vs the iterated model form the opponent (last opponent in each generation/round)
        #     _, _, win_rate = self._evaluate(self.model, n_eval_episodes=n_eval_rep,
        #                                     deterministic=deterministic,
        #                                     sampled_opponents=eval_model_list)
        #     # Save the result to a matrix (nxm) -> n -agent, m -opponents -> Index by round number
        #     # Add this matrix to __init__
        #     # It will be redundent to have 2 matrices but it is fine
        #     self.evaluation_matrix[-1].append(win_rate)

            # --------------------- if each version in the generation/round of 1st agent vs each version in the generation of the 2nd agent ---------------------
            # if(not self.OS):
            #     # if(len(models_names) == 0): Not possible as we are evaluating after training
            #     sampled_opponents = utlst.get_startswith(models_names, startswith=startswith_keyword)
            # if(self.OS):
            #     sampled_opponents = utos.get_startswith(self.eval_sample_path, startswith=startswith_keyword)
            # For each model in the generation/round for opponent
            # for opponent in sampled_opponents:
            #     # For each model in the generation/round for self (the agent that I am evaluating)
            #     for model in sampled_self:
            #         eval_model_list = [opponent for j in range]
            #         # To make it work, change this function to include model
            #         _, _, win_rate = self._evaluate(model, n_eval_episodes=n_eval_rep,
            #                                         deterministic=deterministic,
            #                                         sampled_opponents=eval_model_list)
            # ---------------------------------------------------------------------------------------------


        # return self.eval
        # Save result of this matrix to wandb as heatmap with the current sizes
        # TODO
        # Save the round number with the name of the agent in a list
        # Save this list to a txt file -> This shows the generation/round number and the agent name 

    # This last agent
    # Post evaluate the model against all the opponents from opponents_path
    # TODO: enable retrieving the agents from the archive
    def post_eval(self, opponents_path, startswith_keyword="history", n_eval_rep=3, deterministic=None):
        deterministic = self.deterministic if deterministic is None else deterministic  # https://stackoverflow.com/questions/66455636/what-does-deterministic-true-in-stable-baselines3-library-means
        opponents_models_names = utos.get_sorted(opponents_path, startswith_keyword, utsrt.sort_steps)
        opponents_models_path = [os.path.join(opponents_path, f) for f in opponents_models_names]
        eval_return_list = []
        # self.eval_env.OS = True
        self.eval_env.set_attr("OS", True)
        self.OS = True
        for i, o in enumerate(opponents_models_path):
            eval_model_list = [o for _ in range(n_eval_rep)]
            # Doing this as only logger.record doesn't work, I think I need to call something else for Wandb callback
            # TODO: Fix the easy method (the commented) without using evaluate() function to make the code better
            _, _, win_rate, _ = self._evaluate(self.model, n_eval_episodes=n_eval_rep,
                                            deterministic=deterministic,
                                            sampled_opponents=eval_model_list)
            evaluation_result = win_rate
            wandb.log({f"{self.agent_name}/post_eval/opponent_idx": i})
            wandb.log({f"{self.agent_name}/post_eval/win_rate": evaluation_result})

            
            # self.logger.record("post_eval/opponent_idx", i)
            # evaluation_result = self._evaluate_policy_core(logger_prefix="post_eval", 
            #                                                 n_eval_episodes=n_eval_opponent,
            #                                                 deterministic=deterministic,
            #                                                 sampled_opponents=eval_model_list,
            #                                                 override=True)
            
            eval_return_list.append(evaluation_result)
        data = [[x, y] for (x, y) in zip([i for i in range(len(opponents_models_path))], eval_return_list)]
        table = wandb.Table(data=data, columns = ["opponent idx", "win-rate"])
        wandb.log({f"{self.agent_name}/post_eval/table": wandb.plot.line(table, "opponent idx", "win-rate", title=f"Post evaluation {self.agent_name}")})
        # self.eval_env.OS = False
        self.eval_env.set_attr("OS", False)
        self.OS = False
        return eval_return_list

    def agentVopponentOS(self, agent_path, opponent_path, n_eval_rep=5, deterministic=None, algorithm_class=None):
        deterministic = self.deterministic if deterministic is None else deterministic  # https://stackoverflow.com/questions/66455636/what-does-deterministic-true-in-stable-baselines3-library-means

        self.eval_env.set_attr("OS", True)
        self.OS = True

        agent_model = algorithm_class.load(agent_path, env=self.eval_env)
        eval_model_list = [opponent_path]
        res =  self._evaluate(agent_model, n_eval_episodes=n_eval_rep,
                              deterministic=deterministic,
                              sampled_opponents=eval_model_list)


        self.eval_env.set_attr("OS", False)
        self.OS = False

        return res

    def agentVopponentArchive(self, agent_name, opponent_name, n_eval_rep=5, deterministic=None, algorithm_class=None):
        deterministic = self.deterministic if deterministic is None else deterministic  # https://stackoverflow.com/questions/66455636/what-does-deterministic-true-in-stable-baselines3-library-means

        self.eval_env.set_attr("OS", False)
        self.OS = False

        agent_model = self.archive.load(name=agent_name, env=self.eval_env, algorithm_class=algorithm_class)
        eval_model_list = [opponent_name]
        res =  self._evaluate(agent_model, n_eval_episodes=n_eval_rep,
                              deterministic=deterministic,
                              sampled_opponents=eval_model_list)

        self.eval_env.set_attr("OS", True)
        self.OS = True

        return res