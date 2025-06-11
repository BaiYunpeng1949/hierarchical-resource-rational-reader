import os

import os.path
import time
import yaml
import csv
import json
import copy
import pickle
import pandas as pd
import numpy as np
from typing import Callable
from collections import OrderedDict
import matplotlib.pyplot as plt
from scipy.stats import linregress
import seaborn as sns
from multiprocessing import shared_memory
from mpl_toolkits.mplot3d import Axes3D

import gymnasium as gym
from gymnasium import spaces
import torch as th
from torch import nn

import step5.utils.constants as constants

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# from modules.rl_envs.word_activation_v0218.WordActivationEnvV0218 import WordActivationRLEnv
from sentence_read_v0604.SentenceReadingEnv import SentenceReadingUnderTimePressureEnv
from text_read_v0604.TextReadEnv import TextReadingUnderTimePressureEnv



_MODES = {
    'train': 'train',
    'continual_train': 'continual_train',
    'test': 'test',
    'debug': 'debug',
    'grid_test': 'grid_test'
}


class StatefulInformationExtractor(BaseFeaturesExtractor):

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        # We assume a 1D tensor

        self.net = nn.Sequential(
            nn.Linear(in_features=observation_space.shape[0], out_features=features_dim),
            nn.LeakyReLU(),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.net(observations)


def linear_schedule(initial_value: float, min_value: float, threshold: float = 1.0) -> Callable[[float], float]:
    """
    Linear learning rate schedule. Adapted from the example at
    https://stable-baselines3.readthedocs.io/en/master/guide/examples.html#learning-rate-schedule

    :param initial_value: Initial learning rate.
    :param min_value: Minimum learning rate.
    :param threshold: Threshold (of progress) when decay begins.
    :return: schedule that computes the current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        if progress_remaining > threshold:
            return initial_value
        else:
            return min_value + (progress_remaining/threshold) * (initial_value - min_value)

    return func


class RL:
    def __init__(self, config_file):
        """
        This is the reinforcement learning pipeline where MuJoCo environments are created, and models are trained and tested.
        This pipeline is derived from my trials: context_switch.

        Args:
            config_file: the YAML configuration file that records the configurations.
        """
        # Read the configurations from the YAML file.
        with open(config_file) as f:
            self._config = yaml.load(f, Loader=yaml.FullLoader)

        try:
            self._config_rl = self._config['rl']
        except ValueError:
            print('Invalid configurations. Check your config.yaml file.')

        # Specify the pipeline mode.
        self._mode = self._config_rl['mode']

        # Print the configuration
        if self._mode == _MODES['continual_train'] or self._mode == _MODES['test']:
            print(
                f"    The loaded model checkpoints folder name is: {self._config_rl['train']['checkpoints_folder_name']}\n"
                f"    The loaded model checkpoint is: {self._config_rl['test']['loaded_model_name']}\n"
            )

        # Get the environment class
        # env_class = TextReadingUnderTimePressureEnv    
        env_class = SentenceReadingUnderTimePressureEnv

        # Read the total dataset if training the general oculomotor controller model.
        # if env_class == WordActivationRLEnv:        NOTE: uncomment this later if the word recognizer agent is needed
        #     self._env = WordActivationRLEnv()
        #     def make_env():
        #         env = WordActivationRLEnv()
        #         # env = Monitor(env)
        #         return env
        #     # Initialise parallel environments
        #     self._parallel_envs = make_vec_env(
        #         env_id=make_env,
        #         # env_id=self._env.__class__,
        #         n_envs=self._config_rl['train']["num_workers"],
        #         seed=42,
        #         vec_env_cls=SubprocVecEnv,
        #     )
        if env_class == SentenceReadingUnderTimePressureEnv:
            self._env = SentenceReadingUnderTimePressureEnv()
            def make_env():
                env = SentenceReadingUnderTimePressureEnv()
                # env = Monitor(env)
                return env
            # Initialise parallel environments
            self._parallel_envs = make_vec_env(
                env_id=make_env,
                # env_id=self._env.__class__,
                n_envs=self._config_rl['train']["num_workers"],
                seed=42,
                vec_env_cls=SubprocVecEnv,
            )
        elif env_class == TextReadingUnderTimePressureEnv:
            self._env = TextReadingUnderTimePressureEnv()
            def make_env():
                env = TextReadingUnderTimePressureEnv()
                # env = Monitor(env)
                return env
            # Initialise parallel environments
            self._parallel_envs = make_vec_env(
                env_id=make_env,
                # env_id=self._env.__class__,
                n_envs=self._config_rl['train']["num_workers"],
                seed=42,
                vec_env_cls=SubprocVecEnv,
            )
        else:
            raise ValueError(f'Invalid environment class {env_class}.')

        # Identify the modes and specify corresponding initiates.
        # Train the model, and save the logs and modes at each checkpoint.
        if self._mode == _MODES['train']:
            # Pipeline related variables.
            self._training_logs_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'training', 'logs')
            self._checkpoints_folder_name = self._config_rl['train']['checkpoints_folder_name']
            self._models_save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'training', 'saved_models', self._checkpoints_folder_name)
            self._models_save_file_final = os.path.join(self._models_save_path, self._config_rl['train']['checkpoints_folder_name'])
            # RL training related variable: total time-steps.
            self._total_timesteps = self._config_rl['train']['total_timesteps']

            # Configure the model - HRL - Ocular motor control
            # if isinstance(self._env, SampleFixationVersion1) or isinstance(self._env, SampleFixationVersion2):
            # if isinstance(self._env, WordActivationRLEnv) or isinstance(self._env, SentenceReadingEnv) or isinstance(self._env, TextComprehensionEnv):      NOTE: uncomment this later if the word recognizer agent is needed
            if isinstance(self._env, SentenceReadingUnderTimePressureEnv) or isinstance(self._env, TextReadingUnderTimePressureEnv):
                policy_kwargs = dict(
                    features_extractor_class=StatefulInformationExtractor,       
                    features_extractor_kwargs=dict(features_dim=128),
                    activation_fn=th.nn.LeakyReLU,
                    net_arch=[512, 512],
                    log_std_init=-1.0,
                    normalize_images=False
                )
                policy = "MlpPolicy"     # Choose from CnnPolicy, MlpPolicy, MultiInputPolicy
            else:
                raise ValueError(f'Invalid environment {self._env}.')

            self._model = PPO(
                policy=policy,
                env=self._parallel_envs,
                verbose=1,
                policy_kwargs=policy_kwargs,
                tensorboard_log=self._training_logs_path,
                n_steps=self._config_rl['train']["num_steps"],
                batch_size=self._config_rl['train']["batch_size"],
                # target_kl=self._config_rl['train']["target_kl"],
                # clip_range=linear_schedule(self._config_rl['train']["clip_range"]),
                # ent_coef=self._config_rl['train']["ent_coef"],
                # n_epochs=self._config_rl['train']["n_epochs"],
                learning_rate=linear_schedule(
                    initial_value=float(self._config_rl['train']["learning_rate"]["initial_value"]),
                    min_value=float(self._config_rl['train']["learning_rate"]["min_value"]),
                    threshold=float(self._config_rl['train']["learning_rate"]["threshold"]),
                ),
                gamma=self._config_rl['train']['gamma'],
                device=self._config_rl['train']["device"],
                seed=42,
            )
        # Load the pre-trained models and test.
        elif self._mode == _MODES['test'] or self._mode == _MODES['continual_train'] or self._mode == _MODES['grid_test']:
            # Pipeline related variables.
            self._loaded_model_name = self._config_rl['test']['loaded_model_name']
            self._checkpoints_folder_name = self._config_rl['train']['checkpoints_folder_name']
            self._models_save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'training', 'saved_models', self._checkpoints_folder_name)
            self._loaded_model_path = os.path.join(self._models_save_path, self._loaded_model_name)

            print(f"The loaded model's path is: {self._loaded_model_path}")

            # RL testing related variable: number of episodes and number of steps in each episode
            self._num_episodes = self._config_rl['test']['num_episodes']
            self._num_steps = self._env.ep_len

            # Load the model
            if self._mode == _MODES['test'] or self._mode == _MODES['grid_test']:
                self._model = PPO.load(self._loaded_model_path, self._env)
            elif self._mode == _MODES['continual_train']:
                # Logistics.
                # Pipeline related variables.
                self._training_logs_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'training', 'logs')
                self._checkpoints_folder_name = self._config_rl['train']['checkpoints_folder_name']
                self._models_save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'training', 'saved_models', self._checkpoints_folder_name)

                self._models_save_file_final = os.path.join(self._models_save_path,
                                                            self._config_rl['train']['checkpoints_folder_name'])
                # RL training related variable: total time-steps.
                self._total_timesteps = self._config_rl['train']['total_timesteps']
                # Model loading and register.
                self._model = PPO.load(self._loaded_model_path)
                self._model.set_env(self._parallel_envs)
        # The MuJoCo environment debugs. Check whether the environment and tasks work as designed.
        elif self._mode == _MODES['debug']:
            self._num_episodes = self._config_rl['test']['num_episodes']
            self._loaded_model_name = 'debug'
            # self._num_steps = self._env.num_steps
        # The MuJoCo environment demo display with user interactions, such as mouse interactions.
        else:
            raise ValueError(f'Invalid mode {self._mode}.')

    def _train(self):
        """Add comments """
        # Save a checkpoint every certain steps, which is specified by the configuration file.
        # Ref: https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html
        # To account for the multi-viewer_models' steps, save_freq = max(save_freq // n_envs, 1).
        save_freq = self._config_rl['train']['save_freq']
        n_envs = self._config_rl['train']['num_workers']
        save_freq = max(save_freq // n_envs, 1)
        checkpoint_callback = CheckpointCallback(
            save_freq=save_freq,
            save_path=self._models_save_path,
            name_prefix='rl_model',
        )

        self._model.learn(
                total_timesteps=self._total_timesteps,
                callback=checkpoint_callback,
                # log_interval=1,
            )

    def _continual_train(self):
        """
        This method performs the continual trainings.
        Ref: https://github.com/hill-a/stable-baselines/issues/599#issuecomment-569393193
        """
        save_freq = self._config_rl['train']['save_freq']
        n_envs = self._config_rl['train']['num_workers']
        save_freq = max(save_freq // n_envs, 1)
        checkpoint_callback = CheckpointCallback(
            save_freq=save_freq,
            save_path=self._models_save_path,
            name_prefix='rl_model_continual',
        )

        self._model.learn(
            total_timesteps=self._total_timesteps,
            callback=checkpoint_callback,
            log_interval=1,
            tb_log_name=self._config_rl['test']['continual_logs_name'],
            reset_num_timesteps=False,
        )

        # Save the model as the rear guard.
        self._model.save(self._models_save_file_final)

    def run(self):
        """
        This method helps run the RL pipeline.
        Call it.
        """
        # Check train or not.
        if self._mode == _MODES['train']:
            self._train()
        elif self._mode == _MODES['continual_train']:
            self._continual_train()
        elif self._mode == _MODES['test'] or self._mode == _MODES['debug']:
            # if isinstance(self._env, WordActivationRLEnv):    NOTE: uncomment this later if the word recognizer agent is needed
            #     self._word_activation_test()
            if isinstance(self._env, SentenceReadingUnderTimePressureEnv):
                self._sentence_reading_test()
            elif isinstance(self._env, TextReadingUnderTimePressureEnv):
                self._text_comprehension_test()
            else:
                raise ValueError(f'Invalid environment {self._env}.')
        else:
            raise ValueError(f'Invalid mode {self._mode}.')
    
    def _sentence_reading_test(self):
        """
        Test the sentence reading environment.
        """
        
        if self._mode == _MODES['debug']:
            print('\nSimulation -- Debug mode: ')
        elif self._mode == _MODES['test']:
            print('\nSimulation -- Test mode: ')

        # Start the timer
        start_time = time.time()

        # Prepare to collect logs
        all_episode_logs = []

        for episode in range(1, self._num_episodes + 1):
            obs, info = self._env.reset()
            done = False
            score = 0

            while not done:
                if self._mode == _MODES['debug']:
                    action = self._env.action_space.sample()
                elif self._mode == _MODES['test']:
                    action, _states = self._model.predict(obs, deterministic=True)
                    # action, _states = self._model.predict(obs, deterministic=False)
                else:
                    raise ValueError(f'Invalid mode {self._mode}.')

                obs, reward, done, truncated, info = self._env.step(action)
                score += reward

            print(
                f'Episode:{episode}     Score:{score} \n'
                f'{"-" * 50}\n'
            )

            # Collect episode log after each episode
            episode_log = self._env.get_episode_log()
            episode_log['score'] = score
            all_episode_logs.append(episode_log)

        print(f'Time elapsed for running the DEBUG/TEST: {time.time() - start_time} seconds')

        # Save logs to JSON file in the specified folder structure
        # Folder: simulated_results/<checkpoint_folder>__<model_name>__<num_episodes>
        checkpoint_folder = self._checkpoints_folder_name if hasattr(self, '_checkpoints_folder_name') else 'unknown_checkpoint'
        model_name = self._loaded_model_name if hasattr(self, '_loaded_model_name') else 'unknown_model'
        num_episodes = self._num_episodes if hasattr(self, '_num_episodes') else 'unknown_episodes'
        results_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'sentence_read_v0604',
            'simulated_results',
            f'{checkpoint_folder}__{model_name}__{num_episodes}'
        )
        os.makedirs(results_dir, exist_ok=True)
        results_path = os.path.join(results_dir, 'simulated_episode_logs.json')
        with open(results_path, 'w') as f:
            json.dump(all_episode_logs, f, indent=2)
        print(f"Saved episode logs to {results_path}")

    def _text_comprehension_test(self):
        """
        Test the text comprehension environment.
        """
        
        if self._mode == _MODES['debug']:
            print('\nSimulation -- Debug mode: ')
        elif self._mode == _MODES['test']:
            print('\nSimulation -- Test mode: ')

         # Start the timer
        start_time = time.time()

        # Prepare to collect logs
        all_episode_logs = []

        for episode in range(1, self._num_episodes + 1):
            obs, info = self._env.reset()
            done = False
            score = 0

            while not done:
                if self._mode == _MODES['debug']:
                    action = self._env.action_space.sample()
                elif self._mode == _MODES['test']:
                    action, _states = self._model.predict(obs, deterministic=True)
                    # action, _states = self._model.predict(obs, deterministic=False)
                else:
                    raise ValueError(f'Invalid mode {self._mode}.')

                obs, reward, done, truncated, info = self._env.step(action)
                score += reward

            print(
                f'Episode:{episode}     Score:{score} \n'
                f'{"-" * 50}\n'
            )

            # Collect episode log after each episode
            episode_log = self._env.get_episode_log()
            episode_log['score'] = score
            all_episode_logs.append(episode_log)

        print(f'Time elapsed for running the DEBUG/TEST: {time.time() - start_time} seconds')

        # Save logs to JSON file in the specified folder structure
        # Folder: simulated_results/<checkpoint_folder>__<model_name>__<num_episodes>
        checkpoint_folder = self._checkpoints_folder_name if hasattr(self, '_checkpoints_folder_name') else 'unknown_checkpoint'
        model_name = self._loaded_model_name if hasattr(self, '_loaded_model_name') else 'unknown_model'
        num_episodes = self._num_episodes if hasattr(self, '_num_episodes') else 'unknown_episodes'
        results_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'text_read_v0604',
            'simulated_results',
            f'{checkpoint_folder}__{model_name}__{num_episodes}'
        )
        os.makedirs(results_dir, exist_ok=True)
        results_path = os.path.join(results_dir, 'simulated_episode_logs.json')
        with open(results_path, 'w') as f:
            json.dump(all_episode_logs, f, indent=2)
        print(f"Saved episode logs to {results_path}")

    def __del__(self):
        # Close the environment.
        self._env.close()

        # Visualize the destructor.
        print(
            '\n\n***************************** RL pipeline ends. The MuJoCo environment of the pipeline has been destructed *************************************'
        )