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

from step5.utils import constants as constants
from step5.utils import auxiliaries as aux
from utils import plot_word_activation_without_vision_figures as plot_word_activation_figures

from step5.modules.rl_envs.SupervisoryControllerEnv_v1101_2 import SupervisoryControllerEnv

from step5.modules.rl_envs.SentenceLevelControllerEnv_v1014 import SentenceLevelControllerEnv

from modules.rl_envs.GeneralOculomotorControllerEnv_v1126 import GeneralOculomotorControllerEnv
from modules.rl_envs.OMCRLEnvV0128 import OculomotorControllerRLEnv
from modules.rl_envs.word_activation_v0218.WordActivationEnvV0218 import WordActivationRLEnv

_MODES = {
    'train': 'train',
    'continual_train': 'continual_train',
    'test': 'test',
    'debug': 'debug',
    'grid_test': 'grid_test'
}


class VisionExtractor(BaseFeaturesExtractor):

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        """
        The custom cnn feature extractor.
        Ref: https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html#custom-feature-extractor
        :param observation_space: (gym.Space)
        :param features_dim: (int) Number of features extracted.
            This corresponds to the number of unit for the last layer.
        """
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper

        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=n_input_channels, out_channels=8, kernel_size=(3, 3), padding=(1, 1), stride=(2, 2)),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=(1, 1), stride=(2, 2)),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=(1, 1), stride=(2, 2)),
            nn.LeakyReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            # (batch_size, hidden_channels * changed_width * changed_height)
            n_flatten = self.cnn(th.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))


class FovealPeripheralVisionExtractor(BaseFeaturesExtractor):

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        """
        The custom CNN feature extractor. Separately process fovea and peripheral views when they do not have the same shape.
        Key assumptions: Both foveal and peripheral views are grayscale images with the shape of (1, H, W).
        Ref: https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html#custom-feature-extractor
        :param observation_space: (gym.Space)
        :param features_dim: (int) Number of features extracted.
            This corresponds to the number of units for the last layer.
        """
        super().__init__(observation_space, features_dim)

        n_input_channels = 1

        # Define separate convolutional layers for foveal and peripheral views
        self.foveal_cnn = nn.Sequential(
            nn.Conv2d(in_channels=n_input_channels, out_channels=8, kernel_size=(3, 3), padding=(1, 1), stride=(2, 2)),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=(1, 1), stride=(2, 2)),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=(1, 1), stride=(2, 2)),
            nn.LeakyReLU(),
            nn.Flatten(),
        )

        self.peripheral_cnn = nn.Sequential(
            nn.Conv2d(in_channels=n_input_channels, out_channels=8, kernel_size=(3, 3), padding=(1, 1), stride=(2, 2)),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=(1, 1), stride=(2, 2)),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=(1, 1), stride=(2, 2)),
            nn.LeakyReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            # Sample an input to determine the output size of the CNNs
            sample_input = observation_space.sample()[None]  # Add batch dimension
            n_flatten_foveal = self.foveal_cnn(th.as_tensor(sample_input[:, 0:1, :, :]).float()).shape[1]
            n_flatten_peripheral = self.peripheral_cnn(th.as_tensor(sample_input[:, 1:2, :, :]).float()).shape[1]

        n_flatten = n_flatten_foveal + n_flatten_peripheral

        # Define the final linear layers
        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        foveal_features = self.foveal_cnn(observations[:, 0:1, :, :])
        peripheral_features = self.peripheral_cnn(observations[:, 1:2, :, :])

        # Concatenate the features
        combined_features = th.cat((foveal_features, peripheral_features), dim=1)

        return self.linear(combined_features)


class ProprioceptionExtractor(BaseFeaturesExtractor):

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        """
        Ref: Aleksi - https://github.com/BaiYunpeng1949/uitb-headsup-computing/blob/bf58d715b99ffabae4c2652f20898bac14a532e2/huc/RL.py#L75
        """
        super().__init__(observation_space, features_dim)
        # We assume a 1D tensor

        self.net = nn.Sequential(
            nn.Linear(in_features=observation_space.shape[0], out_features=features_dim),
            nn.LeakyReLU(),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.net(observations)


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


class NumericFeatureExtractor(BaseFeaturesExtractor):

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        # We assume a 1D tensor

        self.net = nn.Sequential(
            nn.Linear(in_features=observation_space.shape[0], out_features=features_dim),
            nn.LeakyReLU(),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.net(observations)


class NumericFeatureExtractorMultiLayers(BaseFeaturesExtractor):
    """The numeric feature extractor with more layers."""

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 32):
        super().__init__(observation_space, features_dim)
        # We assume a 1D tensor

        # Define the network architecture
        self.net = nn.Sequential(
            nn.Linear(in_features=observation_space.shape[0], out_features=128),
            nn.LeakyReLU(),
            nn.Linear(in_features=128, out_features=64),
            nn.LeakyReLU(),
            nn.Linear(in_features=64, out_features=features_dim),
            nn.LeakyReLU(),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.net(observations)

class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(
            self,
            observation_space: spaces.Dict,
            vision_features_dim: int = 256,
            stateful_information_features_dim: int = 256,
    ):
        """
        Custom combined feature extractor for different modalities.
        """
        super().__init__(observation_space, features_dim=vision_features_dim+stateful_information_features_dim)

        self.extractors = nn.ModuleDict({
            "vision": VisionExtractor(observation_space["vision"], vision_features_dim),
            "stateful information": StatefulInformationExtractor(observation_space["stateful information"], stateful_information_features_dim),
        })

    def forward(self, observations) -> th.Tensor:
        encoded_tensor_list = []

        for key, extractor in self.extractors.items():
            encoded_features = extractor(observations[key])
            encoded_tensor_list.append(encoded_features)

        combined_features = th.cat(encoded_tensor_list, dim=1)
        return combined_features


class NoVisionCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict, proprioception_features_dim: int = 256, stateful_information_features_dim: int = 256):

        super().__init__(observation_space, features_dim=proprioception_features_dim+stateful_information_features_dim)

        self.extractors = nn.ModuleDict({
            "proprioception": ProprioceptionExtractor(observation_space["proprioception"], proprioception_features_dim),
            "stateful information": StatefulInformationExtractor(observation_space["stateful information"], stateful_information_features_dim),
        })

    def forward(self, observations) -> th.Tensor:
        encoded_tensor_list = []

        # self.extractors contain nn.Modules that do all the processing.
        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))
        # Return a (B, features_dim=vision_features_dim+proprioception_features_dim) PyTorch tensor, where B is batch dimension.
        return th.cat(encoded_tensor_list, dim=1)


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
        env_class = WordActivationRLEnv # OculomotorControllerRLEnv # GeneralOculomotorControllerEnv           # GeneralOculomotorControllerEnv, SentenceLevelControllerEnv, SupervisoryControllerEnv

        # Load the dataset (if needed)
        shared_dataset_metadata_of_stimuli = None
        shared_dataset_encoded_lexicon = None

        # Read the total dataset if training the general oculomotor controller model.
        if env_class == OculomotorControllerRLEnv:   # GeneralOculomotorControllerEnv:
            # Load the dataset
            print('Loading the dataset...')
            shared_dataset_metadata_of_stimuli, shared_dataset_encoded_lexicon, self._dataset_mode = aux.load_oculomotor_controller_dataset(config=self._config)
            print('Dataset loaded.')

            # Create an instance of the environment for use in other methods
            self._env = OculomotorControllerRLEnv(   # GeneralOculomotorControllerEnv(
                shared_dataset_metadata_of_image_stimuli=shared_dataset_metadata_of_stimuli,
                shared_dataset_encoded_lexicon=shared_dataset_encoded_lexicon
            )
            # self._env = GeneralOculomotorControllerEnv(
            #     shared_metadata_name=self._metadata_shm.name,
            #     metadata_size=len(stimuli_metadata_bytes),
            #     shared_lexicon_name=self._lexicon_shm.name,
            #     lexicon_size=len(encoded_lexicon_bytes)
            # )

            # Define the environment creation function
            def make_env():
                return OculomotorControllerRLEnv( # GeneralOculomotorControllerEnv(
                    shared_dataset_metadata_of_image_stimuli=shared_dataset_metadata_of_stimuli,
                    shared_dataset_encoded_lexicon=shared_dataset_encoded_lexicon
                )
                # return GeneralOculomotorControllerEnv(
                #     shared_metadata_name=self._metadata_shm.name,
                #     metadata_size=len(stimuli_metadata_bytes),
                #     shared_lexicon_name=self._lexicon_shm.name,
                #     lexicon_size=len(encoded_lexicon_bytes)
                # )
            
            # Initialise parallel environments
            self._parallel_envs = make_vec_env(
                env_id=make_env,
                n_envs=self._config_rl['train']["num_workers"],
                seed=42,
                vec_env_cls=SubprocVecEnv,
            )
        elif env_class == WordActivationRLEnv:
            self._env = WordActivationRLEnv()
            def make_env():
                env = WordActivationRLEnv()
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
        elif env_class == SentenceLevelControllerEnv:
            # Get an env instance for further constructing parallel environments.
            self._env = SentenceLevelControllerEnv()    # SentenceLevelControllerEnv()    # SupervisoryControllerEnv()  

            # Initialise parallel environments      
            self._parallel_envs = make_vec_env(
                env_id=self._env.__class__,
                n_envs=self._config_rl['train']["num_workers"],
                seed=42,
                vec_env_cls=SubprocVecEnv,
            )
        elif env_class == SupervisoryControllerEnv:
            # Get an env instance for further constructing parallel environments.
            self._env = SupervisoryControllerEnv()    # SentenceLevelControllerEnv()    # SupervisoryControllerEnv()  

            # Initialise parallel environments      
            self._parallel_envs = make_vec_env(
                env_id=self._env.__class__,
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
            if isinstance(self._env, GeneralOculomotorControllerEnv) or isinstance(self._env, OculomotorControllerRLEnv):
                policy_kwargs = dict(
                    features_extractor_class=CustomCombinedExtractor,       # Previous -- NumericFeatureExtractor, features_dim=128
                    features_extractor_kwargs=dict(
                        vision_features_dim=128,
                        stateful_information_features_dim=128
                    ),       # NumericFeatureExtractorMultiLayers, features_dim=32
                    activation_fn=th.nn.LeakyReLU,
                    net_arch=[512, 512],
                    log_std_init=-1.0,
                    normalize_images=False
                )
                policy = "MultiInputPolicy"     # Choose from CnnPolicy, MlpPolicy, MultiInputPolicy
            elif isinstance(self._env, WordActivationRLEnv):
                policy_kwargs = dict(
                    features_extractor_class=StatefulInformationExtractor,       
                    features_extractor_kwargs=dict(features_dim=128),
                    activation_fn=th.nn.LeakyReLU,
                    net_arch=[512, 512],
                    log_std_init=-1.0,
                    normalize_images=False
                )
                policy = "MlpPolicy"     # Choose from CnnPolicy, MlpPolicy, MultiInputPolicy
            elif isinstance(self._env, SupervisoryControllerEnv):
                policy_kwargs = dict(
                    features_extractor_class=StatefulInformationExtractor,
                    features_extractor_kwargs=dict(features_dim=128),
                    activation_fn=th.nn.LeakyReLU,
                    net_arch=[512, 512],
                    log_std_init=-1.0,
                    normalize_images=False
                )
                policy = "MlpPolicy"  # Choose from CnnPolicy, MlpPolicy, MultiInputPolicy
            elif isinstance(self._env, SentenceLevelControllerEnv):
                policy_kwargs = dict(
                    features_extractor_class=StatefulInformationExtractor,
                    features_extractor_kwargs=dict(features_dim=32),
                    activation_fn=th.nn.LeakyReLU,
                    net_arch=[128, 128],
                    log_std_init=-1.0,
                    normalize_images=False
                )
                policy = "MlpPolicy"  # Choose from CnnPolicy, MlpPolicy, MultiInputPolicy
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

        # try:
        #     # Train the RL model and save the logs. The Algorithm and policy were given,
        #     # but it can always be upgraded to a more flexible pipeline later.
        #     self._model.learn(
        #         total_timesteps=self._total_timesteps,
        #         callback=checkpoint_callback,
        #     )
        # finally:
        #     # Ensure environments are closed
        #     self._parallel_envs.close()

        #     # Clean up the shared memory
        #     self._metadata_shm.close()
        #     self._metadata_shm.unlink()
        #     self._lexicon_shm.close()
        #     self._lexicon_shm.unlink()

    @staticmethod
    def _encode_characters_in_word(word):

        """Encode the word characters into a 16-dim vector from 0 to 1 """
        encoded_word_characters = [-1] * constants.MAX_WORD_LEN

        # Encode every letter in the word using the full ASCII table from 0 to 127
        for idx, character in enumerate(word):
            if idx >= constants.MAX_WORD_LEN:
                break
            encoded_word_characters[idx] = (ord(character) - 32) / (126 - 32)
            # Normalize to range [0, 1] using printable ASCII range

        return encoded_word_characters

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

    def _oculomotor_controller_test(self, grid_test_params=None):
        """
        This method generates the RL env testing results with or without a pre-trained RL model in a manual way.
        """
        if self._mode == _MODES['debug']:
            print('\nThe MuJoCo env and tasks baseline: ')
        elif self._mode == _MODES['test']:
            print('\nThe pre-trained RL model testing: ')

        # Start the timer
        start_time = time.time()

        logger = []

        reward_col = "Reward"
        # Create column names dynamically, e.g., ['A', 'B', 'C']
        column_names = [chr(65 + i) for i in range(self._env.action_space.shape[0])] + [reward_col]  # 65 is the ASCII code for 'A'
        # Initialize an empty DataFrame with these columns
        df = pd.DataFrame(columns=column_names)

        # Some aggregated metrics' lists
        total_num_successful_word_recognitions = 0
        total_num_failed_word_recognitions = 0
        list_of_num_steps_for_word_recognition = []
        list_of_target_words_lengths = []

        for episode in range(1, self._num_episodes + 1):    # TODO check from here
            if grid_test_params is not None:
                obs, info = self._env.reset(ep_idx=episode, w_penalty=grid_test_params['w_penalty'])
            else:
                obs, info = self._env.reset()
            done = False
            score = 0

            # Get the fixations from the reset to record the initial fixation
            last_fixation = self._env.fixations_logger[-1]
            # Create a log entry and append a copy of last_fixation to avoid reference issues
            log_entry = (episode,) + copy.deepcopy(last_fixation)
            logger.append(log_entry)

            word_recognition_step = 0

            while not done:
                if self._mode == _MODES['debug']:
                    action = self._env.action_space.sample()
                elif self._mode == _MODES['test'] or self._mode == _MODES['grid_test']:
                    action, _states = self._model.predict(obs, deterministic=True)
                else:
                    raise ValueError(f'Invalid mode {self._mode}.')
                obs, reward, done, truncated, info = self._env.step(action)
                score += reward

                # Update the word recognition step
                word_recognition_step += 1

                # Flatten the list of tuples and prepend the episode number
                last_fixation = self._env.fixations_logger[-1]

                # Create a log entry and append a copy of last_fixation to avoid reference issues
                log_entry = (episode,) + copy.deepcopy(last_fixation)
                logger.append(log_entry)

                # Record data - include action and reward in the row
                action_list = list(action) if hasattr(action, '__iter__') and not isinstance(action, str) else [
                    action]  # Ensure action is list-like and not a string
                row_to_add = action_list + [reward]  # Append reward to the action values

                # Convert row_to_add into a DataFrame with appropriate columns before concatenating
                row_df = pd.DataFrame([row_to_add], columns=df.columns)
                df = pd.concat([df, row_df], ignore_index=True)

            if score > 0:
                total_num_successful_word_recognitions += 1
            else:
                total_num_failed_word_recognitions += 1
            
            list_of_num_steps_for_word_recognition.append(word_recognition_step)

            list_of_target_words_lengths.append(len(self._env.target_word))

            print(
                f'Episode:{episode}     Score:{score} \n'
                f'{"-"*50}\n'
            )

        # Print the time elapsed
        print(f"The recognition success percentage is: {round((total_num_successful_word_recognitions / self._num_episodes) * 100, 2)}%. "
              f"Average number of steps for word recognition is: {np.mean(list_of_num_steps_for_word_recognition)}. "
              f"Average target word length is: {np.mean(list_of_target_words_lengths)}.")
        print(f'Time elapsed for running the DEBUG/TEST: {time.time() - start_time} seconds')

        if self._mode == _MODES['test'] or self._mode == _MODES['grid_test']:
            # Store the data
            root_path = os.path.dirname(os.path.abspath(__file__))
            rl_model_name = self._config_rl['train']['checkpoints_folder_name'] + '_' + self._config_rl['test']['loaded_model_name']
            folder_path = os.path.join(root_path, "data", "sim_results", rl_model_name)

            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            if grid_test_params is None:
                log_file_name = f'{self._dataset_mode}_{self._num_episodes}ep_logger.csv'
            else:
                log_file_name = f'{self._dataset_mode}_{self._num_episodes}ep_w{grid_test_params["w_penalty"]}_logger.csv'

            log_file_dir = os.path.join(folder_path, log_file_name)

            cols = [
                'episode',
                'step',
                'image idx',
                'words',
                'word num',
                'target word',
                'encoded target word',
                'target word len',
                'target word idx',
                'norm fix x',
                'norm fix y',
                'fix x',
                'fix y',
                'letter sigma',
                'seen target word letters',
                'seen next word letters',
                'internal word',
                "word counters",
                'inferred word',
                'recognize flag',
                'completeness',
                'flag on target',
                'flag updated',
                'reward',
                'done action',
                "done",
            ]

            with open(log_file_dir, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(cols)
                writer.writerows(logger)

            print(f'\nThe log is saved in {log_file_dir}')

    def _word_activation_test(self):
        """
        This method generates the RL env testing results w/ or w/o a pre-trained RL model in a manual way.
        """
        if self._mode == _MODES['debug']:
            print('\nThe MuJoCo env and tasks baseline: ')
        elif self._mode == _MODES['test']:
            print('\nThe pre-trained RL model testing: ')

        # Start the timer
        start_time = time.time()

        # Initialize the logs dictionary
        logs_across_episodes = []

        for episode in range(1, self._num_episodes + 1):

            obs, info = self._env.reset()
            done = False
            score = 0

            # Initialize a list to store step logs for this episode
            episode_logs = []

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
            
            # Output results to check whether achieved the learning objectives. Store them to the json files.
            #   1. optimal word sampling positions and sequences
            #   2. word length's effect
            #   3. word frequency's effect
            #   4. word predictability's effect

            # Get this only after the loop ends -- when the episode ends
            individual_episode_logs = self._env.log_cumulative_version

            # Insert the episode number as the value
            individual_episode_logs['episode_idnex'] = episode - 1
            
            # Append the individual episode logs to the list of logs across episodes
            logs_across_episodes.append(individual_episode_logs)

            print(
                f'Episode:{episode}     Score:{score} \n'
                f'{"-" * 50}\n'
            )
        
        #####################################  Store the data   ######################################
        # Function to convert numpy arrays and other non-serializable types to lists or native Python types
        def convert_ndarray(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()  # Convert numpy arrays to lists
            elif isinstance(obj, np.int64) or isinstance(obj, np.int32):
                return int(obj)  # Convert NumPy integers to Python integers
            elif isinstance(obj, np.float64) or isinstance(obj, np.float32):
                return float(obj)  # Convert NumPy floats to Python floats
            else:
                raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

        # Store the data to a json file
        root_path = os.path.dirname(os.path.abspath(__file__))
        data_log_path = os.path.join(root_path, "data", "sim_results", "word_activation", self._config_rl['train']['checkpoints_folder_name'], self._config_rl['test']['loaded_model_name'], f"{self._num_episodes}ep")
        # Create the directory if it does not exist
        if not os.path.exists(data_log_path):
            os.makedirs(data_log_path)
        file_name = os.path.join(data_log_path, f'logs.json')
        # Write the logs to a JSON file
        with open(file_name, 'w') as f:
            json.dump(logs_across_episodes, f, default=convert_ndarray, indent=4)
        print(f"The logs are saved in {data_log_path}")
        ###############################################################################################

        #####################################  Analyze the data   ######################################
        with open(file_name, "r") as f:
            json_data = f.read()
        # plot_word_activation_figures.analyze_fixations(json_data=json_data, save_file_dir=data_log_path, controlled_word_length=10)
        # plot_word_activation_figures.analyze_fixations(json_data=json_data, save_file_dir=data_log_path)
        # Do the piror's effects -- freq and pred

        prior_data_effect_log_save_path = os.path.join(data_log_path, "prior_effects")
        word_binned_freq_effect_data_csv_file_path = os.path.join(prior_data_effect_log_save_path, "gaze_duration_vs_word_log_frequency_binned.csv")
        word_log_freq_effect_data_csv_file_path = os.path.join(prior_data_effect_log_save_path, "gaze_duration_vs_word_log_frequency.csv")
        word_logit_pred_effect_data_csv_file_path = os.path.join(prior_data_effect_log_save_path, "gaze_duration_vs_word_logit_predictability.csv")
        word_binned_logit_pred_effect_data_csv_file_path = os.path.join(prior_data_effect_log_save_path, "gaze_duration_vs_word_logit_predictability_binned.csv")
        if not os.path.exists(prior_data_effect_log_save_path):
            os.makedirs(prior_data_effect_log_save_path)
                # plot_word_activation_figures.analyze_priors_effect(json_data=json_data, save_file_dir=prior_data_effect_log_save_path)
        plot_word_activation_figures.analyze_priors_effect_on_gaze_duration(
            json_data=json_data, save_file_dir=prior_data_effect_log_save_path, 
            csv_log_freq_file_path=word_log_freq_effect_data_csv_file_path, csv_logit_pred_file_path=word_logit_pred_effect_data_csv_file_path,
            csv_binned_log_freq_file_path=word_binned_freq_effect_data_csv_file_path, csv_binned_logit_pred_file_path=word_binned_logit_pred_effect_data_csv_file_path
        )

        word_length_effect_data_log_save_path = os.path.join(data_log_path, "word_length_effect")
        word_length_effect_data_csv_file_path = os.path.join(word_length_effect_data_log_save_path, "gaze_duration_vs_word_length.csv")
        if not os.path.exists(word_length_effect_data_log_save_path):
            os.makedirs(word_length_effect_data_log_save_path)
        # plot_word_activation_figures.analyze_word_length_effect(json_data=json_data, save_file_dir=word_length_effect_data_log_save_path)
        plot_word_activation_figures.analyze_word_length_gaze_duration(
            json_data=json_data, save_file_dir=word_length_effect_data_log_save_path, csv_file_path=word_length_effect_data_csv_file_path
        )

        # prior_vs_word_length_data_log_save_path = os.path.join(data_log_path, "prior_vs_word_length")
        # if not os.path.exists(prior_vs_word_length_data_log_save_path):
        #     os.makedirs(prior_vs_word_length_data_log_save_path)
        # plot_word_activation_figures.analyze_prior_vs_word_length(json_data=json_data, save_file_dir=prior_vs_word_length_data_log_save_path)

        acc_data_log_save_path = os.path.join(data_log_path, "accuracy")
        if not os.path.exists(acc_data_log_save_path):
            os.makedirs(acc_data_log_save_path)
        plot_word_activation_figures.analyze_accuracy(json_data=json_data, save_file_dir=acc_data_log_save_path)

        print(f'Time elapsed for running the DEBUG/TEST: {time.time() - start_time} seconds')
        ###############################################################################################

    def _supervisory_controller_test(self):     # TODO: get a plot of regression rate vs. appraisal level weights. vs. time constraints.
        """
        This method generates the RL env testing results with or without a pre-trained RL model in a manual way.
        """
        if self._mode == _MODES['debug']:
            print('\nDebugging mode: ')
        elif self._mode == _MODES['test']:
            print('\nThe pre-trained RL model testing: ')

        # Start the timer
        start_time = time.time()

        # Initialize the logs dictionary
        logs = {}

        for episode in range(1, self._num_episodes + 1):

            obs, info = self._env.reset()
            done = False
            score = 0

            # Initialize a list to store step logs for this episode
            episode_logs = []

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

            # Collect the step log only at the end of the episode
            step_log = self._env._get_logs()

            # Process the step log to make it JSON serializable
            step_log_serializable = self._make_serializable(step_log)

            # Store the serializable step log with the episode index as the key
            logs[episode] = step_log_serializable

            # Log -- Optional: Comment this out when training
            final_step_log = self._env._get_logs()
            print(f"The final step log is: {final_step_log}")

            print(
                f'Episode:{episode}     Score:{score} \n'
                f'{"-" * 50}\n'
            )

        print(f'Time elapsed for running the DEBUG/TEST: {time.time() - start_time} seconds')

        # Store the data
        root_path = os.path.dirname(os.path.abspath(__file__))
        rl_model_name = self._config_rl['train']['checkpoints_folder_name'] + '_' + self._config_rl['test']['loaded_model_name']
        folder_path = os.path.join(root_path, "data", "sim_results", rl_model_name)

        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # Define the log file name
        log_file_name = f'rl_supervisory_controller_{self._num_episodes}ep.json'
        log_file_dir = os.path.join(folder_path, log_file_name)

        # Define the number of regression figure file name
        num_regressions_figure_file_name = f'num_regressions_vs_time_constraints_{self._num_episodes}ep.png'
        num_regressions_figure_file_dir = os.path.join(folder_path, num_regressions_figure_file_name)

        # Define the regression rate figure file name
        regression_rate_figure_file_name = f'regression_rate_sentence_level_vs_time_constraints_{self._num_episodes}ep.png'
        regression_rate_figure_file_dir = os.path.join(folder_path, regression_rate_figure_file_name)

        # Define the revisit percentage word level using saccades figure file name
        revisit_percentage_word_level_using_saccades_figure_file_name = f'revisit_percentage_word_level_using_saccades_vs_time_constraints_{self._num_episodes}ep.png'
        revisit_percentage_word_level_using_saccades_figure_file_dir = os.path.join(folder_path, revisit_percentage_word_level_using_saccades_figure_file_name)

        # Define the revisit percentage word level using reading progress figure file name
        revisit_percentage_word_level_using_reading_progress_figure_file_name = f'revisit_percentage_word_level_using_reading_progress_vs_time_constraints_{self._num_episodes}ep.png'
        revisit_percentage_word_level_using_reading_progress_figure_file_dir = os.path.join(folder_path, revisit_percentage_word_level_using_reading_progress_figure_file_name)

        # Save the logs to a JSON file
        self._save_logs_to_file(logs=logs, filepath=log_file_dir)

        # Plot the number of regressions vs. time constraints
        self._plot_revisits_vs_time_constraints(
            log_file_path=log_file_dir, num_episodes=self._num_episodes,
            num_regressions_figure_file_path=num_regressions_figure_file_dir,
            regression_rate_figure_file_path=regression_rate_figure_file_dir,
            revisit_percentage_word_level_using_saccades_figure_file_path=revisit_percentage_word_level_using_saccades_figure_file_dir,
            revisit_percentage_word_level_using_reading_progress_figure_file_path=revisit_percentage_word_level_using_reading_progress_figure_file_dir,
        )

        # Define the appraisal weight vs. revisit percentage figure file name
        appraisal_vs_revisit_figure_file_name = f'appraisal_weights_vs_revisit_percentage_{self._num_episodes}ep.png'
        appraisal_vs_revisit_figure_file_dir = os.path.join(folder_path, appraisal_vs_revisit_figure_file_name)
        # Call the new plotting function
        self._plot_appraisal_weights_vs_revisit_percentage(
            log_file_path=log_file_dir,
            num_episodes=self._num_episodes,
            figure_file_path=appraisal_vs_revisit_figure_file_dir
        )

        # Define the reward weight vs. revisit percentage figure file name
        reward_vs_revisit_figure_file_name = f'reward_explore_weights_vs_revisit_percentage_{self._num_episodes}ep.png'
        reward_vs_revisit_figure_file_dir = os.path.join(folder_path, reward_vs_revisit_figure_file_name)
        # Call the new plotting function
        self._plot_reward_explore_weight_vs_revisit_percentage(
            log_file_path=log_file_dir,
            num_episodes=self._num_episodes,
            figure_file_path=reward_vs_revisit_figure_file_dir
        )

        # Define the reward weights vs. revisit percentage figure file name
        reward_weights_vs_revisit_percentage_figure_file_name = f'reward_weights_vs_revisit_percentage_{self._num_episodes}ep.png'
        reward_weights_vs_revisit_percentage_figure_file_dir = os.path.join(folder_path, reward_weights_vs_revisit_percentage_figure_file_name)
        # Call the new plotting function
        self._plot_weights_vs_revisit_percentage(
            log_file_path=log_file_dir,
            num_episodes=self._num_episodes,
            figure_file_path=reward_weights_vs_revisit_percentage_figure_file_dir
        )

    def _make_serializable(self, data):
        """
        Recursively convert data to JSON serializable format.
        """
        if isinstance(data, dict):
            return {str(k): self._make_serializable(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._make_serializable(v) for v in data]
        elif isinstance(data, tuple):
            return tuple(self._make_serializable(v) for v in data)
        elif isinstance(data, set):
            return [self._make_serializable(v) for v in data]
        elif isinstance(data, np.ndarray):
            return data.tolist()
        elif isinstance(data, OrderedDict):
            return {str(k): self._make_serializable(v) for k, v in data.items()}
        else:
            # Attempt to convert other types directly
            try:
                json.dumps(data)
                return data
            except (TypeError, OverflowError):
                return str(data)

    @staticmethod
    def _save_logs_to_file(logs, filepath):
        """
        Saves the logs to a JSON file at the specified filepath.
        """
        with open(filepath, 'w') as f:
            json.dump(logs, f, indent=4)

        # Print the time elapsed
        print(f'Logs have been saved to {filepath}')

    @staticmethod
    def _plot_revisits_vs_time_constraints(
            log_file_path, num_episodes, num_regressions_figure_file_path, regression_rate_figure_file_path,
            revisit_percentage_word_level_using_saccades_figure_file_path,
            revisit_percentage_word_level_using_reading_progress_figure_file_path,
    ):
        """
        Reads the logs from the JSON file and plots the number of regressions vs. time constraints.
        TODO actually this is revisit, not pure regression, may change it later
        """

        # Load the logs from the JSON file
        with open(log_file_path, 'r') as f:
            logs = json.load(f)

        # Initialize data structures
        condition_30 = constants.TIME_CONSTRAINT_LEVELS['30S']
        condition_60 = constants.TIME_CONSTRAINT_LEVELS['60S']
        condition_90 = constants.TIME_CONSTRAINT_LEVELS['90S']
        time_constraints = [condition_30, condition_60, condition_90]
        regressions_by_time_constraint = {condition_30: [], condition_60: [], condition_90: []}
        regression_rate_sentence_level_by_time_constraint = {condition_30: [], condition_60: [], condition_90: []}
        revisit_percentage_word_level_using_saccades_by_time_constraint = {condition_30: [], condition_60: [], condition_90: []}
        revisit_percentage_word_level_using_reading_progress_by_time_constraint = {condition_30: [], condition_60: [], condition_90: []}

        # Iterate over episodes
        for episode, episode_log in logs.items():
            # Each episode_log is a dictionary of the final step log
            # Get the total time in seconds (time constraint) and number of regressions
            total_time_in_seconds = episode_log.get('total_time_in_seconds')
            num_regressions = episode_log.get('num_regression')
            regression_rates_sentence_level = episode_log.get('regression_rate_sentence_level')
            revisit_percentage_word_level_using_saccades = episode_log.get('revisit_percentage_word_level_using_saccades')
            revisit_percentage_word_level_using_reading_progress = episode_log.get('revisit_percentage_word_level_using_reading_progress')

            # Only consider time constraints of 30, 60, 90
            if total_time_in_seconds in time_constraints:
                regressions_by_time_constraint[total_time_in_seconds].append(num_regressions)
                regression_rate_sentence_level_by_time_constraint[total_time_in_seconds].append(regression_rates_sentence_level)
                revisit_percentage_word_level_using_saccades_by_time_constraint[total_time_in_seconds].append(revisit_percentage_word_level_using_saccades)
                revisit_percentage_word_level_using_reading_progress_by_time_constraint[total_time_in_seconds].append(revisit_percentage_word_level_using_reading_progress)
            else:
                # If total_time_in_seconds is not one of the expected values, you may want to handle it
                pass

        # Prepare data for plotting the revisit percentage at word level using saccades
        revisit_percentage_word_level_using_saccades_means = []
        revisit_percentage_word_level_using_saccades_stds = []
        for tc in time_constraints:
            # Corrected this line to use the appropriate data
            revisit_percentage_word_level_using_saccades = revisit_percentage_word_level_using_saccades_by_time_constraint[tc]
            if revisit_percentage_word_level_using_saccades:
                mean = np.mean(revisit_percentage_word_level_using_saccades)
                std = np.std(revisit_percentage_word_level_using_saccades)
            else:
                mean = 0
                std = 0
            revisit_percentage_word_level_using_saccades_means.append(mean)
            revisit_percentage_word_level_using_saccades_stds.append(std)

        plt.figure(figsize=(8, 6))
        x_pos = np.arange(len(time_constraints))
        bars = plt.bar(
            x_pos,
            revisit_percentage_word_level_using_saccades_means,
            yerr=revisit_percentage_word_level_using_saccades_stds,
            align='center',
            alpha=0.7,
            capsize=10
        )
        plt.xticks(x_pos, [str(tc) for tc in time_constraints])
        plt.xlabel('Time Constraint (seconds)')
        plt.ylabel('Revisit Percentage At Word Level Using Saccades')
        plt.title(f'Revisit Percentage At Word Level Using Saccades vs.\nTime Constraints ({num_episodes} Episodes)')
        plt.grid(True)

        # Add annotations of mean (std) on the bars
        for i, bar in enumerate(bars):
            yval = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2.0,
                yval + 0.01 * yval,
                f'{revisit_percentage_word_level_using_saccades_means[i]:.2f}\n({revisit_percentage_word_level_using_saccades_stds[i]:.2f})',
                ha='center',
                va='bottom',
                fontsize=10
            )

        plt.savefig(revisit_percentage_word_level_using_saccades_figure_file_path)
        print(f'\nThe revisit percentage word level using saccades figure is saved in {revisit_percentage_word_level_using_saccades_figure_file_path}')

        # Prepare data for plotting the revisit percentage at word level using reading progress
        revisit_percentage_word_level_using_reading_progress_means = []
        revisit_percentage_word_level_using_reading_progress_stds = []
        for tc in time_constraints:
            # Corrected this line to use the appropriate data
            revisit_percentage_word_level_using_reading_progress = revisit_percentage_word_level_using_reading_progress_by_time_constraint[tc]
            if revisit_percentage_word_level_using_reading_progress:
                mean = np.mean(revisit_percentage_word_level_using_reading_progress)
                std = np.std(revisit_percentage_word_level_using_reading_progress)
            else:
                mean = 0
                std = 0
            revisit_percentage_word_level_using_reading_progress_means.append(mean)
            revisit_percentage_word_level_using_reading_progress_stds.append(std)

        plt.figure(figsize=(8, 6))
        x_pos = np.arange(len(time_constraints))
        bars = plt.bar(
            x_pos,
            revisit_percentage_word_level_using_reading_progress_means,
            yerr=revisit_percentage_word_level_using_reading_progress_stds,
            align='center',
            alpha=0.7,
            capsize=10
        )
        plt.xticks(x_pos, [str(tc) for tc in time_constraints])
        plt.xlabel('Time Constraint (seconds)')
        plt.ylabel('Revisit Percentage At Word Level Using Reading Progress')
        plt.title(f'Revisit Percentage At Word Level Using Reading Progress\nvs. Time Constraints ({num_episodes} Episodes)')
        plt.grid(True)

        # Add annotations of mean (std) on the bars
        for i, bar in enumerate(bars):
            yval = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2.0,
                yval + 0.01 * yval,
                f'{revisit_percentage_word_level_using_reading_progress_means[i]:.2f}\n({revisit_percentage_word_level_using_reading_progress_stds[i]:.2f})',
                ha='center',
                va='bottom',
                fontsize=10
            )

        plt.savefig(revisit_percentage_word_level_using_reading_progress_figure_file_path)
        print(f'\nThe revisit percentage word level using reading progress figure is saved in {revisit_percentage_word_level_using_reading_progress_figure_file_path}')
    
    @staticmethod
    def _plot_appraisal_weights_vs_revisit_percentage(
            log_file_path, num_episodes, figure_file_path
    ):
        """
        Plots the relationship between appraisal level weights, time constraints, and revisit percentage.
        """
        # Load the logs from the JSON file
        with open(log_file_path, 'r') as f:
            logs = json.load(f)

        # Initialize data structures
        time_constraints = [30, 60, 90]  # Or use your constants
        data_by_time_constraint = {tc: {'appraisal_weights': [], 'revisit_percentages': []} for tc in time_constraints}

        # Iterate over episodes
        for episode_log in logs.values():
            total_time_in_seconds = episode_log.get('total_time_in_seconds')
            avg_appraisal_weight = episode_log.get('average_individual_sentence_appraisal_level_weight')
            revisit_percentage = episode_log.get('revisit_percentage_word_level_using_saccades')
            
            if total_time_in_seconds in time_constraints:
                data_by_time_constraint[total_time_in_seconds]['appraisal_weights'].append(avg_appraisal_weight)
                data_by_time_constraint[total_time_in_seconds]['revisit_percentages'].append(revisit_percentage)

        # Plotting
        plt.figure(figsize=(12, 8))
        for tc in time_constraints:
            appraisal_weights = data_by_time_constraint[tc]['appraisal_weights']
            revisit_percentages = data_by_time_constraint[tc]['revisit_percentages']
            
            plt.scatter(appraisal_weights, revisit_percentages, label=f'Time Constraint: {tc}s')
            
            # Optional: Fit a regression line
            if appraisal_weights and revisit_percentages:
                z = np.polyfit(appraisal_weights, revisit_percentages, 1)
                p = np.poly1d(z)
                plt.plot(appraisal_weights, p(appraisal_weights), linestyle='--')

        plt.xlabel('Average Appraisal Level Weight')
        plt.ylabel('Revisit Percentage at Word Level Using Saccades')
        plt.title(f'Revisit Percentage vs. Appraisal Level Weight\nAcross Time Constraints ({num_episodes} Episodes)')
        plt.legend()
        plt.grid(True)
        plt.savefig(figure_file_path)
        print(f'\nThe appraisal level weights vs. revisit percentage figure is saved in {figure_file_path}')
        # plt.show()

    @staticmethod
    def _plot_reward_explore_weight_vs_revisit_percentage(
            log_file_path, num_episodes, figure_file_path
    ):
        """
        Plots the relationship between appraisal level weights, time constraints, and revisit percentage.
        """
        # Load the logs from the JSON file
        with open(log_file_path, 'r') as f:
            logs = json.load(f)

        # Initialize data structures
        time_constraints = [30, 60, 90]  # Or use your constants
        data_by_time_constraint = {tc: {'explore_weight': [], 'revisit_percentages': []} for tc in time_constraints}

        # Iterate over episodes
        for episode_log in logs.values():
            total_time_in_seconds = episode_log.get('total_time_in_seconds')
            reward_weight = episode_log.get('explore_weight')
            revisit_percentage = episode_log.get('revisit_percentage_word_level_using_saccades')
            
            if total_time_in_seconds in time_constraints:
                data_by_time_constraint[total_time_in_seconds]['explore_weight'].append(reward_weight)
                data_by_time_constraint[total_time_in_seconds]['revisit_percentages'].append(revisit_percentage)

        # Plotting
        plt.figure(figsize=(12, 8))
        for tc in time_constraints:
            reward_weights = data_by_time_constraint[tc]['explore_weight']
            revisit_percentages = data_by_time_constraint[tc]['revisit_percentages']
            
            plt.scatter(reward_weights, revisit_percentages, label=f'Time Constraint: {tc}s')
            
            # Optional: Fit a regression line
            if reward_weights and revisit_percentages:
                z = np.polyfit(reward_weights, revisit_percentages, 1)
                p = np.poly1d(z)
                plt.plot(reward_weights, p(reward_weights), linestyle='--')

        plt.xlabel('Reward Explore Weight')
        plt.ylabel('Revisit Percentage at Word Level Using Saccades')
        plt.title(f'Revisit Percentage vs. Reward Weight\nAcross Time Constraints ({num_episodes} Episodes)')
        plt.legend()
        plt.grid(True)
        plt.savefig(figure_file_path)
        print(f'\nThe reward EXPLORE weights vs. revisit percentage figure is saved in {figure_file_path}')
        # plt.show()
    
    @staticmethod
    def _plot_weights_vs_revisit_percentage(
        log_file_path, num_episodes, figure_file_path
    ):
        # Load the logs from the JSON file
        with open(log_file_path, 'r') as f:
            logs = json.load(f)

        # Initialize data structures
        time_constraints = [30, 60, 90]  # Or use your constants
        data_list = []

        # Iterate over episodes
        for episode_log in logs.values():
            total_time_in_seconds = episode_log.get('total_time_in_seconds')
            failure_penalty_weight = episode_log.get('failure_penalty_weight')  # Replace with actual key
            exploit_weight = episode_log.get('exploit_weight')  # Replace with actual key
            explore_weight = episode_log.get('explore_weight')  # Replace with actual key
            revisit_percentage = episode_log.get('revisit_percentage_word_level_using_saccades')

            # Ensure all values are present
            if None not in (failure_penalty_weight, exploit_weight, explore_weight, revisit_percentage):
                data_list.append({
                    'Time Constraint': total_time_in_seconds,
                    'Failure Penalty Weight': failure_penalty_weight,
                    'Exploit Weight': exploit_weight,
                    'Explore Weight': explore_weight,
                    'Revisit Percentage': revisit_percentage
                })

        # Convert to DataFrame
        df = pd.DataFrame(data_list)

        # Plot
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Customize which weights to plot
        x = df['Failure Penalty Weight']
        y = df['Exploit Weight']
        z = df['Revisit Percentage']
        c = df['Explore Weight']  # Could also use as color coding

        scatter = ax.scatter(x, y, z, c=c, cmap='viridis')

        ax.set_xlabel('Failure Penalty Weight')
        ax.set_ylabel('Exploit Weight')
        ax.set_zlabel('Revisit Percentage')

        # Add color bar if using color coding
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Explore Weight')

        plt.title(f'Revisit Percentage vs. Weights\n({num_episodes} Episodes)')
        plt.savefig(figure_file_path)
        plt.show()

    def _sentence_level_controller_test(self):
        """
        This method generates the RL env testing results with or without a pre-trained RL model in a manual way.
        """
        if self._mode == _MODES['debug']:
            print('\nThe env and tasks baseline: ')
        elif self._mode == _MODES['test']:
            print('\nThe pre-trained RL model testing: ')

        # Start the timer
        start_time = time.time()

        # Initialize the logs dictionary
        logs = {}

        for episode in range(1, self._num_episodes + 1):

            obs, info = self._env.reset()
            done = False
            score = 0

            # Initialize a list to store step logs for this episode
            episode_logs = []

            while not done:
                if self._mode == _MODES['debug']:
                    action = self._env.action_space.sample()
                elif self._mode == _MODES['test']:
                    # action, _states = self._model.predict(obs, deterministic=True)
                    action, _states = self._model.predict(obs, deterministic=False)
                else:
                    raise ValueError(f'Invalid mode {self._mode}.')

                obs, reward, done, truncated, info = self._env.step(action)
                score += reward

            # Collect the step log only at the end of the episode
            step_log = self._env._get_logs()

            # Process the step log to make it JSON serializable
            step_log_serializable = self._make_serializable(step_log)

            # Store the serializable step log with the episode index as the key
            logs[episode] = step_log_serializable

            # Log
            final_step_log = self._env._get_logs()
            print(f"The final step log is: {final_step_log}")
            print(f"The number of words in the sentence: {self._env.num_words_in_sentence}, "
                  f"the allocated time constraint: {self._env._time_constraint_level_key}, "
                  f"the episode length: {self._env.granted_time_constraints_in_steps}.")
            print(f"The number of words read: {self._env.num_words_read_in_sentence}, "
                  f"the number of words skipped: {self._env.num_words_skipped_in_sentence}, "
                  f"the number of word-level saccades: {self._env.num_saccades_on_word_level}."
                  f"The word skipping rate is: {self._env.num_words_skipped_in_sentence /self._env.num_words_read_in_sentence}.")
            print(f"The predictability levels of the words: {self._env._predictability_states}.")
            print(f"The time constraint weight: {self._env._time_constraint_weight}.")
            
            print(
                f'Episode:{episode}     Score:{score} \n'
                f'{"-" * 50}\n'
            )

        print(f'Time elapsed for running the DEBUG/TEST: {time.time() - start_time} seconds')

        # Store the data
        root_path = os.path.dirname(os.path.abspath(__file__))
        rl_model_name = self._config_rl['train']['checkpoints_folder_name'] + '_' + self._config_rl['test']['loaded_model_name']
        folder_path = os.path.join(root_path, "data", "sim_results", rl_model_name)

        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # Define the log file name
        log_file_name = f'rl_word_level_controller_{self._num_episodes}ep.json'
        log_file_dir = os.path.join(folder_path, log_file_name)

        # Define the word skipping rate figure file name
        word_skipping_rate_figure_file_name = f'word_skipping_rate_vs_time_constraints_{self._num_episodes}ep.png'
        word_skipping_rate_figure_file_dir = os.path.join(folder_path, word_skipping_rate_figure_file_name)

        # Define the word skip by appraisal figure file name
        word_skip_by_appraisals_figure_file_name = f'word_skips_vs_appraisals_{self._num_episodes}ep.png'
        word_skip_by_appraisals_figure_file_dir = os.path.join(folder_path, word_skip_by_appraisals_figure_file_name)

        # Define the word skipping rate vs. time awareness weight figure file name
        word_skipping_vs_time_awareness_figure_file_name = f'word_skipping_rate_vs_time_awareness_weight_{self._num_episodes}ep.png'
        word_skipping_vs_time_awareness_figure_file_dir = os.path.join(folder_path, word_skipping_vs_time_awareness_figure_file_name)
        
        # Define the word skipping rate vs. time awareness weight figure file name
        word_skipping_vs_time_constraint_figure_file_name = f'word_skipping_rate_vs_time_constraint_weight_{self._num_episodes}ep.png'
        word_skipping_vs_time_constraint_figure_file_dir = os.path.join(folder_path, word_skipping_vs_time_constraint_figure_file_name)

        # Save the logs to a JSON file
        self._save_logs_to_file(logs=logs, filepath=log_file_dir)

        # Call the updated plotting function
        self._plot_word_skipping_rates(
            log_file_path=log_file_dir, num_episodes=self._num_episodes,
            word_skipping_rate_by_time_constraint_levels_figure_file_path=word_skipping_rate_figure_file_dir,
            word_skip_by_appraisals_figure_file_dir=word_skip_by_appraisals_figure_file_dir,
            word_skipping_vs_time_awareness_figure_file_path=word_skipping_vs_time_awareness_figure_file_dir,
            word_skipping_vs_time_constraint_figure_file_path=word_skipping_vs_time_constraint_figure_file_dir,
        )

    @staticmethod
    def _plot_word_skipping_rates(log_file_path, num_episodes, word_skipping_rate_by_time_constraint_levels_figure_file_path, 
                                  word_skip_by_appraisals_figure_file_dir, word_skipping_vs_time_awareness_figure_file_path, 
                                  word_skipping_vs_time_constraint_figure_file_path):

        """
        Reads the logs from the JSON file and plots:
        - Word skipping rates vs. time constraints
        - Word skipping vs. appraisal levels
        - Word skipping rate vs. time awareness weight
        - Predictabilities vs. time constraint weights (New Plot)
        """

        # Load the logs from the JSON file
        with open(log_file_path, 'r') as f:
            logs = json.load(f)

        # Initialize data structures for time constraints
        condition_30 = '30S'
        condition_60 = '60S'
        condition_90 = '90S'
        time_constraint_levels = [condition_30, condition_60, condition_90]
        word_skipping_rates_by_time_constraint = {condition_30: [], condition_60: [], condition_90: []}

        # Initialize data for appraisal scatter plot
        appraisals = []
        y_positions = []  # This will store the corresponding y position for each condition
        colors = []
        colors_map = {condition_30: 'red', condition_60: 'green', condition_90: 'blue'}
        y_map = {condition_30: 0, condition_60: 1, condition_90: 2}  # Map conditions to y-axis positions
        predictabilities = []
        time_constraint_weights = []

        # Initialize data for time awareness weight plot
        time_awareness_weights = []
        skipping_rates = []

        # Iterate over episodes in logs
        for episode, episode_log in logs.items():
            # Get time constraint and word skipping rate
            time_constraint_level = episode_log.get('time_constraint_level')
            word_skipping_rate = episode_log.get('word_skipping_percentage')
            appraisal_states = episode_log.get('appraisal_states')
            time_constraint_weight = episode_log.get('time_constraint_weight')
            
            # Only consider valid time constraints of 30S, 60S, 90S
            if time_constraint_level in time_constraint_levels:
                word_skipping_rates_by_time_constraint[time_constraint_level].append(word_skipping_rate)

                # For appraisal-based plot
                for word_idx, (appraisal_value, skip_decision) in appraisal_states.items():
                    if skip_decision == 1:  # Only considering skipped words
                        appraisals.append(appraisal_value)
                        y_positions.append(y_map[time_constraint_level])  # Map the condition to the y position
                        colors.append(colors_map[time_constraint_level])
            else:
                print(f"Time constraint level {time_constraint_level} is not one of the expected values.")
            
            # Get appraisal states
            if time_constraint_weight is not None and word_skipping_rate is not None:
                # Convert word_skipping_rate to a value between 0 and 1
                skipping_rate = word_skipping_rate / 100.0
                time_constraint_weights.append(time_constraint_weight)
                skipping_rates.append(skipping_rate)
            else:
                print(f"Missing data in episode {episode} for time awareness CONSTRAINT plot.")

        # -------------------------------------------------------------------------------------------
        # Updated Plot: Appraisal Levels by Binned Time Constraint Weights

        # Collect appraisal values and corresponding time constraint weights for skipped words
        skipped_appraisals = []
        skipped_time_constraint_weights = []

        for episode, episode_log in logs.items():
            time_constraint_weight = episode_log.get('time_constraint_weight')
            appraisal_states = episode_log.get('appraisal_states')
            if appraisal_states and time_constraint_weight is not None:
                for word_idx, (appraisal_value, skip_decision) in appraisal_states.items():
                    if skip_decision == 1:  # Only considering skipped words
                        skipped_appraisals.append(appraisal_value)
                        skipped_time_constraint_weights.append(time_constraint_weight)
            else:
                print(f"Missing data in episode {episode} for appraisal levels by time constraint weights.")

        # Convert to numpy arrays
        skipped_appraisals = np.array(skipped_appraisals)
        skipped_time_constraint_weights = np.array(skipped_time_constraint_weights)

        # Define bins for Time Constraint Weight
        bins = np.linspace(0, 1, 11)  # 10 bins from 0 to 1
        bin_labels = [f'{bins[i]:.1f}-{bins[i+1]:.1f}' for i in range(len(bins)-1)]

        # Bin the Time Constraint Weight data
        binned_indices = np.digitize(skipped_time_constraint_weights, bins) - 1  # Adjust indices to start from 0

        # Group appraisal values by binned time constraint weights
        appraisals_by_tcw_bin = [[] for _ in range(len(bins)-1)]

        for idx, bin_idx in enumerate(binned_indices):
            if 0 <= bin_idx < len(appraisals_by_tcw_bin):
                appraisals_by_tcw_bin[bin_idx].append(skipped_appraisals[idx])
            else:
                print(f"Time constraint weight {skipped_time_constraint_weights[idx]} is out of bin range.")

        # Remove bins with no data
        appraisals_by_tcw_bin_filtered = []
        bin_labels_filtered = []
        for i, appraisals_in_bin in enumerate(appraisals_by_tcw_bin):
            if appraisals_in_bin:
                appraisals_by_tcw_bin_filtered.append(appraisals_in_bin)
                bin_labels_filtered.append(bin_labels[i])

        # Create the horizontal box plot
        plt.figure(figsize=(8, 6))
        box = plt.boxplot(
            appraisals_by_tcw_bin_filtered,
            vert=False,
            patch_artist=True,
            labels=bin_labels_filtered
        )

        # Customize the plot
        plt.xlabel('Appraisal Value')
        plt.ylabel('Time Constraint Weight Bins')
        plt.title(f'Appraisal Levels by Time Constraint Weights ({num_episodes} Episodes)')
        plt.grid(True)

        # Save the figure
        plt.savefig(word_skip_by_appraisals_figure_file_dir, bbox_inches='tight')
        plt.close()
        print(f"The appraisal levels by time constraint weights figure is saved in {word_skip_by_appraisals_figure_file_dir}")


        # -------------------------------------------------------------------------------------------
        # New Plot: Word Skipping Rate vs. Time Constraint Weight with Improved Visualization

        # Convert data to numpy arrays
        time_constraint_weights = np.array(time_constraint_weights)
        skipping_rates = np.array(skipping_rates)

        # Create a DataFrame
        data_df = pd.DataFrame({
            'Time Constraint Weight': time_constraint_weights,
            'Word Skipping Rate': skipping_rates
        })


        # Define bins for Time Constraint Weight
        bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        labels = ['0.0-0.1', '0.1-0.2', '0.2-0.3', '0.3-0.4', '0.4-0.5', '0.5-0.6', '0.6-0.7', '0.7-0.8', '0.8-0.9', '0.9-1.0']

        # Bin the Time Constraint Weight data
        data_df['TCW Bin'] = pd.cut(data_df['Time Constraint Weight'], bins=bins, labels=labels, include_lowest=True)

        # Calculate mean and standard deviation of Word Skipping Rate for each bin
        grouped = data_df.groupby('TCW Bin')['Word Skipping Rate'].agg(['mean', 'std']).reset_index()

        # Plot the bar chart
        plt.figure(figsize=(8, 6))
        bars = plt.bar(
            grouped['TCW Bin'],
            grouped['mean'],
            yerr=grouped['std'],
            align='center',
            alpha=0.7,
            capsize=10
        )
        plt.xlabel('Time Constraint Weight Bins')
        plt.ylabel('Word Skipping Rate')
        plt.title(f'Word Skipping Rate vs. Time Constraint Weight ({num_episodes} Episodes)')
        plt.grid(True)

        # Add annotations of mean (std) on the bars
        for i, bar in enumerate(bars):
            yval = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2.0,
                yval + 0.01,
                f'{grouped["mean"].iloc[i]:.2f}\n({grouped["std"].iloc[i]:.2f})',
                ha='center',
                va='bottom',
                fontsize=10
            )

        # Save the new figure
        plt.savefig(word_skipping_vs_time_constraint_figure_file_path, bbox_inches='tight')
        plt.close()
        print(f"The word skipping rate vs. time constraint weight figure is saved in {word_skipping_vs_time_constraint_figure_file_path}")


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
            if isinstance(self._env, GeneralOculomotorControllerEnv) or isinstance(self._env, OculomotorControllerRLEnv):
                self._oculomotor_controller_test()
            elif isinstance(self._env, WordActivationRLEnv):
                self._word_activation_test()
            elif isinstance(self._env, SupervisoryControllerEnv):
                self._supervisory_controller_test()
            elif isinstance(self._env, SentenceLevelControllerEnv):
                self._sentence_level_controller_test()
            else:
                raise ValueError(f'Invalid environment {self._env}.')
        elif self._mode == _MODES['grid_test']:
            self._grid_test()
        else:
            raise ValueError(f'Invalid mode {self._mode}.')

    def __del__(self):
        # Close the environment.
        self._env.close()

        # Visualize the destructor.
        print(
            '\n\n***************************** RL pipeline ends. The MuJoCo environment of the pipeline has been destructed *************************************'
        )