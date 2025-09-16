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

# from step5.modules.rl_envs.SupervisoryControllerEnv_v0922 import SupervisoryControllerEnv
# from step5.modules.rl_envs.SupervisoryControllerEnv_v1018 import SupervisoryControllerEnv
# from step5.modules.rl_envs.SupervisoryControllerEnv_v1023 import SupervisoryControllerEnv
# from step5.modules.rl_envs.SupervisoryControllerEnv_v1030 import SupervisoryControllerEnv
# from step5.modules.rl_envs.SupervisoryControllerEnv_v1031 import SupervisoryControllerEnv
# from step5.modules.rl_envs.SupervisoryControllerEnv_v1101 import SupervisoryControllerEnv
from step5.modules.rl_envs.SupervisoryControllerEnv_v1101_2 import SupervisoryControllerEnv

# from step5.modules.rl_envs.SentenceLevelControllerEnv_v0925 import SentenceLevelControllerEnv
# from step5.modules.rl_envs.SentenceLevelControllerEnv_v1011 import SentenceLevelControllerEnv
from step5.modules.rl_envs.SentenceLevelControllerEnv_v1014 import SentenceLevelControllerEnv

# from step5.modules.rl_envs.GeneralOculomotorControllerEnv_v0725 import GeneralOculomotorControllerEnv
# from step5.modules.rl_envs.GeneralOculomotorControllerEnv_v1010 import GeneralOculomotorControllerEnv
# from step5.modules.rl_envs.GeneralOculomotorControllerEnv_v1013 import GeneralOculomotorControllerEnv
# from step5.modules.rl_envs.GeneralOculomotorControllerEnv_v1122 import GeneralOculomotorControllerEnv
# from step5.modules.rl_envs.GeneralOculomotorControllerEnv_v1126 import GeneralOculomotorControllerEnv
from modules.rl_envs.GeneralOculomotorControllerEnv_v1126 import GeneralOculomotorControllerEnv
from modules.rl_envs.OMCRLEnvV0128 import OculomotorControllerRLEnv
# from modules.rl_envs.WordActivationEnvV0205 import WordActivationRLEnv
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

            # # Serialize the dictionaries for putting on a shared-memory
            # stimuli_metadata_bytes = pickle.dumps(shared_dataset_metadata_of_stimuli)
            # encoded_lexicon_bytes = pickle.dumps(shared_dataset_encoded_lexicon)
            # # Create shared memory for metadata
            # self._metadata_shm = shared_memory.SharedMemory(create=True, size=len(stimuli_metadata_bytes))
            # self._metadata_shm.buf[:len(stimuli_metadata_bytes)] = stimuli_metadata_bytes
            # # Create shared memory for encoded lexicon
            # self._lexicon_shm = shared_memory.SharedMemory(create=True, size=len(encoded_lexicon_bytes))
            # self._lexicon_shm.buf[:len(encoded_lexicon_bytes)] = encoded_lexicon_bytes

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

    def _word_activation_grid_test(self):
        """
        Sweep kappa over [start, end] with the given step.
        For each kappa:
        - run self._num_episodes episodes
        - collect ALL episode logs into logs.json (list of episodes)
        - run analyzers to produce prior-effect CSVs (raw + binned), word-length CSV, and accuracy
        """

        # ----- grid from config (inclusive end) -----
        k_start, k_end, k_step = self._config_rl['test']['params']['kappa']
        num_kappas = int((k_end - k_start) / k_step) + 1

        # ----- output base dir (your new path) -----
        root_path = os.path.dirname(os.path.abspath(__file__))    # /home/.../step5
        base_dir = os.path.join(
            root_path, "modules", "rl_envs", "word_activation_v0218",
            "parameter_inference", "simulation_data"
        )
        os.makedirs(base_dir, exist_ok=True)

        def _fmt_kappa_folder(k):
            # e.g., kappa_2p00 (safe folder name)
            return f"kappa_{str(float(k)).replace('.', 'p')}"

        # json helper
        def convert_ndarray(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.float64, np.float32)):
                return float(obj)
            else:
                raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

        start_time = time.time()
        for k_idx in range(num_kappas):
            kappa = k_start + k_idx * k_step

            # folder for THIS kappa
            kappa_dir = os.path.join(base_dir, _fmt_kappa_folder(kappa))
            os.makedirs(kappa_dir, exist_ok=True)

            logs_across_episodes = []

            # run N episodes for THIS kappa
            for ep_idx in range(1, self._num_episodes + 1):

                obs, info = self._env.reset(params={"kappa": float(kappa)}, ep_idx=ep_idx)
                # Defensive: make sure the env created the log structure (some envs do it lazily)
                try:
                    if not hasattr(self._env, "log_cumulative_version") or \
                    "fixations" not in getattr(self._env, "log_cumulative_version", {}):
                        _ = self._env._get_logs(is_initialization=True, mode="test")
                except Exception:
                    # if your env doesn't expose _get_logs publicly, it's fine—logging should still proceed on steps
                    pass

                done = False
                score = 0.0
                while not done:
                    action, _ = self._model.predict(obs, deterministic=True)
                    obs, reward, done, truncated, step_info = self._env.step(action)
                    score += reward

                ep_log = self._env.log_cumulative_version
                ep_log["episode_idnex"] = ep_idx  # keep your original spelling
                ep_log["kappa"] = float(kappa)

                logs_across_episodes.append(ep_log)

                # console progress
                if (ep_idx % max(1, self._num_episodes // 5)) == 0 or ep_idx == self._num_episodes:
                    print(f"[grid_test] kappa={kappa:.3f}  episode {ep_idx}/{self._num_episodes}")

            # save ALL episodes for this kappa
            logs_path = os.path.join(kappa_dir, "logs.json")
            with open(logs_path, "w", encoding="utf-8") as f:
                json.dump(logs_across_episodes, f, default=convert_ndarray, indent=4)

            # analyze once per kappa (aggregates all episodes)
            with open(logs_path, "r", encoding="utf-8") as f:
                json_data = f.read()

            # prior effects (raw + binned)
            csv_log_freq_file_path          = os.path.join(kappa_dir, "gaze_duration_vs_word_log_frequency.csv")
            csv_logit_pred_file_path        = os.path.join(kappa_dir, "gaze_duration_vs_word_logit_predictability.csv")
            csv_binned_log_freq_file_path   = os.path.join(kappa_dir, "gaze_duration_vs_word_log_frequency_binned.csv")
            csv_binned_logit_pred_file_path = os.path.join(kappa_dir, "gaze_duration_vs_word_logit_predictability_binned.csv")

            plot_word_activation_figures.analyze_priors_effect_on_gaze_duration(
                json_data=json_data, save_file_dir=kappa_dir,
                csv_log_freq_file_path=csv_log_freq_file_path,
                csv_logit_pred_file_path=csv_logit_pred_file_path,
                csv_binned_log_freq_file_path=csv_binned_log_freq_file_path,
                csv_binned_logit_pred_file_path=csv_binned_logit_pred_file_path
            )

            # word-length effect
            wl_csv = os.path.join(kappa_dir, "gaze_duration_vs_word_length.csv")
            plot_word_activation_figures.analyze_word_length_gaze_duration(
                json_data=json_data, save_file_dir=kappa_dir, csv_file_path=wl_csv
            )

            # accuracy
            acc_dir = os.path.join(kappa_dir, "accuracy")
            os.makedirs(acc_dir, exist_ok=True)
            plot_word_activation_figures.analyze_accuracy(json_data=json_data, save_file_dir=acc_dir)

            print(f"[grid_test] kappa={kappa:.3f}  -> analyses saved in {kappa_dir}")

        print(f"Time elapsed for GRID TEST: {time.time() - start_time:.2f} s")

    @staticmethod
    def _save_logs_to_file(logs, filepath):
        """
        Saves the logs to a JSON file at the specified filepath.
        """
        with open(filepath, 'w') as f:
            json.dump(logs, f, indent=4)

        # Print the time elapsed
        print(f'Logs have been saved to {filepath}')


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
                # self._oculomotor_controller_test()
                pass
            elif isinstance(self._env, WordActivationRLEnv):
                self._word_activation_test()
            elif isinstance(self._env, SupervisoryControllerEnv):
                # self._supervisory_controller_test()
                pass
            elif isinstance(self._env, SentenceLevelControllerEnv):
                # self._sentence_level_controller_test()
                pass
            else:
                raise ValueError(f'Invalid environment {self._env}.')
        elif self._mode == _MODES['grid_test']:
            self._word_activation_grid_test()
        else:
            raise ValueError(f'Invalid mode {self._mode}.')

    def __del__(self):
        # Close the environment.
        self._env.close()

        # Visualize the destructor.
        print(
            '\n\n***************************** RL pipeline ends. The MuJoCo environment of the pipeline has been destructed *************************************'
        )