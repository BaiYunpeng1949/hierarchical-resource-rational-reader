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
from utils import plot_sentence_reading_figures as plot_sentence_reading_figures

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
# from modules.rl_envs.sentence_read_v0306.SentenceReadingEnvV0306 import SentenceReadingEnv
from modules.rl_envs.sentence_read_v0319.SentenceReadingEnv import SentenceReadingEnv
from modules.rl_envs.text_comprehension_v0516.TextComprehensionEnv import TextComprehensionEnv


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
        env_class = TextComprehensionEnv # SentenceReadingEnv # WordActivationRLEnv # OculomotorControllerRLEnv # GeneralOculomotorControllerEnv           # GeneralOculomotorControllerEnv, SentenceLevelControllerEnv, SupervisoryControllerEnv

        # Load the dataset (if needed)
        shared_dataset_metadata_of_stimuli = None
        shared_dataset_encoded_lexicon = None

        # Read the total dataset if training the general oculomotor controller model.
        if env_class == WordActivationRLEnv:
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
        elif env_class == SentenceReadingEnv:
            self._env = SentenceReadingEnv()
            def make_env():
                env = SentenceReadingEnv()
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
        elif env_class == TextComprehensionEnv:
            self._env = TextComprehensionEnv()
            def make_env():
                env = TextComprehensionEnv()
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
            if isinstance(self._env, WordActivationRLEnv) or isinstance(self._env, SentenceReadingEnv) or isinstance(self._env, TextComprehensionEnv):
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
    

    def _text_comprehension_test(self):
        """
        Test the text comprehension environment.
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

            while not done:
                if self._mode == _MODES['debug']:
                    action = self._env.action_space.sample()
                elif self._mode == _MODES['test']:
                    action, _states = self._model.predict(obs, deterministic=False)
                else:
                    raise ValueError(f'Invalid mode {self._mode}.')

                obs, reward, done, truncated, info = self._env.step(action)
                score += reward
            
            print(
                f'Episode:{episode}     Score:{score}\n'
                f'{"-" * 50}\n'
            )

        # Save logs if in test mode
        if self._mode == _MODES['test']:
            # Create the log directory
            root_path = os.path.dirname(os.path.abspath(__file__))
            rl_model_name = self._config_rl['train']['checkpoints_folder_name'] + '_' + self._config_rl['test']['loaded_model_name']
            log_dir = os.path.join(root_path, "data", "sim_results", "sentence_reading", rl_model_name, f"{self._num_episodes}ep")

            # Create the directory if it does not exist
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)

            # Save the logs to a json file
            with open(os.path.join(log_dir, "raw_simulated_results.json"), "w") as f:
                json.dump(logs_across_episodes, f, indent=4)


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
            elif isinstance(self._env, SentenceReadingEnv):
                self._sentence_reading_test()
            elif isinstance(self._env, TextComprehensionEnv):
                self._text_comprehension_test()
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
