import yaml
import os
import warnings

from stable_baselines3 import PPO

from step5.modules.rl_envs.OculomotorControllerEnv import OculomotorControllerEnv
from step5.modules.rl_envs.GeneralOculomotorControllerEnv import GeneralOculomotorControllerEnv
import step5.utils.constants as const


class OculomotorController:

    def __init__(
            self,
            config: str = r'D:\Users\91584\PycharmProjects\reading-model\step5\config.yaml',
    ):
        """
        Deploy the pre-trained oculomotor controller environment and model.
            It only simulates the lowest word identification control -- one word one instance.
        :param config: the yaml configuration file path.
        """
        # Read the configuration file
        with open(config, 'r') as stream:
            self._config = yaml.load(stream, Loader=yaml.FullLoader)

        assert self._config['rl']['mode'] == const.SIMULATE, "The RL mode should be set to simulate in the config.yaml."

        self._sim_rl_config = self._config['simulate']['computationally_rational_model']

        print(f"{const.LV_ONE_DASHES}Oculomotor Controller -- Deploying the environment {self._sim_rl_config['env_name']} in the simulation mode.")

        # Load the environment within the pre-trained model
        self._root_dir = os.path.dirname(config)    # The config.yaml is in the root directory
        self._oculomotor_controller_model_path = os.path.join(
            self._root_dir, 'training', 'saved_models', self._sim_rl_config['checkpoints_folder_name'], self._sim_rl_config['loaded_model_name']
        )

        self._env = GeneralOculomotorControllerEnv()    # OculomotorControllerEnv()

        # Custom object to handle deserialization issues
        custom_objects = {}

        try:
            self._model = PPO.load(self._oculomotor_controller_model_path, self._env, custom_objects=custom_objects)
        except (RuntimeError, TypeError) as e:
            warnings.warn(f"Could not deserialize object: {e}")
            raise e

        print(f"{const.LV_TWO_DASHES}Initiate the Oculomotor Controller. Loading the pre-trained model from {self._oculomotor_controller_model_path}")

    def step(
            self,
            inputs: dict = None,
    ) -> list:
        """
        Deploy the pre-trained oculomotor controller environment and model for ONE time.
        So looping over the number of episodes is outside, in the simulator, which is calling this method.
        :return: all necessary raw data
        """
        assert inputs is not None, "The inputs should be provided by the supervisory controller."

        obs, info = self._env.reset(inputs=inputs)
        done = False
        score = 0
        logger = []

        while not done:
            action, _states = self._model.predict(obs, deterministic=True)
            obs, rewards, done, truncated, info = self._env.step(action)
            score += rewards

            # Save the raw data from the logger
            logger.append(self._env.logger)

        return logger
