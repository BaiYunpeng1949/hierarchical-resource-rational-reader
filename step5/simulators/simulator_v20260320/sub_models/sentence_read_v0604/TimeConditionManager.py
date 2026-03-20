import os
import json
import random
from . import Constants
# import Constants


class TimeConditionManager():
    """
    Time Condition Manager that handles loading and sampling time conditions from the study conditions.
    """

    def __init__(self):
        self._time_conditions = Constants.TIME_CONDITIONS

    def reset(self, inputs: dict=None):

        if inputs is None:
            # Randomly select a time condition, return the key and value
            time_condition = random.choice(list(self._time_conditions.keys()))
            return time_condition, self._time_conditions[time_condition]
        else:
            time_condition = inputs["time_condition"]
            return time_condition, self._time_conditions[time_condition]


if __name__ == "__main__":
    time_condition_manager = TimeConditionManager()
    time_condition_entry = time_condition_manager.reset()
    print(time_condition_entry)