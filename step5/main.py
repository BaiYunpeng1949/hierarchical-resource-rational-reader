import os
import sys

# Make the repo root importable
# fixes bug on mac
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from STB3RL import RL


def main():
    # Run the RL pipeline with the given configurations.
    config_file_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.yaml')
    rl = RL(config_file=config_file_dir)
    rl.run()


if __name__ == "__main__":
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    main()
