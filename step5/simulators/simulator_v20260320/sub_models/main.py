import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")))
from rl import RL


def main():
    # Run the RL pipeline with the given configurations.
    config_file_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.yaml')

    rl = RL(config_file=config_file_dir)
    rl.run()


if __name__ == "__main__":
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    main()
