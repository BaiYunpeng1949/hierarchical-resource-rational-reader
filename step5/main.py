import os
import sys

# Make the repo root importable so `import step5.*` resolves regardless of
# the directory you launch from (fixes ModuleNotFoundError: No module named 'step5').
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from STB3RL import RL
import os


def main():
    # Run the RL pipeline with the given configurations.
    config_file_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.yaml')
    rl = RL(config_file=config_file_dir)
    rl.run()


if __name__ == "__main__":
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    main()