from setuptools import setup

setup(
   name='read-agent',
   version='0.1.0',
   author='Bai Yunpeng',
   author_email='byp19971001@gmail.com',
   packages=['ra'],
   url='https://github.com/BaiYunpeng1949/reading-model',
   license='LICENSE',
   description='Modeling and simulating reading behaviors.',
   long_description=open('README.md').read(),
   python_requires='>=3.8',
   install_requires=[
       "ale_py==0.7.4",
       "annoy==1.17.3",
       "gym==0.21.0",
       "matplotlib==3.5.1",
       "openai", "Pillow", "PyYAML",
       "stable_baselines3>=1.4.0", "torch",
       "wandb", "tensorboard",
       "numpy", "matplotlib", "scipy",
       "opencv-python",
       "ruamel.yaml",
   ],
)
