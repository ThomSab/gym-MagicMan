from setuptools import setup,find_packages
import os

git_url = r'https://github.com/ThomSab/gym-MagicMan'

setup(name='gym_MagicMan',
        version='0.0.3',
        author='Jasper Vogel',
        url=git_url,
        install_requires=['gymnasium','numpy','torch'],
      )


#https://github.com/ThomSab/gym-MagicMan/tree/main/envs