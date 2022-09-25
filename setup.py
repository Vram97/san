"""This file helps you run "Pythopn3 setup.py" with options from the command line with options given by the documentation of setup.
This file is not required to run our code for the DR"""

#Importing libraries
import os
from setuptools import setup, find_packages


def parse_requirements(file):
    required_packages = []
    with open(os.path.join(os.path.dirname(__file__), file)) as req_file:
        for line in req_file:
            required_packages.append(line.strip())
    return required_packages


setup(name='san',
      version='0.32',
      description="Feature ranking with self-attention networks",
      url='http://github.com/skblaz/san',
      author='Blaž Škrlj and Matej Petković',
      author_email='blaz.skrlj@ijs.si',
      license='MIT',
      packages=find_packages(),
      zip_safe=False,
      include_package_data=True,
      install_requires=parse_requirements('requirements.txt'))
