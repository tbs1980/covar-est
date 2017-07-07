from os.path import realpath, dirname, join
from setuptools import setup
from setuptools import find_packages

PROJECT_ROOT = dirname(realpath(__file__))
REQUIREMENTS_FILE = join(PROJECT_ROOT, 'requirements.txt')

with open(REQUIREMENTS_FILE) as f:
    install_reqs = f.read().splitlines()

setup(
    name='covariance_estimation',
    version='0.1.0',
    description='Estimate covariance in the presence of noise',
    author='Sreekumar Thaithara Balan',
    author_email='sreekumar.balan@alpha-i.co',
    packages=find_packages(exclude=['doc', 'tests*']),
    install_requires=install_reqs
)
