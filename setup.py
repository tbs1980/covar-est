from setuptools import setup
from setuptools import find_packages

setup(
    name='covariance_estimation',
    version='0.1.0',
    description='Estimate covariance in the presence of noise',
    author='Sreekumar Thaithara Balan',
    author_email='sreekumar.balan@alpha-i.co',
    packages=find_packages(exclude=['doc', 'tests*']),
    install_requires=[
        'numpy==1.13.0',
        'emcee==2.2.1',
        'corner==2.0.1',
        'scipy==0.19.0',
        'pymc3==3.0'
    ]
)
