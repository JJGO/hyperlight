import os

import setuptools
from setuptools import setup

# Get requirements from file
with open('requirements.txt') as f:
    install_requires = f.read().splitlines()


# Use repo README for PyPi description
with open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()


classifiers=[
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.9',
    'Programming Language :: Python :: 3.10',
    'License :: OSI Approved :: Apache Software License 2.0 (Apache-2.0)',
],

extra_deps = {}


setup(
    name='hyperlight',
    version='0.1.0',
    author='JJGO',
    author_email='josejg@mit.edu',
    description=
    'Hyperlight is a Pytorch hypernetwork framework with a streamlined API',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/JJGO/hyperlight/',
    packages=setuptools.find_packages(exclude=['tests*']),
    classifiers=classifiers,
    install_requires=install_requires,
    extras_require=extra_deps,
    python_requires='>=3.7',
)