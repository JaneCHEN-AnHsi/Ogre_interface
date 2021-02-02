# -*- coding: utf-8 -*-

from setuptools import find_packages
from distutils.core import setup

setup(
    name='ogre_interfaces',
    version='0.0',
    install_requires=['numpy', 'matplotlib', 'pymatgen==2019.5.8', "seaborn",
                      "ase==3.17.0", "scipy", "argparse", "itertools"],
    include_package_data=True,
)
