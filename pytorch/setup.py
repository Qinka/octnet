import os 
import sys

from setuptools import setup, find_packages

import build

this_file = os.path.dirname(__file__)

setup(
    name="octnet",
    version="0.1",
    description = "OctNet on pytorch",
    author =" Johann Lee, OctNet authors",
    install_requores=['cffi>=1.0.0'],
    setup_requires=['cffi>=1.0.0'],
    packages=find_packages(exclude=['build']),
    cffi_modules=[
        os.path.join(this_file, "build.py:ffi")
    ],
)