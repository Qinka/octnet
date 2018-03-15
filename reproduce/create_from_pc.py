import sys
import numpy as np
import time
import os
from glob import glob
import random
import multiprocessing
import urllib
import zipfile

sys.path.append('../py/')

import pyoctnet


def worker(file : str, vx_res, n_threads=1):
    dat = np.loadtxt(file)
    pts = dat[:,0:3]
    ft  = dat[:,4:8]
    