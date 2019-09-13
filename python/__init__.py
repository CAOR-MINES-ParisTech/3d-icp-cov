import numpy as np
import os
from scipy.special import bernoulli
from scipy import linalg
from scipy.linalg import block_diag
import alphashape
import subprocess
import pickle
import matplotlib
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['axes.titlesize'] = 'x-large'
import matplotlib.pyplot as plt


__author__ = "Martin Brossard"
__email__ = "martin.brossard@mines-paristech.fr"