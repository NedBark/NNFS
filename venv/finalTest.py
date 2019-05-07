import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from tensorflow import keras
import pandas as pd
#from PIL import Image
from scipy import ndimage
from functions import *
from TwoLayersFunctions import *
#from lr_utils import load_dataset
import _pickle as pickle

f = open('params.txt', 'r')
g = open('costs.txt', 'r')

params = pickle.loads(f)
costs = pickle.loads(g)