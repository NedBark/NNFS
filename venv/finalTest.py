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

dataset_path = keras.utils.get_file("nursery.data",
                                    "https://archive.ics.uci.edu/ml/machine-learning-databases/nursery/nursery.data",
                                    cache_subdir="D:\\NNStuff\\NNSF\\PrHard")

column_names = ['parents', 'has_nurs', 'form', 'children', 'housing',
                'finance', 'social', 'health', 'result']
raw_dataset = pd.read_csv(dataset_path, names=column_names,
                          na_values="?", comment='\t',
                          sep=",", skipinitialspace=True)

#print(raw_dataset.dtypes)
dataset = raw_dataset.copy()
dataset.tail()

# has_nurs
dataset["has_nurs"] = dataset["has_nurs"].replace(["very_crit", "critical", "improper", "less_proper", "proper"],[0, 1, 2, 3, 4])
# parents
dataset["parents"] = dataset["parents"].replace(["great_pret", "pretentious", "usual"], [0, 1, 2])
# form
dataset["form"] = dataset["form"].replace(["foster", "incomplete", "completed", "complete"], [0, 1, 2, 3])
# children
dataset = dataset.replace(["1","2","3","more"], [0,1,2,3])
# housing
dataset["housing"] = dataset["housing"].replace(["critical", "less_conv", "convenient"], [0, 1, 2])
# finance
dataset["finance"] = dataset["finance"].replace(["convenient", "inconv"], [1, 0])
# social
dataset["social"] = dataset["social"].replace(["nonprob", "slightly_prob", "problematic"], [2, 1, 0])
# health
dataset["health"] = dataset["health"].replace(["recommended", "priority", "not_recom"], [2, 1, 0])
# result
dataset["result"] = dataset["result"].replace(["not_recom", "recommend", "very_recom", "priority", "spec_prior"], [0, 1, 2, 3, 4])

#make values yes and no
dataset["result"] = dataset["result"].replace([0, 1, 2, 3, 4],[0, 0, 1, 1, 1])

train_dataset = dataset.sample(frac=0.5, random_state=0)
test_dataset = dataset.drop(train_dataset.index)


pop_train=train_dataset.pop('result')
pop_test = test_dataset.pop('result')

train_labels=pd.DataFrame(columns=['result'])
train_labels['result']=pop_train
test_labels=pd.DataFrame(columns=['result'])
test_labels['result']=pop_test

# One hot?
#train_labels=pd.get_dummies(train_labels, prefix=['result'])
#print(train_labels)

train_dataset=train_dataset.transpose()
test_labels=test_labels.transpose()
train_labels=train_labels.transpose()
test_dataset=test_dataset.transpose()
#print(train_dataset.head())
#print(train_labels.head())
#print(test_labels.head())


print ("train_dataset shape: " + str(train_dataset.shape))
print ("train_labels shape: " + str(train_labels.shape))
print ("test_dataset shape: " + str(test_dataset.shape))
print ("test_labels shape: " + str(test_labels.shape))

f = open('params3.pickle', 'rb')
g = open('costs3.pickle', 'rb')

params = pickle.load(f)
costs = pickle.load(g)

print(params)

predict_train=L_predict(train_dataset, train_labels, params)
predict_train=L_predict(test_dataset,test_labels, params)
