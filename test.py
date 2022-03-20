import pandas as pd
import numpy as np
import pickle
import os
from tpot import TPOTClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import train_test_split
try:
    import misvm 
except:
    print("please use command to install MIL modelling package \n pip install -e git+https://github.com/garydoranjr/misvm.git#egg=misvm")
    quit()

from modelling_functions import *


dask = False
## Selecting encoding method, can be changed to RDKF if desired
encoding = "MACCS"

##          Step 1: splitting data into a hold out validation dataset
training_data, validation_data = train_test_split(pd.read_pickle("encoded_data.pk1"), test_size=0.2, stratify=pd.read_pickle("encoded_data.pk1")["Ames"], random_state=34783)
training_data = training_data.reset_index(drop=True);   validation_data = validation_data.reset_index(drop=True)

##          Step 2: Repeated stratified crossvalidation on training data
rskf = RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=6234794)
for fold,[train_index, test_index] in enumerate(rskf.split(training_data, training_data["Ames"])):
    train   =   training_data.iloc[train_index]
    test    =   training_data.iloc[test_index]
    develop_models(training_data=train,testing_data=test,encoding = encoding,suffix={"fold":fold%10,"iteration":fold//10},save_model=False,dask=dask)
    print("Done Fold", "    fold:",fold%10,"    iteration:",fold//10)

# ##          Step 3: model building on training data against holdout validation data
# develop_models(training_data=train,testing_data=test,encoding = encoding,suffix={"fold":"","iteration":"validation"},save_model=True)
print("DONE")