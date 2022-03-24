import pandas as pd
import numpy as np
import pickle
import os
from tpot import TPOTClassifier

import sklearn
print (sklearn.__version__)
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import train_test_split
import misvm
# try:
#     import misvm 
# except:
#     print("please use command to install MIL modelling package \n pip install -e git+https://github.com/garydoranjr/misvm.git#egg=misvm")
#     quit()

from modelling_functions import *

## Selecting encoding method, can be changed to RDKF if desired
encoding = "MACCS"

##          Step 1: splitting data into a hold out validation dataset
training_data, test_data = train_test_split(pd.read_pickle("encoded_data.pk1"), test_size=0.2, stratify=pd.read_pickle("encoded_data.pk1")["Ames"], random_state=34783)
training_data = training_data.reset_index(drop=True);   test_data = test_data.reset_index(drop=True)

##          Step 2: Repeated stratified crossvalidation on training data
rskf = RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=6234794)
for fold,[train_index, validation_index] in enumerate(rskf.split(training_data, training_data["Ames"])):
    train   =   training_data.iloc[train_index];        validation    =   training_data.iloc[validation_index]
    develop_models(training_data=train,testing_data=validation,encoding = encoding,suffix={"fold":fold%10,"iteration":fold//10},save_model=False,MIL=True,TPOT=False)
        # Note, i run MIL and TPOT seperately due to the time difference in building the models and to catch errors easier
for fold,[train_index, validation_index] in enumerate(rskf.split(training_data, training_data["Ames"])):
    train   =   training_data.iloc[train_index];        validation    =   training_data.iloc[validation_index]
    develop_models(training_data=train,testing_data=validation,encoding = encoding,suffix={"fold":fold%10,"iteration":fold//10},save_model=False,MIL=False,TPOT=True)
    # print("Done Fold", "    fold:",fold%10,"    iteration:",fold//10)


# ##          Step 3: model building on training data against holdout test data
# develop_models(training_data=train,testing_data=test_data,encoding = encoding,suffix={"fold":"","iteration":"Hold out test"},save_model=True)

# ## predict proba for some