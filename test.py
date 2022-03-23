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


training_data, test_data = train_test_split(pd.read_pickle("encoded_data.pk1"), test_size=0.2, stratify=pd.read_pickle("encoded_data.pk1")["Ames"], random_state=34783)
training_data = training_data.reset_index(drop=True);   test_data = test_data.reset_index(drop=True)

encoding = "MACCS"
##      Building model, note encoding already performed
instances = np.array(training_data[encoding].to_list())
labels = training_data["Ames"].to_list()    
tpot_optimisation = TPOTClassifier(generations=10, population_size=500, cv=5, verbosity=2, n_jobs=-1)                                                       
tpot_optimisation.fit(instances,labels)    
##      Testing model
model = tpot_optimisation.fitted_pipeline_  ## This takes the best fitted pipeline developed
instances = np.array(test_data[encoding].to_list())
true_labels = test_data["Ames"].to_list()       
predictions = model.predict(instances) 
predicted_probabilities = model.predict_proba(instances)                                  
predicted_labels = list(map(pos_or_neg,predictions))   