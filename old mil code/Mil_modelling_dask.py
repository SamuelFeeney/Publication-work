#!/usr/bin/env python
# coding: utf-8

# ## Calling packages and assigning variables
# Here i call the necessary packages as well as assigning variables. Of note are the paths to collected data which will need to be changed for replication in another system

# In[22]:


import pandas as pd
import misvm
import rdkit
import numpy as np
import pickle
import os
from dask.distributed import Client
from tpot import TPOTClassifier
from rdkit import Chem
from rdkit.Chem import MACCSkeys
from mordred import Calculator, descriptors
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import train_test_split

if not os.path.isdir("src/misvm"):
    get_ipython().system('pip install -e git+https://github.com/garydoranjr/misvm.git#egg=misvm')

data = pd.read_csv("/home/samuel/honours_redo/1-data/selected_molecules.csv")
metabolite_data = pd.read_csv("/home/samuel/honours_redo/publication_work/biotransformer_output_phaseII.csv").append(pd.read_csv("/home/samuel/honours_redo/publication_work/biotransformer_output_cyp1.csv"))


# # Cleaning Biotransformer data
# This section is used to transform the input data such that it is usable for model building. This involves matching metabolite to parent molecules as well as finding
# thier canonical smiles. <br /><br />
# Additionally this section calculates the encoding of each molecule for modeling to save on time

# In[23]:


def normalize_smiles(smi):
    try:
        smi_norm = Chem.MolToSmiles(Chem.MolFromSmiles(smi))
        return smi_norm
    except:
        return np.nan

def parent_finder(smi):
    for parent in data['smiles']:
        try:
            if Chem.MolToSmiles(Chem.MolFromSmiles(smi)) == Chem.MolToSmiles(Chem.MolFromSmiles(parent)):
                return parent
        except:
            continue
    return "No parent found"

def number_check(x):
    try:
        return float(x)
    except:
        return "broken"

def get_ml_encoding(df, function=MACCSkeys.GenMACCSKeys):                                                                    
    df1 = df.copy()                                                                     
    df1['fp_list'] = df1['smiles'].apply(lambda x: list(function(Chem.MolFromSmiles(x))))     
    df1 = df1.dropna(axis = 1, how = 'any')                                             

    df2 = pd.DataFrame(df1['fp_list'].to_list())                                        
    df2 = df2.applymap(number_check).dropna(axis =1, how = "any")                       
    df2 = df2.drop(columns=df2.columns[(df2 == 'broken').any()])                        
    X = [[int(i) for i in lst] for lst in df2.values.tolist()]                                                                                                                          
    return X

def bag_parent(smiles,met_df,function):
    mol_family          =   met_df[met_df["parent smiles"]==smiles].append({'smiles':smiles},ignore_index=True).drop_duplicates(subset=["smiles"])
    mol_family_encoded  =   get_ml_encoding(df = mol_family, function = function)
    return mol_family_encoded


# In[24]:


if os.path.isfile("encoded_data.pk1"):
    print("Data already encoded")

else:
    ##          Step 1: Normailizing metabolite smiles and matching to parent (approx 220 secs) 
    metabolite_data['smiles'] = metabolite_data['SMILES'].apply(lambda x: normalize_smiles(x))
    metabolite_data = metabolite_data.dropna(axis=0,subset=['smiles'])
    metabolite_data['parent smiles'] = metabolite_data['Precursor SMILES'].apply(lambda x:parent_finder(x))

    ##          Step 2: Pre calculating encoding for molecules, requires evaluation of lists on loading csv (approx 110 secs)
    data["MACCS"] = get_ml_encoding(df = data, function = MACCSkeys.GenMACCSKeys)
    data["RDKF"] = get_ml_encoding(df = data, function =  Chem.RDKFingerprint)
    data["MACCS_MIL"] = data.apply(lambda row: bag_parent(smiles = row['smiles'], met_df = metabolite_data, function = MACCSkeys.GenMACCSKeys),axis=1)
    data["RDKF_MIL"] = data.apply(lambda row: bag_parent(smiles = row['smiles'], met_df = metabolite_data, function = Chem.RDKFingerprint),axis=1)
    data = data.drop(["Molecule"],axis=1)
    ##          Step 3: Saved to a pickle, rather than a csv this stores the lists and is much faster to load (~10x)
    data.to_pickle("encoded_data.pk1")


# # Defining functions
# This section is where i define functions for model development. <br /> If you're curious on how it is done please look here

# In[39]:


def check_if_tested(suffix,model_name,encoding):
    if not os.path.isfile("total_results.pk1"): ## Checking if this has already been tested to save on time
        already_complete = False
    else:
        results = pd.read_pickle("total_results.pk1")
        already_complete = all([(suffix["fold"] in results["fold"]),(suffix["iteration"] in results["iteration"]),(model_name in results["model"]),(encoding in results["encoding"])])
    return already_complete

def build_test_mil_model(training_data,testing_data,MIL,encoding,suffix,save_model,model_name = ""):
    already_complete = check_if_tested(suffix=suffix,encoding=encoding,model_name=model_name)
    if not already_complete:
        bags = training_data[encoding+"_MIL"].to_list()
        labels = training_data["Ames"].apply(lambda x: x if x==1 else -1).to_list()
        model = MIL                                                              
        model.fit(bags,labels)    

        bags = testing_data[encoding+"_MIL"].to_list()
        true_labels = testing_data["Ames"].apply(lambda x: x if x==1 else -1).to_list()
        predictions = model.predict(bags)                                        
        predicted_labels = list(map(pos_or_neg,predictions))                            
        df = pd.DataFrame({
            'predicted' : predictions,
            'predicted labal' : predicted_labels,
            'true label' : true_labels
        })  
        save_results(df = df, suffix = suffix, model = model_name, encoding = encoding)
        if save_model:
            save_models(model = model, path = "/home/samuel/honours_redo/publication_work/saved_models/"+model_name+"_"+str(suffix["fold"])+"_"+str(suffix["iteration"]+".sav"))
    else:
        print("Already tested   fold:",suffix["fold"],"   iteration:",suffix["iteration"],"   model:",model_name,"   encoding:",encoding)

def build_test_ml_model(training_data,testing_data,encoding,ML,suffix,save_model):
    already_complete = check_if_tested(suffix=suffix,encoding=encoding,model_name="TPOT")
    if not already_complete:
        instances = np.array(training_data[encoding].to_list())
        labels = np.array(training_data["Ames"].to_list())       
        tpot_optimisation = ML                                                          
        tpot_optimisation.fit(instances,labels)    
        model = tpot_optimisation.fitted_pipeline_                                                                 

        instances = testing_data[encoding].to_list()
        true_labels = testing_data["Ames"].to_list()       
        predictions = model.predict(instances)                                   
        predicted_labels = list(map(pos_or_neg,predictions))                            
        df = pd.DataFrame({
            'predicted' : predictions,
            'predicted labal' : predicted_labels,
            'true label' : true_labels
        })   
        save_results(df = df, suffix = suffix, model = "TPOT", encoding = encoding)  
        if save_model:
            save_models(model = model, path = "/home/samuel/honours_redo/publication_work/saved_models/TPOT_"+str(suffix["fold"])+"_"+str(suffix["iteration"]+".sav"))

    else:
        print("Already tested   fold:",suffix["fold"],"   iteration:",suffix["iteration"],"   model:","TPOT","   encoding:",encoding)

def pos_or_neg(x):
    if x>0:
        return 1
    else:
        return -1

def format_results(df,suffix,model,encoding):
    df["fold"]  =   suffix["fold"]
    df["iteration"] =   suffix["iteration"]
    df['index'] = df.index
    df["model"] =   model
    df["encoding"] =   encoding
    return df

def save_results(df,suffix,model,encoding):
    if not os.path.isfile("total_results.pk1"):
        df_formatted = format_results(df=df,suffix=suffix,model=model,encoding=encoding)
        df_formatted.to_pickle("total_results.pk1")
    else:
        total_results = pd.read_pickle("total_results.pk1")
        df_formatted = format_results(df=df,suffix=suffix,model=model,encoding=encoding)
        total_results = total_results.append(df_formatted)
        total_results.to_pickle("total_results.pk1")
        
def save_models(model,path):
    pickle.dump(model, open(path, 'wb'))

def develop_models(training_data,testing_data,suffix={"fold":"","iteration":""},encoding="MACCS",save_model=False):
    tested_mils =  [["MICA", misvm.MICA(max_iters=50,verbose=False)],     
                ["MISVM", misvm.MISVM(kernel='linear', C=1.0, max_iters=50,verbose=False)],
                ['SIL', misvm.SIL(verbose=False)],
                ['NSK', misvm.NSK(verbose=False)],
                ['sMIL', misvm.sMIL(verbose=False)]]

    fps = ["MACCS","RDFP"]
    if not encoding in fps:
        print('Please use expected encoding: ["MACCS", "RDFP"]')
        return
    
    # Iterate over the used MILs
    for mil in tested_mils:
        print("     Building and testing:",mil[0],"    fold:",suffix["fold"],"    Iteration:",suffix["iteration"])
        build_test_mil_model(training_data=training_data,testing_data=testing_data,suffix=suffix,MIL=mil[1],encoding=encoding,model_name=mil[0],save_model=save_model)
    
    # Build and test TPOT model
    print("     Building and testing: TPOT     fold:",suffix["fold"],"    Iteration:",suffix["iteration"])
    build_test_ml_model(training_data=training_data,testing_data=testing_data, ML = TPOTClassifier(generations=10, population_size=500, cv=5, verbosity=2,use_dask=True, n_jobs=-1),encoding=encoding,suffix=suffix,save_model=save_model)


# ## Building models
# Here the above functions are used to build models. This section can be altered to build additional models if desired

# In[26]:


## Setting up Dask to allow parrallel training. If you don't want this please hash this out and change "use_dask=True" to "use_dask=False" in the develop models function
client = Client()
client


# In[40]:


## Selecting encoding method, can be changed to RDKF if desired
encoding = "MACCS"

##          Step 1: splitting data into a hold out validation dataset
encoded_data = pd.read_pickle("encoded_data.pk1")
training_data, validation_data = train_test_split(encoded_data, test_size=0.2, stratify=encoded_data["Ames"], random_state=34783)
training_data = training_data.reset_index(drop=True);   validation_data = validation_data.reset_index(drop=True)

##          Step 2: Repeated stratified crossvalidation on training data
rskf = RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=6234794)
for fold,[train_index, test_index] in enumerate(rskf.split(training_data, training_data["Ames"])):
    train   =   training_data.iloc[train_index]
    test    =   training_data.iloc[test_index]
    develop_models(training_data=train,testing_data=test,encoding = encoding,suffix={"fold":fold%10,"iteration":fold//10},save_model=False)
    print("Done Fold", "    fold:",fold%10,"    iteration:",fold//10)

# ##          Step 3: model building on training data against holdout validation data
# develop_models(training_data=train,testing_data=test,encoding = encoding,suffix={"fold":"","iteration":"validation"},save_model=True)


# ## Model Validation
# Here the model results are assessed

# In[ ]:


# develop_models(data,bt_data,validation_data,validation_metabolite_data,suffix="_validation")


# ## Model Analysis
# Here the results of each fold are calculated as well as deviation within crossvalidation

# In[ ]:


def confusion_matrix(df):
    TP = len(df[(df["predicted label"] == 1) & (df["true label"] == 1)])
    TN = len(df[(df["predicted label"] == 0) & (df["true label"] == 0)])
    FP = len(df[(df["predicted label"] == 1) & (df["true label"] == 0)])
    FN = len(df[(df["predicted label"] == 0) & (df["true label"] == 1)])
    return [TP,TN,FP,FN]


# In[ ]:


# rslt_list = []

# crossvalidation_results = pd.read_pickle("total_results.pk1")
# for iteration in crossvalidation_results["iteration"].unique():
#     for fold in crossvalidation_results["fold"].unique():
#         for model in crossvalidation_results["model"].unique():
#             for encoding in crossvalidation_results["encoding"].unique():
#                 working_data = crossvalidation_results[(crossvalidation_results["fold"]==fold)&(crossvalidation_results["iteration"]==iteration)&(crossvalidation_results["model"]==model)&(crossvalidation_results["encoding"]==encoding)]
#                 [TP,TN,FP,FN] = confusion_matrix(working_data)
#                 rslt_list += [{"encoding":encoding, "model":model, "fold":fold, "iteration":iteration, "TP":TP, "TN":TN, "FP":FP, "FN":FN}]
# rslt_df = pd.Dataframe(rslt_list)

# mean_rslt_list = []
# for model in rslt_df["model"].unique():
#     for encoding in rslt_df["encoding"].unique():
#         working_data = rslt_df[(rslt_df["model"]==model)&(rslt_df["encoding"]==encoding)]
#         mean_rslt_list += [{"encoding":encoding, "model":model, "Mean TP":working_data["TP"].mean(), "Mean TN":working_data["TN"].mean(), "Mean FP":working_data["FP"].mean(), "Mean FN":working_data["FN"].mean()}]
# mean_rslt_df = pd.DataFrame(mean_rslt_list)

