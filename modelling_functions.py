import pandas as pd
import misvm
import os
import pickle
import numpy as np
from tpot import TPOTClassifier


def check_if_tested(suffix,model_name,encoding):    ## Checking if this build/test has already been done. Saves on time if a run crashes
    if not os.path.isfile("total_results.pk1"): 
        already_complete = False
    else:
        results = pd.read_pickle("total_results.pk1")
        already_complete = ((results["fold"].isin([suffix["fold"]])) & (results["iteration"].isin([suffix["iteration"]])) & (results["model"].isin([model_name])) & (results["encoding"].isin([encoding]))).any()
    return already_complete

def build_test_mil_model(training_data,testing_data,MIL,encoding,suffix,save_model,model_name = ""):    ## Build and test a MIL model
    already_complete = check_if_tested(suffix=suffix,encoding=encoding,model_name=model_name)
    if not already_complete:
        ##      Building model, note encoding already performed
        bags = training_data[encoding+"_MIL"].to_list()
        labels = training_data["Ames"].apply(lambda x: x if x==1 else -1).to_list()
        model = MIL                                                              
        model.fit(bags,labels)    
        ##      Testing model
        bags = testing_data[encoding+"_MIL"].to_list()
        labels = testing_data["Ames"].apply(lambda x: x if x==1 else -1).to_list()
        predictions = model.predict(bags)                                        
        predicted_labels = list(map(pos_or_neg,predictions))                            
        df = pd.DataFrame({
            'predicted' : predictions,
            'predicted labal' : predicted_labels,
            'true label' : labels
        })  
        save_results(df = df, suffix = suffix, model = model_name, encoding = encoding)
        if save_model:
            save_models(model = model, path = "/saved_models/"+model_name+"_"+str(suffix["fold"])+"_"+str(suffix["iteration"]+".sav"))
    else:
        print("Already tested   fold:",suffix["fold"],"   iteration:",suffix["iteration"],"   model:",model_name,"   encoding:",encoding)

def build_test_ml_model(training_data,testing_data,encoding,ML,suffix,save_model):                      ## Build and test a machine learning model
    already_complete = check_if_tested(suffix=suffix,encoding=encoding,model_name="TPOT")
    if not already_complete:
        ##      Building model, note encoding already performed
        instances = np.array(training_data[encoding].to_list())
        labels = training_data["Ames"].to_list()    
        tpot_optimisation = ML                                                          
        tpot_optimisation.fit(instances,labels)    
        ##      Testing model
        model = tpot_optimisation.fitted_pipeline_  ## This takes the best fitted pipeline developed
        instances = np.array(testing_data[encoding].to_list())
        true_labels = testing_data["Ames"].to_list()       
        predictions = model.predict(instances) 
        try:
            predicted_probabilities = model.predict_proba(instances)      
        except:
            predicted_probabilities = predictions
        predicted_labels = list(map(pos_or_neg,predictions))                            
        df = pd.DataFrame({
            'predicted' : [i[1] for i in predicted_probabilities],
            'predicted labal' : predicted_labels,
            'true label' : true_labels
        })   
        save_results(df = df, suffix = suffix, model = "TPOT", encoding = encoding)  
        if save_model:
            save_models(model = model, path = "/saved_models/TPOT_"+str(suffix["fold"])+"_"+str(suffix["iteration"]+".sav"))
    else:
        print("Already tested   fold:",suffix["fold"],"   iteration:",suffix["iteration"],"   model:","TPOT","   encoding:",encoding)

def pos_or_neg(x):  ## Simple function used to translate predictions between MIL and ML into a single form
    if x>0:
        return 1
    else:
        return 0

def format_results(df,suffix,model,encoding):  ## adds informative columns to the df used for saving results 
    df["fold"]  =   suffix["fold"]
    df["iteration"] =   suffix["iteration"]
    df['index'] = df.index
    df["model"] =   model
    df["encoding"] =   encoding
    return df

def save_results(df,suffix,model,encoding):     ## saves results to a single pickle, adding to it or generating it
    if not os.path.isfile("total_results.pk1"):
        df_formatted = format_results(df=df,suffix=suffix,model=model,encoding=encoding)
        df_formatted.to_pickle("total_results.pk1")
    else:
        total_results = pd.read_pickle("total_results.pk1")
        df_formatted = format_results(df=df,suffix=suffix,model=model,encoding=encoding)
        total_results = pd.concat([total_results,df_formatted], ignore_index=True)
        total_results.to_pickle("total_results.pk1")
        
def save_models(model,path):                    ## Saves model to a path
    pickle.dump(model, open(path, 'wb'))

def develop_models(training_data,testing_data,suffix={"fold":"","iteration":""},encoding="MACCS",save_model=False, dask=False,MIL=True,TPOT=True):     ## single function to complete whole pipeline for a set of data to all expected models
    ##      Step 0:     Checking that the encoding method described is expected
    fps = ["MACCS","RDFP"]
    if not encoding in fps:
        print('Please use expected encoding: ["MACCS", "RDFP"]')
        return
    
    ##      Step 1:     Model generation
    tested_mils =  [
                # ["MICA", misvm.MICA(max_iters=50,verbose=False)],     
                ["MISVM", misvm.MISVM(kernel='linear', C=1.0, max_iters=50,verbose=False)],
                ['SIL', misvm.SIL(verbose=False)],
                ['NSK', misvm.NSK(verbose=False)],
                ['sMIL', misvm.sMIL(verbose=False)]]
            ## note: Either as dask or non-dask TPOT can be used, defined in function variables
    tpot_model = TPOTClassifier(generations=10, population_size=500, cv=5, verbosity=2, n_jobs=-1)

    ##      Step 2:     Build and test models
    if MIL:         # Iterate over the used MILs
        for mil in tested_mils:
            print("     Building and testing:",mil[0],"    fold:",suffix["fold"],"    Iteration:",suffix["iteration"])
            build_test_mil_model(training_data=training_data,testing_data=testing_data,suffix=suffix,MIL=mil[1],encoding=encoding,model_name=mil[0],save_model=save_model)
    if TPOT:        # Build and test TPOT model   
        print("     Building and testing: TPOT     fold:",suffix["fold"],"    Iteration:",suffix["iteration"])
        build_test_ml_model(training_data=training_data,testing_data=testing_data, ML = tpot_model,encoding=encoding,suffix=suffix,save_model=save_model)