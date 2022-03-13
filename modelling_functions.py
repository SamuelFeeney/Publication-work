import pandas as pd
import misvm
import os
import numpy as np
from tpot import TPOTClassifier

def check_if_tested(suffix,model_name,encoding):
    if not os.path.isfile("total_results.pk1"): ## Checking if this has already been tested to save on time
        already_complete = False
    else:
        results = pd.read_pickle("total_results.pk1")
        already_complete = ((results["fold"].isin([suffix["fold"]])) & (results["iteration"].isin([suffix["iteration"]])) & (results["model"].isin([model_name])) & (results["encoding"].isin([encoding]))).any()
    return already_complete

def build_test_mil_model(training_data,testing_data,MIL,encoding,suffix,save_model,model_name = ""):
    already_complete = check_if_tested(suffix=suffix,encoding=encoding,model_name=model_name)
    if not already_complete:
        ##      Building model
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
            save_models(model = model, path = "/saved_models/TPOT_"+str(suffix["fold"])+"_"+str(suffix["iteration"]+".sav"))

    else:
        print("Already tested   fold:",suffix["fold"],"   iteration:",suffix["iteration"],"   model:","TPOT","   encoding:",encoding)

def pos_or_neg(x):
    if x>0:
        return 1
    else:
        return 0

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
    tested_mils =  [
                ["MICA", misvm.MICA(max_iters=50,verbose=False)],     
                ["MISVM", misvm.MISVM(kernel='linear', C=1.0, max_iters=50,verbose=False)],
                ['SIL', misvm.SIL(verbose=False)],
                ['NSK', misvm.NSK(verbose=False)],
                ['sMIL', misvm.sMIL(verbose=False)]]
    # tpot_model = TPOTClassifier(generations=10, population_size=500, cv=5, verbosity=2,use_dask=True, n_jobs=-1)
    tpot_model = TPOTClassifier(generations=10, population_size=500, cv=5, verbosity=2)

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
    build_test_ml_model(training_data=training_data,testing_data=testing_data, ML = tpot_model,encoding=encoding,suffix=suffix,save_model=save_model)