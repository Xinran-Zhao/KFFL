import datasets
import utilites as utils
from models import BinaryLogisticRegression,NN
import main
import pickle
import os
import torch
import sys
import numpy as np
# dataset = datasets.get_adult()

# local_datasets,test = utils.create_local_datasets(dataset,'Non-IID',4,40)

# num_features = utils.get_num_features(dataset)
#model = NN(num_features - 1)

# print(len(dataset))
# print(len(local_datasets))
# torch.autograd.set_detect_anomaly(True)



#print("Correct File")
dataset = ['ADULT'] #,#'COMPAS']
model = ['LR'] #'LR']  # 'NN' --> KRTD and KRTWD
#methods = ['KRTWD','KRTD']#'KRTD' , # 'FairFed_w_FairBatch_kernel'
methods = ['FairFed_w_FairBatch'] # 'KRTWD','Central','FairFed_w_FairBatch', 'FedAvg','MinMax''FairFed_w_FairBatch_kernel'
#methods = ['KRTWD']
dist = ['Non-IID'] #'IID'] 

#torch.manual_seed(0) 
numsims = 1

# Define fairness parameters for KRTWD and KRTD
fairness_params = {
    'KRTWD': list( np.linspace(20, 1000, 20)),#list( np.logspace(-9, -6, 7)), ## between 10** -8 and 10** -9
    'KRTD': [100],#list( np.logspace(-3, -1, 7)) #list( np.logspace(-3, -1, 7))#list( np.logspace(-3, -1, 7)), #*0.0000005],#[1e-10, 1e0, 1e1, 1e2, 1e10]
    'FairFed_w_FairBatch_kernel':[0],
    'Central' : [10**-8,10**-9,10**-10,10**-11,10**-12,10**-13]
}
#seeds =[i for i in range(numsims)]
for i in range(numsims):
    # Comparison by methods, dataset, model, distribution, and fairness weight
    print(f"**************************Simulation is {i}***************")
    #torch.manual_seed(seeds[i])
    for distribution in dist:
        
        print(f"Distribution is {distribution}")
        for d in dataset:
            for mdl in model:
                for m in methods:
                    if m in fairness_params:
                        fairness_weights = fairness_params[m]
                    else:
                        fairness_weights = [None]  # Use None for methods other than KRTWD and KRTD
                    
                    for fairness_weight in fairness_weights:
                        print(f'Fairness weight is : {fairness_weight}')
                        if fairness_weight is not None:
                            if(distribution =='IID'):
                                
                                filename = f"{m}_{mdl}_{d}_{distribution}_{fairness_weight}_test_results.pickle"
                            else:
                                filename = f"{m}_{mdl}_{d}_{distribution}_{fairness_weight}_test_results_90_10.pickle"
                        else:
                            if(distribution  == 'IID'):
                                
                                filename = f"{m}_{mdl}_{d}_{distribution}_{fairness_weight}_test_results.pickle"
                            else:
                                filename = f"{m}_{mdl}_{d}_{distribution}_{fairness_weight}_test_results_90_10.pickle"
                        
                        #params = {'fairness': fairness_weight}
                        #results, results_std = main.simulation_runs(m, mdl, distribution, d, numsims, params)
                        
                        # Check if the file already exists
                        #if not os.path.exists(filename):
                        params = {'fairness': fairness_weight} if fairness_weight is not None else None
                        results, results_std = main.simulation_runs(m, mdl, distribution, d, numsims, params)
                        with open(filename, "wb") as file:
                            # Save variables using pickle
                            pickle.dump(results, file)
                            pickle.dump(results_std, file)
                            pickle.dump(d, file)
                            pickle.dump(mdl, file)
                            pickle.dump(distribution, file)
                            if fairness_weight is not None:
                                pickle.dump(fairness_weight, file)
                       # else:
                       #     print("Should not print this")
                       #     print(f"File {filename} already exists, skipping simulation.")



