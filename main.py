import pandas as pd
import models
from models import BinaryLogisticRegression,NN,AdultNN
import datasets
import torch
from methods import run_KRTWD,run_KRTD,run_FedAvg,run_MinMax,run_FairFed,run_FairFed_kernel,run_Centralized
import utilites as utils
from utilites import get_num_features



def simulation_spec(method,model,client_distribution,desired_data,fairness=None,
                    step_size=0.01, fb_lr=0.005, num_rounds=200, alpha=0.1):
    ######## 1. Choose a dataset ############
    
    if(desired_data == 'ADULT'):
        
        dataset = datasets.get_adult() ####
        protected_index = 40 ## Corresponds to sex

    elif(desired_data == 'COMPAS'):
        
        dataset = datasets.get_compass()  ####
        protected_index = 3  ## Corresponds to African American race
    else:
        print("The following dataset is not suppported at the moment. Choose from: ADULT, COMPAS ")
    
     
    num_sensattr  = 1 ### Default Value
    num_features = get_num_features(dataset)
     
     
    ######## 2. Choose a model #########
    if(model == 'LR'):
        model = BinaryLogisticRegression(num_features - num_sensattr)
         
    elif(model == 'NN'):
         model = AdultNN(num_features - num_sensattr)
        

    
    ######## 3. Choose a method #########
    if(method == 'KRTWD'):
      
        params = {
         'step_size':0.1,
         'batch_size': 64,
         'local_epochs' : 5,
          "total_clients": 4,
          "num_sel":4,
          "num_rounds":10,
         "fairness_weight":fairness,
         "R":50,
         "T":50,
         "D":10,
         }
    
        acc,spd,eod,comm_cost = run_KRTWD(method,model,client_distribution,dataset,protected_index,params)
        
    
    elif(method == 'KRTD'):
        print(fairness)
        params = {
         'step_size':0.1,
         'batch_size': 64,
         'local_epochs' : 5,
          "total_clients": 4,
          "num_sel":4,
          "num_rounds":10,
         "fairness_weight":fairness,
         "R":50,
         "T":50,
         "D":10,
         }
        
        acc,spd,eod,comm_cost = run_KRTD(method,model,client_distribution,dataset,protected_index,params)

    elif(method == 'Central'):

        params = {'epochs': 150,
          'step_size':0.1,
          'batch_size': 64,
          "fairness_weight":fairness,
          }
        # Under Construction
        print(f'The fair weight is {fairness}' ) 
        acc,spd,eod  = run_Centralized(method,model,dataset,protected_index,params)
        comm_cost = 0
        #acc,spd,eod,comm_cost  = run_FairFed(method,model,client_distribution,dataset,protected_index,params,fmetric)
    
    elif(method == 'FedAvg'):
        
        params = {'local_epochs' :5 ,
          "total_clients": 4,
          "num_sel":4,
          'step_size':0.1,
          'batch_size': 64,
          "num_rounds":10}
        
        acc,spd,eod,comm_cost  = run_FedAvg(method,model,client_distribution,dataset,protected_index,params)
    
    elif(method == 'MinMax'):
        params = {'local_epochs' :5 ,
          "total_clients": 4,
          "num_sel":4,
          'step_size':0.1,
          'batch_size': 64,
          "global_adversary_rate":0.0000000001, #0.01,0.001
          "num_rounds":10}
        acc,spd,eod,comm_cost  = run_MinMax(method,model,client_distribution,dataset,protected_index,params)
        
    elif(method == 'FairFed_w_FairBatch'):
        params = {'local_epochs': 5,
          "total_clients": 5,
          "num_sel": 5,
          'step_size': step_size,
          'fb_lr': fb_lr,
          'batch_size': 64,
          'beta': 1,
          "num_rounds": num_rounds,
          'alpha': alpha}
        fmetric = 'eod'
        acc,spd,eod,comm_cost  = run_FairFed(method,model,client_distribution,dataset,protected_index,params,fmetric)
        
        
    elif(method == 'FairFed_w_FairBatch_kernel'):
       params = {'local_epochs' :5,
         "total_clients": 1,
         "num_sel":1,
         'step_size':0.01,
         'batch_size': 64,
         'beta': 1,
         "fairness":fairness,
         "num_rounds":10}
       fmetric = 'spd' #'eod','spd'
       acc,spd,eod,comm_cost  = run_FairFed_kernel(method,model,client_distribution,dataset,protected_index,params,fmetric)
    
    else:
        print("Incorrect Method")
    
    results = {'Acc' : acc, 'SPD': spd, 'EOD' : eod, 'Comm_Cost' : comm_cost}
    
    return results

## Simulation Runs 
def simulation_runs(method, model, client_distribution, dataset, num_simulation, params=None,
                    step_size=0.01, fb_lr=0.005, num_rounds=200, alpha=0.1):
    acc = []
    spd = []
    eod = []
    comm_cost = []
    seeds = [i for i in range(num_simulation)]
    for i in range(num_simulation):
        torch.manual_seed(seeds[i])
        if method in ('KRTD', 'KRTWD', 'FairFed_w_FairBatch_kernel', 'Central'):
            results = simulation_spec(method, model, client_distribution, dataset,
                                      fairness=params['fairness'],
                                      step_size=step_size, fb_lr=fb_lr,
                                      num_rounds=num_rounds, alpha=alpha)
        else:
            results = simulation_spec(method, model, client_distribution, dataset,
                                      step_size=step_size, fb_lr=fb_lr,
                                      num_rounds=num_rounds, alpha=alpha)
        
        acc.append(results['Acc'])
        spd.append(results['SPD'])
        eod.append(results['EOD'])
        comm_cost.append(results['Comm_Cost'])
    
    ACC = pd.DataFrame(acc)
    SPD = pd.DataFrame(spd)
    EOD = pd.DataFrame(eod)
    COMM_COST = pd.DataFrame(comm_cost)
    
    acc_mean = ACC.mean()
    acc_std = ACC.std()
    
    spd_mean = SPD.mean()
    spd_std = SPD.std()
    
    eod_mean = EOD.mean()
    eod_std = EOD.std()
    
    comm_cost_mean = COMM_COST.mean()
    comm_cost_std = COMM_COST.std()
    
    results = {'Acc': acc_mean, 'SPD': spd_mean, 'EOD': eod_mean, 'Comm_Cost': comm_cost_mean}
    results_std = {'Acc': acc_std, 'SPD': spd_std, 'EOD': eod_std, 'Comm_Cost': comm_cost_std}
    
    return results, results_std
    
    
    ### Save file
    
    return results,results_std




        