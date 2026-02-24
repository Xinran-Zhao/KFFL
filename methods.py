import models
import utilites as utils
import kernel_utils as k_util
import torch
import numpy as np
from utilites import fairbatch_dataset
from eval_metrics import spd,eoo_binary_attribute,model_accuracy
from tqdm import tqdm
from kernel_utils import rbf_kernel



def run_KRTWD(method,model,client_distribution,dataset,protected_index,params):
   
    device = 'cpu'
    
    ## Loading the client models and global model
    global_model = model
    
    ## Make these clients in a dictionary next time?
    client_models = [ model.to(device) for _ in range(params["num_sel"])]
    
    
    fair_gradients = [ torch.zeros(5,5) for _ in range(params["num_sel"])]
    

    
    for model in client_models:    
        model.load_state_dict(global_model.state_dict())
    
    Sub_round = ['First','Second','Third']
    return_obj = None
    global_fairness_term = None
    #communication_cost = {'Uplink':0, 'Downlink': 0}
    
    
    ################ Generation of Random Seed ###############################################
    random_seeds = torch.randint(torch.iinfo(torch.int32).max,
                                     torch.iinfo(torch.int64).max, 
                                     tuple([1]), dtype=torch.int64)

    #client_fair_loss = 0
    
    train_datasets,trainset,testset = utils.create_local_datasets(dataset,client_distribution,params["num_sel"],protected_index)
    
    weights,weights_norm = utils.calculate_weights(train_datasets)
    
    per_round_cost = 0 
    cost_w_iter = []
    accuracy = []
    statistical_parity_difference =[]
    equalized_odds = []
    for r in range(params["num_rounds"]):
        
        client_idx = np.random.permutation(params["total_clients"])[:params["num_sel"]] # Handles Partial Participation
        client_local_information =[]
        #fair_gradients = client_models.clone()
        for round in Sub_round:
            for i in tqdm(range(params["num_sel"])):
                if(round == 'First'):
                    
                    return_obj = k_util.client_update(client_models[i],global_model,train_datasets[client_idx[i]],protected_index,
                    random_seeds[0],round,global_fairness_term,params,epoch = r)
                    client_local_information.append(return_obj['local_contirbution'])

                    #communication_cost['Downlink']= communication_cost['Downlink'] +  utils.parameter_count(client_models[i])
                    #communication_cost['Uplink']= communication_cost['Uplink'] +  utils.parameter_count(client_models[i])
                elif(round == 'Second'):
                    #print(f"client_models{i}")
                    return_obj = k_util.client_update(client_models[i],global_model,train_datasets[client_idx[i]],protected_index,
                    random_seeds[0],round,global_fairness_terms,params,weights[i],epoch = r)
                    #print(f"Issue here????")
                    client_models[i] = return_obj['client_model']
                   # print(i)
                    fair_gradients[i] = return_obj['fair_grad']
                   # communication_cost['Downlink']= communication_cost['Downlink'] +  utils.parameter_count(client_models[i])
                    #communication_cost['Uplink']= communication_cost['Uplink']  + 2 * params['R']*params['D'] +  2 * params['T']*params['D']
                    
                elif(round == 'Third'):

                    return_obj= k_util.client_update(client_models[i],global_model,train_datasets[client_idx[i]],protected_index,
                    random_seeds[0],round,global_fairness_terms,params,epoch = r)
                    #print(return_obj)
                    client_models[i] = return_obj['client_model']
                    client_models[i].eval()

                    ## Save the gradients here as well:
                    
                    
                   # communication_cost['Downlink']= communication_cost['Downlink']  + 2 * params['R']*params['D'] +  2 * params['T']*params['D']
                    #communication_cost['Uplink']= communication_cost['Uplink'] +  utils.parameter_count(client_models[i])
                    
                    # client_fair_loss = return_obj['fair_loss']
                    # fair_loss+=client_fair_loss
                    
                    
                    client_models[i].eval()
                    
            
            if(round =='First'):
                
                return_obj = k_util.server_aggregate(global_model,client_models,client_local_information,weights_norm,weights,params, 
                                                     round)
               
                global_fairness_terms= return_obj

            elif(round == 'Second'):
                return_obj = k_util.server_aggregate(global_model,client_models,client_local_information,weights_norm,weights,params,
                                                 round,fair_gradients)
                client_models = return_obj['client_model']

                
            elif(round == 'Third'):
                return_obj = k_util.server_aggregate(global_model,client_models,client_local_information,weights_norm,weights,params,
                                                 round)     
                #print(return_obj)
                client_models = return_obj['client_model']
                
            
           # per_round_cost = utils.convert_to_megabytes(sum(communication_cost.values()) * 32)   
             
       
        if(r%1==0):

            with torch.no_grad():
                cost_w_iter.append(per_round_cost) 
                modified_testset = utils.drop_attribute(testset,protected_index,weighted=False)
                
                acc = model_accuracy(global_model, modified_testset, binary = True)
                spd_act,_,_= spd(global_model, testset,protected_index)
                eod_act,_,_ = eoo_binary_attribute(global_model, testset,protected_index)
                
                accuracy.append(acc)
                statistical_parity_difference .append(spd_act)
                equalized_odds.append(eod_act)
                
                
                
                print(f'Round Number {r}')
                print(f'accuracy {acc}: SPD: {spd_act :.4f} EOD: {eod_act:.4f} Cost: {per_round_cost: .4f}')

    return accuracy,statistical_parity_difference,equalized_odds,cost_w_iter




def run_KRTD(method,model,client_distribution,dataset,protected_index,params):
   
    device = 'cpu'
    
    ## Loading the client models and global model
    global_model = model
    
    ## Make these clients in a dictionary next time?
    client_models = [ model.to(device) for _ in range(params["num_sel"])]
    
    
    fair_gradients = [ torch.zeros(5,5) for _ in range(params["num_sel"])]
    

    
    for model in client_models:    
        model.load_state_dict(global_model.state_dict())
    
    return_obj = None
    global_fairness_term = None
    #communication_cost = {'Uplink':0, 'Downlink': 0}
    
    random_seeds = torch.randint(torch.iinfo(torch.int32).max,
                                     torch.iinfo(torch.int64).max, 
                                     tuple([1]), dtype=torch.int64)

    #client_fair_loss = 0
    
    train_datasets,trainset,testset = utils.create_local_datasets(dataset,client_distribution,params["num_sel"],protected_index)
    weights,weights_norm = utils.calculate_weights(train_datasets)
    per_round_cost = 0 
    cost_w_iter = []
    accuracy = []
    statistical_parity_difference =[]
    equalized_odds = []
    for r in range(params["num_rounds"]):
    
        client_idx = np.random.permutation(params["total_clients"])[:params["num_sel"]]
        client_local_information =[] 
        for i in tqdm(range(params["num_sel"])):
            #import pdb;pdb.set_trace()
            return_obj= k_util.client_update_TD(client_models[i],global_model,train_datasets[client_idx[i]],protected_index,
            random_seeds[0],r,global_fairness_term,params,weights[i],epoch = r)
                    #print(return_obj)
            client_models[i] = return_obj['client_model']
            fair_gradients[i] = return_obj['fair_grad']
            client_local_information .append(return_obj['local_contributions'])
            client_models[i].eval() 
            
            
         ## Server Calling
        if(r==0):
            
            return_obj = k_util.server_aggregate_TD(global_model,client_models,
                                             client_local_information,weights_norm,weights,
                                             params,r)
            #import pdb;pdb.set_trace()
            client_models = return_obj['client_model']
            global_fairness_term= return_obj['global_fairness']
            
        else:
            
            return_obj = k_util.server_aggregate_TD(global_model,client_models,
                                             client_local_information,weights_norm,weights,
                                             params,r,fair_gradients)
            client_models = return_obj['client_model']
            global_fairness_term= return_obj['global_fairness']
         #per_round_cost = utils.convert_to_megabytes(sum(communication_cost.values()) * 32)   
             
       
        if(r%1==0):

            with torch.no_grad():
                cost_w_iter.append(per_round_cost) 
                modified_testset = utils.drop_attribute(testset,protected_index,weighted=False)
                
                acc = model_accuracy(global_model, modified_testset, binary = True)
                spd_act,_,_= spd(global_model, testset,protected_index)
                eod_act,_,_ = eoo_binary_attribute(global_model, testset,protected_index)
                
                accuracy.append(acc)
                statistical_parity_difference .append(spd_act)
                equalized_odds.append(eod_act)
                
                
                
                print(f'Round Number {r}')
                print(f'accuracy {acc}: SPD: {spd_act :.4f} EOD: {eod_act:.4f} Cost: {per_round_cost: .4f}')

    return accuracy,statistical_parity_difference,equalized_odds,cost_w_iter

def run_FedAvg(method,model,client_distribution,dataset,protected_index,params):
   
    device = 'cpu'
    
    ## Loading the client models and global model
    global_model = model
    client_models = [ model.to(device) for _ in range(params["num_sel"])] 
    for model in client_models:    
        model.load_state_dict(global_model.state_dict())
    

    communication_cost = {'Uplink':0, 'Downlink': 0}


    #client_fair_loss = 0
    
    train_datasets,trainset,testset = utils.create_local_datasets(dataset,client_distribution,params["num_sel"],protected_index)
    weights,weights_norm = utils.calculate_weights(train_datasets)
  
    per_round_cost = 0 
    cost_w_iter = []
    accuracy = []
    statistical_parity_difference =[]
    equalized_odds = []
    
    for r in range(params["num_rounds"]):
        
        client_idx = np.random.permutation(params["total_clients"])[:params["num_sel"]]
     
        for i in tqdm(range(params["num_sel"])):
            client_models[i] = k_util.client_update_FAVG(client_models[i],global_model,train_datasets[client_idx[i]],protected_index,params,epoch = r)
                    #print(return_obj)
            communication_cost['Downlink']= communication_cost['Downlink'] +  utils.parameter_count(client_models[i])
            communication_cost['Uplink']= communication_cost['Uplink'] +  utils.parameter_count(client_models[i])
         
            client_models[i].eval()
            
       

       
            
        client_models = k_util.server_aggregate_FAVG(global_model,client_models,weights_norm,params)
        per_round_cost = utils.convert_to_megabytes(sum(communication_cost.values()) * 32)   
             
       
        if(r%1==0):

            with torch.no_grad():
                cost_w_iter.append(per_round_cost) 
                modified_testset = utils.drop_attribute(testset,protected_index,weighted=False)
                
                acc = model_accuracy(global_model, modified_testset, binary = True)
                spd_act,_,_= spd(global_model, testset,protected_index)
                eod_act,_,_ = eoo_binary_attribute(global_model, testset,protected_index)
                
                accuracy.append(acc)
                statistical_parity_difference .append(spd_act)
                equalized_odds.append(eod_act)
                
                
                
                print(f'Round Number {r}')
                print(f'accuracy {acc}: SPD: {spd_act :.4f} EOD: {eod_act:.4f} Cost: {per_round_cost: .4f}')

    return accuracy,statistical_parity_difference,equalized_odds,cost_w_iter




def run_MinMax (method,model,client_distribution,dataset,protected_index,params):
    device = 'cpu'
    
    ## Loading the client models and global model
    device = 'cpu'
    global_model = model
    client_models = [ model.to(device) for _ in range(params["num_sel"])] 
    for model in client_models:    
        model.load_state_dict(global_model.state_dict())
    

    communication_cost = {'Uplink':0, 'Downlink': 0}


    
    train_datasets,trainset,testset = utils.create_local_datasets(dataset,client_distribution,params["num_sel"],protected_index)
    weights,weights_norm = utils.calculate_weights(train_datasets)
    
        # Initialize p
    p,group_stats = utils.rho(trainset,group_index = protected_index)
    p = torch.tensor(p)    
    # Weighing coefficients 
    u = p
    all_client_risks = {}
    per_round_cost = 0 
    cost_w_iter = []
    accuracy = []
    statistical_parity_difference =[]
    equalized_odds = []
    for r in range(params["num_rounds"]):
        
        w = torch.div(u,p)
       # print(w.shape)
    #     #client_idx = np.random.permutation(params["total_clients"])[:params["num_sel"]]
    #     #print("Verifying the selected clients : {}".format(client_idx))
        for i in tqdm(range(params["num_sel"])):
            client_models[i],group_risks,scaled_group_risks= k_util.client_update_minmax(client_models[i],train_datasets[i],w,protected_index)
           # print("Scaled Group Risk",scaled_group_risks)
            all_client_risks[i] = scaled_group_risks
            communication_cost['Downlink']= communication_cost['Downlink'] +  utils.parameter_count(client_models[i]) + len(w)
            communication_cost['Uplink']= communication_cost['Uplink'] +  utils.parameter_count(client_models[i]) + + len(scaled_group_risks)
            client_models[i].eval()
            
        client_models,u = k_util.server_aggregate_minmax(global_model, client_models,weights_norm,all_client_risks,u,group_stats,params)   
        per_round_cost = utils.convert_to_megabytes(sum(communication_cost.values()) * 32)

        if(r%1==0):
            with torch.no_grad():
                cost_w_iter.append(per_round_cost) 
                modified_testset = utils.drop_attribute(testset,protected_index,weighted=False)
                
                acc = model_accuracy(global_model, modified_testset, binary = True)
                spd_act,_,_ = spd(global_model, testset,protected_index)
                eod_act,_,_ = eoo_binary_attribute(global_model, testset,protected_index)
                
                accuracy.append(acc)
                statistical_parity_difference .append(spd_act)
                equalized_odds.append(eod_act)
                
                
                
                print(f'Round Number {r}')
                print(f'accuracy {acc}: SPD: {spd_act :.4f} EOD: {eod_act:.4f} Cost: {per_round_cost: .4f}')

    return accuracy,statistical_parity_difference,equalized_odds,cost_w_iter


def run_FairFed(method,model,client_distribution,dataset,protected_index,params,fmetric = 'spd'):
    
    ## Loading the client models and global model
    device = 'cpu'
    global_model = model
    client_models = [ model.to(device) for _ in range(params["num_sel"])] 
    for model in client_models:    
        model.load_state_dict(global_model.state_dict())
    

    communication_cost = {'Uplink':0, 'Downlink': 0}


    
    train_datasets,trainset,testset = utils.create_local_datasets(dataset,client_distribution,params["num_sel"],protected_index)
    utils.print_client_gender_distribution(train_datasets, protected_index)

    weights,weights_norm = utils.calculate_weights(train_datasets)
    client_weights = weights_norm
    Weights = {}
    Local_Acc = {}
    Local_Fair = {}
    Local_Gap = {}
    
    for i in range(params['total_clients']):
        Weights["Client " + str(i)] = weights_norm[i]
    
    per_round_cost = 0 
    cost_w_iter = []
    accuracy = []
    statistical_parity_difference =[]
    equalized_odds = []
    
    for r in range(params["num_rounds"]):

        flag = 'one'
        for i in tqdm(range(params['total_clients'])):
            client_models[i]= k_util.client_update_fedfair(client_models[i],Weights,Local_Acc,Local_Fair,Local_Gap,i,flag,trainset,train_datasets[i],protected_index,None,None,None,params,fmetric)
            client_models[i].eval()
            communication_cost['Downlink']= communication_cost['Downlink'] +  utils.parameter_count(client_models[i]) 
            communication_cost['Uplink']= communication_cost['Uplink'] + len(Local_Acc)
        
        F_global = sum([client_weights[i] * Local_Fair["Client " + str(i)] for i in range(params['total_clients'])])
        Acc_global = sum([client_weights[i] * Local_Acc["Client " + str(i)] for i in range(params['total_clients'])])
            
        flag = 'two'
        for i in tqdm(range(params['total_clients'])):
            client_models[i]= k_util.client_update_fedfair(client_models[i],Weights,Local_Acc,Local_Fair,Local_Gap,i,flag,trainset,train_datasets[i],protected_index,Acc_global,F_global,None,params,fmetric)
            client_models[i].eval()
            communication_cost['Downlink']= communication_cost['Downlink'] + 2 # len(Acc_global) + len(F_global)
            communication_cost['Uplink']= communication_cost['Uplink'] + len(Local_Gap)
        
        
        global_metric_gap = sum(Local_Gap.values())/params['total_clients']
            
        flag = 'three'
        for i in tqdm(range(params['total_clients'])):
            client_models[i]= k_util.client_update_fedfair(client_models[i],Weights,Local_Acc,Local_Fair,Local_Gap,i,flag,trainset,train_datasets[i],protected_index,Acc_global,F_global,global_metric_gap,params,fmetric)
            client_models[i].eval()
            communication_cost['Downlink']= communication_cost['Downlink'] +  len(Weights) + 1#len(global_metric_gap)
            communication_cost['Uplink']= communication_cost['Uplink'] + len(Local_Gap)
    
        weights_sum = sum( Weights.values())
        
        
        for i in range(params['total_clients']):
            Weights["Client " + str(i)] = Weights["Client " + str(i)] / weights_sum
        
        weights_norm = list(Weights.values())
        #print("Weights:", Weights)
        
        client_models = k_util.server_aggregate_FAVG(global_model,client_models,weights_norm,params)
        per_round_cost = utils.convert_to_megabytes(sum(communication_cost.values()) * 32)

        if(r%1==0):
            with torch.no_grad():
                cost_w_iter.append(per_round_cost) 
                modified_testset = utils.drop_attribute(testset,protected_index,weighted=False)
                
                acc = model_accuracy(global_model, modified_testset, binary = True)
                spd_act,_,_ = spd(global_model, testset,protected_index)
                eod_act,_,_ = eoo_binary_attribute(global_model, testset,protected_index)
                
                accuracy.append(acc)
                statistical_parity_difference .append(spd_act)
                equalized_odds.append(eod_act)
                
                
                print(f'Weights {Weights}')
                print(f'Round Number {r}')
                print(f'accuracy {acc}: SPD: {spd_act :.4f} EOD: {eod_act:.4f} Cost: {per_round_cost: .4f}')

    return accuracy,statistical_parity_difference,equalized_odds,cost_w_iter
    






def run_Centralized(method,model,dataset,protected_index,params):
    train_datasets,trainset,testset = utils.create_local_datasets(dataset,'IID',1,protected_index)
    optim_init_func = utils.create_optimizer_init_func(torch.optim.Adam)
    optimizer = optim_init_func(model)
    b = params['batch_size']
    loader_init_func = utils.create_dataloader_init_func({"batch_size": b})
    trainset_1 = utils.drop_attribute(trainset, protected_index, weighted=False)
    loader = torch.utils.data.DataLoader(trainset, batch_size=len(trainset), shuffle=False)
    generator = torch.Generator()
    device = 'cpu'
    loss_func = torch.nn.BCELoss()
    loss = 0

    accuracy = []
    statistical_parity_difference =[]
    equalized_odds = []

    for epoch in range(params['epochs']):
        for inputs, targets in loader:
            sensitive_attributes = (inputs[:, protected_index])[:, None]
            inputs = utils.drop_attribute_tensor(inputs,protected_index)
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            if outputs.dim() == 1:
                outputs = outputs.unsqueeze(1)
                    
            sigma_x = 1.0  # Sigma for Kx
            sigma_s = 1.0  # Sigma for Ks
            Kx = rbf_kernel(outputs, outputs, sigma_x)
            Ks = rbf_kernel(sensitive_attributes, sensitive_attributes, sigma_s)
                    
            #import pdb;pdb.set_trace()
            # Calculate H using the provided formula
            n = len(outputs)
            I = torch.eye(n)
            H = I - (1/n) * (I @ I.T)

            # Compute intermediate quantities
            H_squared = H @ H
            H_Kx_H = H_squared @ Kx

            # Compute the quantity Tr(H * Ks * H^2 * Kx * H)
            fairloss = torch.trace(H @ Ks @ H_Kx_H @H)
                
            loss = loss_func(outputs, targets) +  params["fairness_weight"] * fairloss
                    
            loss.backward()
            optimizer.step()

        if(epoch%10==0):

            with torch.no_grad():
                modified_testset = utils.drop_attribute(testset,protected_index,weighted=False)
                
                acc = model_accuracy(model, modified_testset, binary = True)
                spd_act,_,_= spd(model, testset,protected_index)
                eod_act,_,_ = eoo_binary_attribute(model, testset,protected_index)
                
                accuracy.append(acc)
                statistical_parity_difference .append(spd_act)
                equalized_odds.append(eod_act)

                print(f'Epoch Number {epoch}')
                print(f'accuracy {acc}: SPD: {spd_act :.4f} EOD: {eod_act:.4f}')

    return accuracy,statistical_parity_difference,equalized_odds




##### The new kernel one ################





def run_FairFed_kernel(method,model,client_distribution,dataset,protected_index,params,fmetric = 'spd'):
    
    ## Loading the client models and global model
    device = 'cpu'
    global_model = model
    client_models = [ model.to(device) for _ in range(params["num_sel"])] 
    for model in client_models:    
        model.load_state_dict(global_model.state_dict())
    

    communication_cost = {'Uplink':0, 'Downlink': 0}


    
    train_datasets,trainset,testset = utils.create_local_datasets(dataset,client_distribution,params["num_sel"],protected_index)
    weights,weights_norm = utils.calculate_weights(train_datasets)
    client_weights = weights_norm
    Weights = {}
    Local_Acc = {}
    Local_Fair = {}
    Local_Gap = {}
    
    for i in range(params['total_clients']):
        Weights["Client " + str(i)] = weights_norm[i]
    
    per_round_cost = 0 
    cost_w_iter = []
    accuracy = []
    statistical_parity_difference =[]
    equalized_odds = []
    
    for r in range(params["num_rounds"]):

        flag = 'one'
        for i in tqdm(range(params['total_clients'])):
            client_models[i]= k_util.client_update_fedfair_kernel(client_models[i],Weights,Local_Acc,Local_Fair,Local_Gap,i,flag,trainset,train_datasets[i],protected_index,None,None,None,params,fmetric)
            client_models[i].eval()
            communication_cost['Downlink']= communication_cost['Downlink'] +  utils.parameter_count(client_models[i]) 
            communication_cost['Uplink']= communication_cost['Uplink'] + len(Local_Acc)
        
        F_global = sum([client_weights[i] * Local_Fair["Client " + str(i)] for i in range(params['total_clients'])])
        Acc_global = sum([client_weights[i] * Local_Acc["Client " + str(i)] for i in range(params['total_clients'])])
            
        flag = 'two'
        for i in tqdm(range(params['total_clients'])):
            client_models[i]= k_util.client_update_fedfair_kernel(client_models[i],Weights,Local_Acc,Local_Fair,Local_Gap,i,flag,trainset,train_datasets[i],protected_index,Acc_global,F_global,None,params,fmetric)
            client_models[i].eval()
            communication_cost['Downlink']= communication_cost['Downlink'] + 2 # len(Acc_global) + len(F_global)
            communication_cost['Uplink']= communication_cost['Uplink'] + len(Local_Gap)
        
        
        global_metric_gap = sum(Local_Gap.values())/params['total_clients']
            
        flag = 'three'
        for i in tqdm(range(params['total_clients'])):
            client_models[i]= k_util.client_update_fedfair_kernel(client_models[i],Weights,Local_Acc,Local_Fair,Local_Gap,i,flag,trainset,train_datasets[i],protected_index,Acc_global,F_global,global_metric_gap,params,fmetric)
            client_models[i].eval()
            communication_cost['Downlink']= communication_cost['Downlink'] +  len(Weights) + 1#len(global_metric_gap)
            communication_cost['Uplink']= communication_cost['Uplink'] + len(Local_Gap)
    
        weights_sum = sum( Weights.values())
        
        
        for i in range(params['total_clients']):
            Weights["Client " + str(i)] = Weights["Client " + str(i)] / weights_sum
        
        weights_norm = list(Weights.values())
        #print("Weights:", Weights)
        
        client_models = k_util.server_aggregate_FAVG(global_model,client_models,weights_norm,params)
        per_round_cost = utils.convert_to_megabytes(sum(communication_cost.values()) * 32)

        if(r%1==0):
            with torch.no_grad():
                cost_w_iter.append(per_round_cost) 
                modified_testset = utils.drop_attribute(testset,protected_index,weighted=False)
                
                acc = model_accuracy(global_model, modified_testset, binary = True)
                spd_act,_,_ = spd(global_model, testset,protected_index)
                eod_act,_,_ = eoo_binary_attribute(global_model, testset,protected_index)
                
                accuracy.append(acc)
                statistical_parity_difference .append(spd_act)
                equalized_odds.append(eod_act)
                
                
                print(f'Weights {Weights}')
                print(f'Round Number {r}')
                print(f'accuracy {acc}: SPD: {spd_act :.4f} EOD: {eod_act:.4f} Cost: {per_round_cost: .4f}')

    return accuracy,statistical_parity_difference,equalized_odds,cost_w_iter
    