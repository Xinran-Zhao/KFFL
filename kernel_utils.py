import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import random_split
from torch.nn import Module
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import Optimizer
from torch.nn import Module
import numpy as np
import copy
from typing import Callable
from torch.utils.data import Dataset, DataLoader, TensorDataset
#from src.mitigating_bias_in_fl_algos.weighted_dataset import WeightedDataset
from torch.utils.data import DataLoader, random_split 
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import random_split
from torch.nn import Module
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import Optimizer
from torch.nn import Module
import numpy as np
import copy
from typing import Callable
from torch.utils.data import Dataset, DataLoader, TensorDataset
#from src.mitigating_bias_in_fl_algos.weighted_dataset import WeightedDataset
from torch.utils.data import DataLoader, random_split 
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics.pairwise import rbf_kernel
try:
    from pyrfm import OrthogonalRandomFeature,CompactRandomFeature,RandomFourier,FastFood
except ImportError:
    OrthogonalRandomFeature = CompactRandomFeature = RandomFourier = FastFood = None
from tqdm import tqdm as _tqdm
from functools import partial
tqdm = partial(_tqdm, disable=True)
import utilites as utils
from eval_metrics import spd,model_accuracy,eoo_binary_attribute
from utilites import fairbatch_dataset
from FairBatchSampler import FairBatch, CustomDataset
import torch.nn.functional as F




class Kernel_Mapping(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, input):

        #import pdb; pdb.set_trace()

        ## Try using torch.vmap to carry out this mapping operation ....
        
        random_state = ctx.random_state if hasattr(ctx, 'random_state') else 0
        bandwidth = ctx.random_state if hasattr(ctx, 'bandwidth') else 10
        
        transformer = OrthogonalRandomFeature(n_components=bandwidth, gamma=0.001, 
                                              random_state= random_state)
        # transformer = CompactRandomFeature(n_components=10**4, 
        #                                       random_state=0)
        
#         transformer = RandomFourier(n_components=1024,
#                                     kernel='rbf',
#                                     use_offset=True, random_state=0)
        
#         transformer = FastFood(n_components=1024, 
#                                gamma=0.001, 
#                                distribution='gaussian', 
#                                random_fourier=True, 
#                                use_offset=False, 
#                                random_state=0)
        
        inputs = input.detach().numpy()
        inputs_trans = transformer.fit_transform(inputs)
        ## Do I want to save the transformed for backward?
        ctx.save_for_backward(input)
        return torch.as_tensor(inputs_trans, dtype=input.dtype)

    @staticmethod
    def backward(ctx,g):
        return g
    
def khsic_local_contribution(model : Module, dataset : Dataset, sensitive_attribute_idx : int,params,random_seed = 0):
    
    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    device = 'cpu'
    
    for inputs, targets in loader:
        
        sensitive_attributes = (inputs[:, sensitive_attribute_idx])[:, None]
        inputs = utils.drop_attribute_tensor(inputs,sensitive_attribute_idx)
        
        inputs = inputs.to(device)
        targets = targets.to(device)
        outputs = model(inputs)
        
        if outputs.dim() == 1:
           outputs = outputs.unsqueeze(1)
         

       # outputs = outputs.unsqueeze(1)
       
       
        random_state = random_seed  # Your desired random state
        bandwidth = 100 #  params['D']

        #print(f'The seed is {random_seed}')
        
        Z_outputs_local= Kernel_Mapping.apply(outputs)
        Z_sens_attr_local  = Kernel_Mapping.apply(sensitive_attributes)
        
        
        
        local_sens_mean = torch.mean(Z_sens_attr_local ,dim = 0)
        local_output_mean = torch.mean(Z_outputs_local,dim = 0)
        
        local_interaction = Z_sens_attr_local.T @ Z_outputs_local

        #R = params['R']
        #T = params['T']
        
        #phi_hat = torch.normal(0., 1.0/R, (len(dataset), R), generator=generator.manual_seed(random_seed))
        #omega_hat = torch.normal(0., 1.0/T, (len(dataset), T), generator=generator.manual_seed(int(random_seed/2)))

        #phi_sens_attr = phi_hat.T@Z_sens_attr
        #phi_outputs = phi_hat.T@Z_outputs
        #omega_sens_attr = omega_hat.T@Z_sens_attr
        #omega_outputs = omega_hat.T@Z_outputs

    return {"local_interaction" : local_interaction,"mu_f":  local_output_mean,"mu_s": local_sens_mean}

def server_aggregate(global_model,client_models,client_local_information,weights_norm,weights,params,round = 'Second',fair_grad = None):
    """
    This function has aggregation method 'mean'
    """
    rt_obj = {}
    if(round == 'First'):
        
        D = params['D']
        global_fairness = torch.zeros((D,D), dtype=torch.float32)
        global_fair_mean = torch.zeros((D,1), dtype=torch.float32)
        global_model_mean = torch.zeros((D,1), dtype=torch.float32)

        ### Communication between clients and server to compute A and B matrices
        for i, local_contributions in enumerate(client_local_information):
                ## I am sending the same seed to all the workers
                #print(i)
                mu_f_scaled = local_contributions["mu_f"] * weights_norm[i]
                mu_s_scaled = local_contributions["mu_s"] * weights[i]
                # global_fairness += ( local_contributions["local_interaction"] -  weights[i] * local_contributions["mu_s"] @ weights_norm[i] * local_contributions["mu_s"].T)
                global_fairness += local_contributions["local_interaction"] - torch.matmul(mu_f_scaled.unsqueeze(1), mu_s_scaled.unsqueeze(1).T)
                #import pdb;pdb.set_trace()
                global_fair_mean += weights_norm[i] * local_contributions["mu_s"].unsqueeze(1)
                global_model_mean +=  weights_norm[i] * local_contributions["mu_f"].unsqueeze(1)



        rt_obj = {'global_fairness':global_fairness,'global_fair_mean': global_fair_mean, 'global_model_mean': global_model_mean }
 

    elif(round == 'Second'):
        for i in range(len(client_models)):
            fair_grad[i] = utils.scale_gradients(fair_grad[i],weights_norm[i])
        
        # Initialize an accumulator dictionary to store the accumulated gradients
        accumulated_gradients = {}
        
        # Iterate through the fair_grad dictionary (gradients from different clients)
        for i, gradients in enumerate(fair_grad):
            # Iterate through the gradients of each client
            for param_name, gradient in gradients.items():
                if param_name in accumulated_gradients:
                    # If the parameter name is already in the accumulator, add the gradient
                    accumulated_gradients[param_name] += gradient
                else:
                    # If the parameter name is not in the accumulator, initialize it with the gradient
                    accumulated_gradients[param_name] = gradient.clone()
        
        # Divide the accumulated gradients by the number of clients to get the average gradients
        #num_clients = len(fair_grad)
       # average_gradients = {param_name: gradient / num_clients for param_name, gradient in accumulated_gradients.items()}
       # import pdb;pdb.set_trace()
        for param_name, param in global_model.named_parameters():
            if param_name in accumulated_gradients:
                reg = params['fairness_weight']/( (sum(weights)- 1)**2)
                param.data = param.data - 2 * params['step_size'] *reg*accumulated_gradients[param_name]
        
        for model in client_models:
            model.load_state_dict(global_model.state_dict())

        rt_obj = {'client_model':client_models} 

    else:     
        for i in range(len(client_models)):
            client_models[i] = utils.scale_model(client_models[i],weights_norm[i])
        
        global_dict = global_model.state_dict()
        for k in global_dict.keys():
            global_dict[k] = torch.stack([model.state_dict()[k] for i,model in enumerate(client_models)], 0)
            global_dict[k] = global_dict[k].sum(0)
            global_model.load_state_dict(global_dict)

        global_model.load_state_dict(global_dict)
        for model in client_models:
            model.load_state_dict(global_model.state_dict())
        #print("Round 1")
        rt_obj = {'client_model':client_models}   

  
    return rt_obj

def server_aggregate_TD(global_model,client_models,client_local_information,weights_norm,weights,params,round = 'Second',fair_grad = None,epoch = 1):
    """
    This function has aggregation method 'mean'

    """
    rt_obj = {}
    global_round = epoch
    if(round == 'First'):
        
        D = params['D']
        global_fairness = torch.zeros((D,D), dtype=torch.float32)
        global_fair_mean = torch.zeros((D,1), dtype=torch.float32)
        global_model_mean = torch.zeros((D,1), dtype=torch.float32)

        ### Communication between clients and server to compute A and B matrices
        for i, local_contributions in enumerate(client_local_information):
                ## I am sending the same seed to all the workers
                #print(i)
                local_contributions= local_contributions['local_contirbution']
                mu_f_scaled = local_contributions["mu_f"] * weights_norm[i]
                mu_s_scaled = local_contributions["mu_s"] * weights[i]
                # global_fairness += ( local_contributions["local_interaction"] -  weights[i] * local_contributions["mu_s"] @ weights_norm[i] * local_contributions["mu_s"].T)
                global_fairness += local_contributions["local_interaction"] - torch.matmul(mu_f_scaled.unsqueeze(1), mu_s_scaled.unsqueeze(1).T)
                #import pdb;pdb.set_trace()
                global_fair_mean += weights_norm[i] * local_contributions["mu_s"].unsqueeze(1)
                global_model_mean +=  weights_norm[i] * local_contributions["mu_f"].unsqueeze(1)


        #import pdb;pdb.set_trace()
        rt_obj = {'global_fairness':global_fairness,'global_fair_mean': global_fair_mean, 'global_model_mean': global_model_mean }
        
        if(global_round > 0): 
            
            for i in range(len(client_models)):
                fair_grad[i] = utils.scale_gradients(fair_grad[i],weights_norm[i])
            
            # Initialize an accumulator dictionary to store the accumulated gradients
            accumulated_gradients = {}
            
            # Iterate through the fair_grad dictionary (gradients from different clients)
            for i, gradients in enumerate(fair_grad):
                # Iterate through the gradients of each client
                for param_name, gradient in gradients.items():
                    if param_name in accumulated_gradients:
                        # If the parameter name is already in the accumulator, add the gradient
                        accumulated_gradients[param_name] += gradient
                    else:
                        # If the parameter name is not in the accumulator, initialize it with the gradient
                        accumulated_gradients[param_name] = gradient.clone()
            
            # Divide the accumulated gradients by the number of clients to get the average gradients
            #num_clients = len(fair_grad)
           # average_gradients = {param_name: gradient / num_clients for param_name, gradient in accumulated_gradients.items()}
           # import pdb;pdb.set_trace()
            for param_name, param in global_model.named_parameters():
                if param_name in accumulated_gradients:
                    reg = params['fairness_weight']/( (sum(weights)- 1)**2)
                    param.data = param.data - 2 * params['step_size'] *reg*accumulated_gradients[param_name]
            
            for model in client_models:
                model.load_state_dict(global_model.state_dict())
    
            rt_obj['client_model'] = client_models
        #rt_obj = {'client_model':client_models} 

    else:     
        for i in range(len(client_models)):
            client_models[i] = utils.scale_model(client_models[i],weights_norm[i])
        
        global_dict = global_model.state_dict()
        for k in global_dict.keys():
            global_dict[k] = torch.stack([model.state_dict()[k] for i,model in enumerate(client_models)], 0)
            global_dict[k] = global_dict[k].sum(0)
            global_model.load_state_dict(global_dict)

        global_model.load_state_dict(global_dict)
        for model in client_models:
            model.load_state_dict(global_model.state_dict())
        #print("Round 1")
        rt_obj = {'client_model':client_models}   

  
    return rt_obj



def client_update_TD(client_model,global_model,train_dataset,protected_index,random_seed = None,round = 'Second',global_fairness_term= None, params= None,weights = None,epoch = 1):
    
    step_size = params['step_size']
    global_round  = epoch
    
    client_model.train()
    optim_init_func = utils.create_optimizer_init_func(torch.optim.Adam)
    optimizer = optim_init_func(client_model)
    
    train_dataset_1 = utils.drop_attribute(train_dataset,protected_index,weighted=False)
    
    batch = params['batch_size']
    
    loader_init_func = utils.create_dataloader_init_func({"batch_size" : batch})
    loader = loader_init_func(train_dataset_1)
    device = 'cpu'
    
    loss_func = torch.nn.BCELoss()
    loss = 0
    rt_obj = {}

    ### Clients Compute Local Contribution 
    # "First" Corresponds to Fair Flag
    if(round == 'First'):
        
        local_contribution = khsic_local_contribution(client_model, train_dataset,protected_index,params,random_seed)  
        rt_obj = {'local_contirbution':local_contribution}        

        #local_interaction = local_contributions["local_interaction"] - weights *  torch.matmul(global_fairness_term["global_fair_mean"].detac,global_fairness_term["global_model_mean"].T)
        if(global_round > 0):
            #Q1) Is the error happening due to the presence of global fair mean that depends on other models? Ans) Yes, so we need to detach them?
            global_fair_mean = global_fairness_term["global_fair_mean"].detach()
            global_model_mean = global_fairness_term["global_model_mean"].detach()
            #import pdb;pdb.set_trace()
            #local_contribution = khsic_local_contribution(client_model, train_dataset,protected_index,params,random_seed) 
            local_interaction =  local_contribution['local_interaction']  - weights *  torch.matmul(global_fair_mean,global_model_mean.T)
            
            global_interaction = global_fairness_term['global_fairness'] ### G(W)
        
            ## Remove all the old gradients
            
            for param in client_model.parameters():
                if param.grad is not None:
                    param.grad.zero_()
            
            ### Calculate new gradients on the fairness term
            #local_interaction.backward(gradient =  global_interaction)
            #import pdb;pdb.set_trace()
            local_interaction.backward(gradient =  global_interaction.detach())
            # for j in tqdm(range(params['local_epochs'])):
            #     for inputs, targets in loader:
            #         inputs = inputs.to(device)
            #         targets = targets.to(device)
            #         optimizer.zero_grad()
            #         outputs = client_model(inputs)
            #         if outputs.dim() > targets.dim():
            #             outputs = outputs.squeeze(1)
            #         elif targets.dim() > outputs.dim():
            #              targets= targets.squeeze(1)
            #         loss = loss_func(outputs, targets)
                    
                    
            #         inner_arg_sens = matrices['phi_s']@ matrices['omega_s'].T
            #         inner_arg_output = matrices['omega_f'] @ matrices['phi_f'].T
            #         inner_arg = inner_arg_sens @ inner_arg_output
            #         fair_loss = torch.trace(inner_arg)
            #         fair_loss.backward(retain_graph=True)
                    

            #         # Save gradients
            model_gradients = {}
            for name, param in client_model.named_parameters():
                if param.requires_grad and param.grad is not None: 
                    model_gradients[name] = param.grad.clone()
                    
                
            rt_obj['fair_grad'] =   model_gradients  
                #optimizer.step()
        
    elif(round == 'Second'): # Corresponds to the Fair Round
        
        for j in tqdm(range(params['local_epochs'])):
            for inputs, targets in loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                optimizer.zero_grad()
                outputs = client_model(inputs) 
                if outputs.dim() > targets.dim():
                    outputs = outputs.squeeze(1)
                elif targets.dim() > outputs.dim():
                    targets= targets.squeeze(1)
                    
                loss = loss_func(outputs, targets)
                loss = loss   + (1.0)/(2.0*step_size) * (utils.model_l2(utils.subtract_models(client_model, global_model)))
                loss.backward()
                #loss.backward(retain_graph = True)
                optimizer.step()
        
        
        rt_obj = {'client_model':client_model}
    return rt_obj



def client_update(client_model,global_model,train_dataset,protected_index,random_seed = None,round = 'Second',global_fairness_term= None, params= None,weights = None,epoch = 1):
    
    step_size = params['step_size']
    
    client_model.train()
    optim_init_func = utils.create_optimizer_init_func(torch.optim.Adam)
    optimizer = optim_init_func(client_model)
    
    train_dataset_1 = utils.drop_attribute(train_dataset,protected_index,weighted=False)
    
    batch = params['batch_size']
    
    loader_init_func = utils.create_dataloader_init_func({"batch_size" : batch})
    loader = loader_init_func(train_dataset_1)
    device = 'cpu'
    
    loss_func = torch.nn.BCELoss()
    loss = 0
    rt_obj = {}

    ### Clients Compute Local Contribution 
    if(round == 'First'):
        local_contribution = khsic_local_contribution(client_model, train_dataset,protected_index,params,random_seed)  
        rt_obj = {'local_contirbution':local_contribution}        
        

       
    elif(round == 'Second'):

        # rt_obj = {'global_fairness':global_fairness,'global_fair_mean': global_fair_mean, 'global_model_mean': global_model_mean }
        #import pdb;pdb.set_trace()
        local_contributions = khsic_local_contribution(client_model, train_dataset,protected_index,params,random_seed)

        #Q1) Is the error happening due to the presence of global fair mean that depends on other models? Ans) Yes, so we need to detach them?
        global_fair_mean = global_fairness_term["global_fair_mean"].detach()
        global_model_mean = global_fairness_term["global_model_mean"].detach()

        local_interaction = local_contributions["local_interaction"] - weights *  torch.matmul(global_fair_mean,global_model_mean.T)
        
        #local_interaction = local_contributions["local_interaction"] - weights *  torch.matmul(global_fairness_term["global_fair_mean"].detac,global_fairness_term["global_model_mean"].T)
        
        global_interaction = global_fairness_term['global_fairness'] ### G(W)
        
        ## Remove all the old gradients
        
        for param in client_model.parameters():
            if param.grad is not None:
                param.grad.zero_()
        
        ### Calculate new gradients on the fairness term
        #local_interaction.backward(gradient =  global_interaction)
        #import pdb;pdb.set_trace()
        local_interaction.backward(gradient =  global_interaction.detach())
        # for j in tqdm(range(params['local_epochs'])):
        #     for inputs, targets in loader:
        #         inputs = inputs.to(device)
        #         targets = targets.to(device)
        #         optimizer.zero_grad()
        #         outputs = client_model(inputs)
        #         if outputs.dim() > targets.dim():
        #             outputs = outputs.squeeze(1)
        #         elif targets.dim() > outputs.dim():
        #              targets= targets.squeeze(1)
        #         loss = loss_func(outputs, targets)
                
                
        #         inner_arg_sens = matrices['phi_s']@ matrices['omega_s'].T
        #         inner_arg_output = matrices['omega_f'] @ matrices['phi_f'].T
        #         inner_arg = inner_arg_sens @ inner_arg_output
        #         fair_loss = torch.trace(inner_arg)
        #         fair_loss.backward(retain_graph=True)
                

        #         # Save gradients
        model_gradients = {}
        for name, param in client_model.named_parameters():
            if param.requires_grad and param.grad is not None: 
                model_gradients[name] = param.grad.clone()
                
                
                
                #optimizer.step()
        rt_obj = {'fair_grad': model_gradients,'client_model':client_model}
        
    elif(round == 'Third'):
        
        for j in tqdm(range(params['local_epochs'])):
            for inputs, targets in loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                optimizer.zero_grad()
                outputs = client_model(inputs) 
                if outputs.dim() > targets.dim():
                    outputs = outputs.squeeze(1)
                elif targets.dim() > outputs.dim():
                    targets= targets.squeeze(1)
                    
                loss = loss_func(outputs, targets)
                loss = loss   + (1.0)/(2.0*step_size) * (utils.model_l2(utils.subtract_models(client_model, global_model)))
                loss.backward()
                #loss.backward(retain_graph = True)
                optimizer.step()
        
        
        rt_obj = {'client_model':client_model}
    return rt_obj






def server_aggregate_FAVG(global_model,client_models,weights_norm,params):
    """
    This function has aggregation method 'mean'
    """


    for i in range(len(client_models)):
        client_models[i] = utils.scale_model(client_models[i],weights_norm[i])
    
    global_dict = global_model.state_dict()
    for k in global_dict.keys():
        global_dict[k] = torch.stack([model.state_dict()[k] for i,model in enumerate(client_models)], 0)
        global_dict[k] = global_dict[k].sum(0)
        global_model.load_state_dict(global_dict)

    global_model.load_state_dict(global_dict)
    for model in client_models:
        model.load_state_dict(global_model.state_dict())
  

    
    return client_models






def client_update_FAVG(client_model,global_model,train_dataset,protected_index,params= None,epoch = 1):
    
    #step_size = params['step_size']
    
    client_model.train()
    optim_init_func = utils.create_optimizer_init_func(torch.optim.Adam)
    optimizer = optim_init_func(client_model)
    
    train_dataset_1 = utils.drop_attribute(train_dataset,protected_index,weighted=False)
    
    batch = params['batch_size']
    
    loader_init_func = utils.create_dataloader_init_func({"batch_size" : batch})
    loader = loader_init_func(train_dataset_1)
    device = 'cpu'

    
    loss_func = torch.nn.BCELoss()
    loss = 0
 
    for j in tqdm(range(params['local_epochs'])):
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = client_model(inputs)
            outputs = outputs.view(-1)  # align [B,1] → [B] to match targets [B]
            loss = loss_func(outputs, targets)

            loss.backward()
            optimizer.step()
        
        
    
    return client_model






def client_update_minmax(client_model,train_dataset,w,group_index=40):
    
    client_model.train()
    #step_size = params['step_size']
    optim_init_func = utils.create_optimizer_init_func(torch.optim.Adam)
    optimizer = optim_init_func(client_model)
    loss_func = torch.nn.BCELoss(reduction = 'mean')
    group_datasets,lengths = utils.split_dataset_group(train_dataset,group_index)
    ## Create loaders based on the datasets
    loaders = {}
    for i,(keys,grp) in enumerate(group_datasets.items()):
        len_grp = lengths[str(i)]
        grp_1 = utils.drop_attribute(grp,group_index,weighted=False)
        loader_init_func = utils.create_dataloader_init_func({"batch_size" : len_grp})
        
        #print(f'grp shape {type(grp)}')
        loaders[i] = loader_init_func(grp_1)
    
  
    total_samples = sum(lengths.values())
    
    
    Group_wise_loss = {}
    scaled_group_loss = {}
    group_loss = 0
    device = 'cpu'
    for j in tqdm(range(len(loaders))):
        loss = 0
        for inputs, targets in loaders[j]:
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            #print(f'Inputs shape{inputs.shape}')
            outputs = client_model(inputs)
            outputs = outputs.view(-1)  # align [B,1] → [B]
            loss = loss_func(outputs, targets)
            
        ## Do check what the loss function gives you as output sum or the mean?
        
        Group_wise_loss[j] = loss.item()
        scaled_group_loss[i] = loss.item() * lengths[str(j)]
        group_loss += w[j]*(lengths[str(j)]/total_samples)*loss
    
    group_loss.backward()
    optimizer.step()
    
     
    return client_model,Group_wise_loss,scaled_group_loss

def server_aggregate_minmax(global_model, client_models,weights_norm,all_client_risks,u,group_stats,params = None):
    """
    This function has aggregation method 'mean'
    """

    #########
    for i in range(len(client_models)):
        client_models[i] = utils.scale_model(client_models[i],weights_norm[i])
    global_dict = global_model.state_dict()
    for k in global_dict.keys(): 
       global_dict[k] = torch.stack([model.state_dict()[k] for i,model in enumerate(client_models)], 0)
       global_dict[k] = global_dict[k].sum(0)
       global_model.load_state_dict(global_dict)
    
    global_model.load_state_dict(global_dict)
    for model in client_models:
        model.load_state_dict(global_model.state_dict())
    
    ## Calculate the global group risk ####
    
    r_a = utils.global_risk(all_client_risks,group_stats)
    
   # params["global_adversary_rate"] = 0.1
    
    #print(f"Shape of u is {} and Shape of r_a is".format(shape(u),shape(r_a))
    u = u +  params["global_adversary_rate"] * (r_a)
    
    # Doing the projection onto the simplex
    u = utils.euclidean_proj_simplex(u, s=1)
    
    return client_models,u


def client_update_fedfair(client_model, Weights, Local_Acc, Local_Fair, Local_Gap, client_num=1, flag='one', trainset_global=None, trainset=None, protected_attribute_index=40, Acc_global=None, F_global=None, global_metric_gap=None, params=None,fmetric ='spd'):
    # step_size = params['step_size']
    client_model.train()
    fairbatch = True
    device = 'cpu'
    #seed = 0
    if flag == 'one':
        ############## Local training / Debiasing ##################
        lr = params.get('step_size', 0.01)
        fb_lr = params.get('fb_lr', 0.005)
        optim_init_func = utils.create_optimizer_init_func(
            torch.optim.Adam, {'lr': lr}
        )
        optimizer = optim_init_func(client_model)
        b = params['batch_size']
        loader_init_func = utils.create_dataloader_init_func({"batch_size": b})
        trainset_1 = utils.drop_attribute(trainset, protected_attribute_index, weighted=False)
        
        if (fairbatch):
            xz_train, z_train, y_train = fairbatch_dataset(trainset, protected_attribute_index)
            train_data = CustomDataset(xz_train, y_train, z_train)
            if(fmetric == 'spd'): 
                sampler = FairBatch(client_model, train_data.x, train_data.y, train_data.z, batch_size=b, alpha=fb_lr, target_fairness='dp', replacement=False)
            elif(fmetric =='eod'):
                sampler = FairBatch(client_model, train_data.x, train_data.y, train_data.z, batch_size=b, alpha=fb_lr, target_fairness='eqodds', replacement=False)     
            else:
                pass
            loader = torch.utils.data.DataLoader(train_data, sampler=sampler, num_workers=0)
            loss_func = torch.nn.BCELoss()

            loss = 0

            for j in tqdm(range(params['local_epochs'])): 
                for batch_idx, (inputs, targets, z) in enumerate (loader):
                    inputs = inputs.to(device)
                    targets = targets.to(device)
                    optimizer.zero_grad()
                    outputs = client_model(inputs)
                    targets_01 = (targets.squeeze() + 1) / 2  # FairBatch uses {-1,1}; BCELoss needs [0,1]
                    loss = loss_func((F.tanh(outputs.squeeze()) + 1) / 2, targets_01)
                    loss.backward()
                    optimizer.step()

        else:
            
            loader = loader_init_func(trainset_1)
            loss_func = torch.nn.BCELoss()
            loss = 0
            for j in tqdm(range(params['local_epochs'])):
                for inputs, targets in loader:
                    inputs = inputs.to(device)
                    targets = targets.to(device)
                    optimizer.zero_grad()
                    outputs = client_model(inputs)
                    outputs = outputs.view(-1)  # align [B,1] → [B]
                    loss = loss_func(outputs, targets)
                    loss.backward()
                    optimizer.step()

        ##########################################
        groups_stats, prob_vector = utils.get_stats(trainset_global, protected_attribute_index)
        if(fmetric == 'spd'): 
            local_spd, g1_rate, g0_rate = spd(client_model, trainset, protected_attribute_index)
        elif(fmetric =='eod'):
            print('here')
            local_eod, g1_rate, g0_rate = eoo_binary_attribute(client_model, trainset, protected_attribute_index)
        else:
            pass
        prob_A_0 = prob_vector['A0 Y1'] + prob_vector['A0 Y0']
        prob_A_1 = prob_vector['A1 Y1'] + prob_vector['A1 Y0']
        
        if(fmetric == 'spd'):
            local_spd_view = g0_rate / prob_A_0 - g1_rate / prob_A_1
            Local_Fair["Client " + str(client_num)] = local_spd_view
        elif(fmetric =='eod'):
            local_eod_view = g0_rate / prob_A_0 - g1_rate / prob_A_1
            Local_Fair["Client " + str(client_num)] = local_eod_view
            

        # local_eod_view = g1_rate/prob_vector['A1 Y1']  - g1_rate/prob_vector['A0 Y1']
        Local_Acc["Client " + str(client_num)] = model_accuracy(client_model, trainset_1)
       

    elif flag == 'two':
        if Local_Fair["Client " + str(client_num)] is None:
            Local_Gap["Client " + str(client_num)] = abs(Local_Acc["Client " + str(client_num)] - Acc_global)
        else:
            Local_Gap["Client " + str(client_num)] = abs(Local_Fair["Client " + str(client_num)] - F_global)

    elif flag == 'three':
        # TODO: Use the beta (fairness penalty weight) value to update the weights
        Weights["Client " + str(client_num)] = Weights["Client " + str(client_num)] - (params['beta']) * (Local_Gap["Client " + str(client_num)] - global_metric_gap)

    return client_model

    
def server_aggregate_fedfair(global_model,client_models,weights_norm,params):
    """
    This function has aggregation method 'mean'
    """


    for i in range(len(client_models)):
        client_models[i] = utils.scale_model(client_models[i],weights_norm[i])
    
    global_dict = global_model.state_dict()
    for k in global_dict.keys():
        global_dict[k] = torch.stack([model.state_dict()[k] for i,model in enumerate(client_models)], 0)
        global_dict[k] = global_dict[k].sum(0)
        global_model.load_state_dict(global_dict)

    global_model.load_state_dict(global_dict)
    for model in client_models:
        model.load_state_dict(global_model.state_dict())
  

    
    return client_models    











############################ New Kernel Method #########################3
def rbf_kernel(x1, x2, sigma=1.0):
    pairwise_squared_distances = torch.cdist(x1, x2, p=2.0) ** 2
    return torch.exp(-pairwise_squared_distances / (2 * sigma**2))


def client_update_fedfair_kernel(client_model, Weights, Local_Acc, Local_Fair, Local_Gap, client_num=1, flag='one', trainset_global=None, trainset=None, protected_attribute_index=40, Acc_global=None, F_global=None, global_metric_gap=None, params=None,weights=None,fmetric ='spd'):
    # step_size = params['step_size']
    client_model.train()
    fairbatch = True
    device = 'cpu'
    #seed = 0
    if flag == 'one':
        ############## Local training / Debiasing ##################
        optim_init_func = utils.create_optimizer_init_func(torch.optim.Adam)
        optimizer = optim_init_func(client_model)
        b = params['batch_size']
        loader_init_func = utils.create_dataloader_init_func({"batch_size": b})
        trainset_1 = utils.drop_attribute(trainset, protected_attribute_index, weighted=False)
        kernel  = True
        if (kernel):
            loader = DataLoader(trainset, batch_size=len(trainset), shuffle=False)
            generator = torch.Generator()
            device = 'cpu'
            loss_func = torch.nn.BCELoss()
            loss = 0
            for inputs, targets in loader:
                
                sensitive_attributes = (inputs[:, protected_attribute_index])[:, None]
                inputs = utils.drop_attribute_tensor(inputs,protected_attribute_index)
                
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = client_model(inputs)
                
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
               
                loss = loss_func(outputs.view(-1), targets.view(-1)) +  params["fairness"] * fairloss
                
                loss.backward(retain_graph=True)
                optimizer.step()
                   
                
           

        else:
            
            loader = loader_init_func(trainset_1)
            loss_func = torch.nn.BCELoss()
            loss = 0
            for j in tqdm(range(params['local_epochs'])):
                for inputs, targets in loader:
                    inputs = inputs.to(device)
                    targets = targets.to(device)
                    optimizer.zero_grad()
                    outputs = client_model(inputs)
                    outputs = outputs.view(-1)  # align [B,1] → [B]
                    loss = loss_func(outputs, targets)
                    loss.backward()
                    optimizer.step()

        ##########################################
        groups_stats, prob_vector = utils.get_stats(trainset_global, protected_attribute_index)
        if(fmetric == 'spd'): 
            local_spd, g1_rate, g0_rate = spd(client_model, trainset, protected_attribute_index)
        elif(fmetric =='eod'):
            print('here')
            local_eod, g1_rate, g0_rate = eoo_binary_attribute(client_model, trainset, protected_attribute_index)
        else:
            pass
        prob_A_0 = prob_vector['A0 Y1'] + prob_vector['A0 Y0']
        prob_A_1 = prob_vector['A1 Y1'] + prob_vector['A1 Y0']
        
        if(fmetric == 'spd'):
            local_spd_view = g0_rate / prob_A_0 - g1_rate / prob_A_1
            Local_Fair["Client " + str(client_num)] = local_spd_view
        elif(fmetric =='eod'):
            local_eod_view = g0_rate / prob_A_0 - g1_rate / prob_A_1
            Local_Fair["Client " + str(client_num)] = local_eod_view
            

        # local_eod_view = g1_rate/prob_vector['A1 Y1']  - g1_rate/prob_vector['A0 Y1']
        Local_Acc["Client " + str(client_num)] = model_accuracy(client_model, trainset_1)
       

    elif flag == 'two':
        if Local_Fair["Client " + str(client_num)] is None:
            Local_Gap["Client " + str(client_num)] = abs(Local_Acc["Client " + str(client_num)] - Acc_global)
        else:
            Local_Gap["Client " + str(client_num)] = abs(Local_Fair["Client " + str(client_num)] - F_global)

    elif flag == 'three':
        Weights["Client " + str(client_num)] = Weights["Client " + str(client_num)] - (params['beta']) * (Local_Gap["Client " + str(client_num)] - global_metric_gap)

    return client_model

    
 