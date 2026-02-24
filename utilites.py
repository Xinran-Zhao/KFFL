#import pandas as pd
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
    # pyrfm (neonnnnn/pyrfm) is not installed; kernel-based methods (KRTWD, KRTD,
    # FairFed_w_FairBatch_kernel) will not work, but all other methods are unaffected.
    OrthogonalRandomFeature = CompactRandomFeature = RandomFourier = FastFood = None
from tqdm import tqdm
import sys, os
import numpy as np
import math
import random
import itertools
import copy
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler
from sklearn.model_selection import train_test_split
from FairBatchSampler import FairBatch, CustomDataset

def get_num_features(trainset):
    loader = DataLoader(trainset,batch_size = len(trainset))
    for x,y in loader:
        n,d = x.shape[:]
    return d

def print_client_gender_distribution(train_datasets, protected_idx, label_0='Female', label_1='Male'):
    """
    打印每个client数据集中的性别分布。
    protected_idx: 性别特征的列索引（ADULT数据集中为40，即sex_Male）
    label_0: 值为0表示的性别（Female）
    label_1: 值为1表示的性别（Male）
    """
    print("\n========== Client Gender Distribution ==========")
    for i, ds in enumerate(train_datasets):
        loader = DataLoader(ds, batch_size=len(ds))
        for inputs, _ in loader:
            gender_col = inputs[:, protected_idx]
            n_total = len(gender_col)
            n_male   = int((gender_col == 1).sum().item())
            n_female = int((gender_col == 0).sum().item())
            pct_male   = 100.0 * n_male   / n_total if n_total > 0 else 0
            pct_female = 100.0 * n_female / n_total if n_total > 0 else 0
            print(f"  Client {i}: Total={n_total:5d} | "
                  f"{label_1}={n_male:5d} ({pct_male:.1f}%) | "
                  f"{label_0}={n_female:5d} ({pct_female:.1f}%)")
    print("================================================\n")


def drop_attribute(dataset : Dataset, attribute_idx : int, weighted : bool):
    
    loader = DataLoader(dataset, batch_size=len(dataset))

    for loader_val in loader:
        if not weighted:
            inputs, targets = loader_val
        else:
            inputs, targets, weights = loader_val
        cols = [i for i in range(inputs.shape[1]) if i != attribute_idx]
        if not weighted:
            new_dataset = TensorDataset(inputs[:, cols], targets)
        else:
            pass
            #new_dataset = WeightedDataset(inputs[:, cols], targets, weights)

    return new_dataset

def create_dataloader_init_func(dataloader_kwargs : 'dict[str, any]'):
    def dataloader_initializer(dataset):
        return torch.utils.data.DataLoader(dataset, **dataloader_kwargs)
    return dataloader_initializer

def create_optimizer_init_func(optimizer, optimizer_kwargs : 'dict[str, any]' = None):
    def optimizer_initializer(model):
        if optimizer_kwargs:
            return optimizer(model.parameters(), **optimizer_kwargs)
        else:
            return optimizer(model.parameters())
    return optimizer_initializer


def create_perturbations_optimizer_init_func(optimizer, optimizer_kwargs=None):
    def optimizer_initializer(perturbation_tensor):
        if optimizer_kwargs:
            return optimizer([perturbation_tensor], **optimizer_kwargs)
        else:
            return optimizer([perturbation_tensor])
    return optimizer_initializer


def convert_to_gigabytes(number):
    gigabytes = number / (1024**3)
    return gigabytes

def convert_to_megabytes(number):
    megabytes = number / (1024**2)
    return megabytes
def convert_to_kilobytes(number):
    megabytes = number / (1024)
    return megabytes

def divide_odd_number(number):
    even_part = number // 2
    odd_part = number - even_part
    return even_part, odd_part

def split_odd_number(number, num_parts):
    quotient = number // num_parts
    remainder = number % num_parts

    parts = [quotient] * num_parts
    for i in range(remainder):
        parts[i] += 1

    return parts
def count_group_samples_in_clients(train_datasets, group_key):
    """
    Count the number of samples of a specific group (group_key) in each client's dataset.

    Args:
        train_datasets (list): A list of client datasets (tuples with client index and dataset).
        group_key (str): The key representing the group you want to count.

    Returns:
        list: A list of tuples containing client index and the count of group samples.
    """
    group_counts = []

    for client_tuple in train_datasets:
        print(client_tuple)
        client_index, client_dataset = client_tuple  # Unpack the tuple
        group_samples = sum(1 for sample in client_dataset if sample['group'] == group_key)
        group_counts.append((client_index, group_samples))

    return group_counts

def split_dataset_group_NonIID(train_dataset:Dataset , group_index:int):
    
    loader = DataLoader(train_dataset, batch_size=len(train_dataset))
    group_ds = {}
    lengths = {}
    for loader_val in loader:
        
        inputs, targets, = loader_val
        group_col = inputs[:,group_index]
        
        group_unique_vals = torch.unique(group_col)
        group_num_unique_vals = group_unique_vals.size(0)
        
        targets_unique_vals = torch.unique(targets)
        targets__num_unique_vals = targets_unique_vals.size(0)
        
        
        for i in group_unique_vals:
            indices = torch.where(group_col == i)
            lengths["Group " + str(i)] = inputs[ indices[0]].shape[0]
            #print(inputs[ indices[0]].shape)
            
            group_ds[str(int(i))]= TensorDataset(inputs[indices[0]], targets[indices[0]])
    return group_ds,lengths,targets_unique_vals

def create_local_datasets(dataset, split_type: str, num_clients: int, protected_idx: int):
    # dataset = SensrAdultDataset('./Raw_Data/Adult/')
    partition = 0.8 # 0.8,0.2
    # Use below only for centralized case....
    central_split = True
    if(central_split):
        # Considering a small subset of the training_data
        central_partition = 0.2
        dataset,_ = random_split(dataset, [int(len(dataset) * central_partition), int(len(dataset) - int(len(dataset) * central_partition))])

    trainset, testset = random_split(dataset, [int(len(dataset) * partition), int(len(dataset) - int(len(dataset) * partition))])
    # n = len(trainset)
    # trainset1 = drop_attribute(trainset, 40, weighted=False)
    # testset1 = drop_attribute(testset, 40, weighted=False)

    train_datasets = []

    if split_type == 'Non-IID':
        if num_clients > 2:
            group_ds, lengths, targets_unique_vals = split_dataset_group_NonIID(trainset, protected_idx)
    
            # Calculate the number of odd and even clients
            [odd, even] = split_odd_number(num_clients, 2)
    
            # Initialize lists to store samples for odd and even clients
            samples_odd = []
            samples_even = []
    
            for key, value in group_ds.items():
                if key == '0':
                    # Calculate the target distribution for odd and even clients
                    dist_odd = [int(len(value) * 0.1) for _ in range(odd)]
                    dist_even = [int(len(value) * 0.9) for _ in range(even)]
    
                    # Distribute samples to odd clients
                    for i in range(odd):
                        samples_odd.extend(random_split(value, [dist_odd[i], len(value) - dist_odd[i]])[0])
    
                    # Distribute samples to even clients
                    for i in range(even):
                        samples_even.extend(random_split(value, [dist_even[i], len(value) - dist_even[i]])[0])
    
                elif key == '1':
                    # Calculate the target distribution for odd and even clients
                    dist_odd = [int(len(value) * 0.1) for _ in range(odd)]
                    dist_even = [int(len(value) * 0.9) for _ in range(even)]
    
                    # Distribute samples to odd clients
                    for i in range(odd):
                        samples_odd.extend(random_split(value, [dist_odd[i], len(value) - dist_odd[i]])[0])
    
                    # Distribute samples to even clients
                    for i in range(even):
                        samples_even.extend(random_split(value, [dist_even[i], len(value) - dist_even[i]])[0])
    
            # Shuffle the samples to ensure randomness
            random.shuffle(samples_odd)
            random.shuffle(samples_even)
    
            # Assign samples to odd and even clients
            for i in range(odd):
                train_datasets.append(samples_odd[i::odd])
    
            for i in range(even):
                train_datasets.append(samples_even[i::even])
    
        elif num_clients == 1:
            train_datasets.append(trainset)
        else:
            print("Invalid number of clients")


    elif split_type == 'IID':
        ### Doing this for IID Distribution ##############
        if num_clients > 2:
            extra_samples = len(trainset) - num_clients * int(len(trainset) / num_clients)
            trainset, _ = random_split(trainset, [len(trainset) - extra_samples, extra_samples])
            dist = [int(len(trainset) / num_clients) for _ in range(num_clients)]
            train_datasets = random_split(trainset, dist)
        elif num_clients == 1:
            train_datasets.append(trainset)
        else:
            print("Invalid number of clients")

    else:
        print('Not a valid distribution type. Choose from IID or Non-IID ')

   # import pdb;pdb.set_trace() 
   # Example usage:
       
########################################
    # group_key = '0'  # Replace with the group key you want to count
    # group_counts = count_group_samples_in_clients(train_datasets, group_key)
    
    # # Print the counts for each client
    # for client_index, count in group_counts:
    #     print(f"Client {client_index}: {count} samples of group {group_key}")
 #########################################################################################       
        
        
        
    assert len(train_datasets) == num_clients
    
    return train_datasets, trainset, testset



def model_l2(model : Module, dtype : torch.dtype=torch.float, requires_grad : bool=True):
    
    l2 = torch.tensor([0.0], dtype=dtype, requires_grad=requires_grad)
    for param in model.parameters():
        l2 = l2.add(torch.sum(param * param))
    return l2

def subtract_models(model1 : Module, model2: Module) -> Module:
    model1_params = [param for param in model1.parameters()]
    model2_params = [param for param in model2.parameters()]
    new_model = copy.deepcopy(model1)
    for i, param in enumerate(new_model.parameters()):
        new_param = model1_params[i] - model2_params[i]
        param.data = new_param.data
        
    return new_model

def scale_model(model,scale_factor):
    
    model_dict = model.state_dict()
    
    for k in model_dict.keys():
        model_dict[k] = model_dict[k] * scale_factor
    
    model.load_state_dict(model_dict)
    
    return model
def scale_gradients(gradients,scale_factor):
    
    
    for k in gradients.keys():
        gradients[k] = gradients[k] * scale_factor
    
    
    return gradients


def calculate_weights(train_datasets):
    weights = []
    for i in train_datasets:
        weights.append(len(i))
    weights_norm = [w /sum(weights) for w in weights]
    
    return weights,weights_norm


def get_attribute_tensor(inputs, attribute_idx : int):
    #loader = DataLoader(dataset, batch_size=len(dataset))
    cols_non_sens = [i for i in range(inputs.shape[1]) if i != attribute_idx]
    cols_sens = [i for i in range(inputs.shape[1]) if i == attribute_idx]
    non_sens_feature = inputs[:,cols_non_sens]
    sens_feature = inputs[:,cols_sens]
    #new_dataset = TensorDataset(inputs[:, cols], targets)
       
            #new_dataset = WeightedDataset(inputs[:, cols], targets, weights)

    return non_sens_feature ,sens_feature


def drop_attribute_tensor(inputs, attribute_idx : int):
    #loader = DataLoader(dataset, batch_size=len(dataset))
    cols = [i for i in range(inputs.shape[1]) if i != attribute_idx]
    feature = inputs[:,cols]
    #new_dataset = TensorDataset(inputs[:, cols], targets)
       
            #new_dataset = WeightedDataset(inputs[:, cols], targets, weights)

    return feature

def parameter_count(model):
    count = 0 
    for i in model.parameters():
        count += i.flatten().shape[0]
    return count

def split_dataset_group(train_dataset:Dataset , group_index:int):
    
    ## Drop the sensitive attribute and make it like rho
    
    loader = DataLoader(train_dataset, batch_size=len(train_dataset))
    group_ds = {}
    lengths = {}
    for loader_val in loader:
        
        inputs, targets, = loader_val
        group_col = inputs[:,group_index]
        unique_vals = torch.unique(group_col)
        num_unique_vals = unique_vals.size(0)
        
        for i in range(len(unique_vals)):
            indices = torch.where(group_col == i)
            targets[indices]
            lengths[str(i)] = inputs[ indices[0]].shape[0]
            #print(inputs[ indices[0]].shape)
            group_ds[str(i)]= TensorDataset(inputs[indices[0]], targets[indices[0]])
        
    return group_ds,lengths


def rho(dataset,group_index = 40):     
    
    loader = DataLoader(dataset, batch_size=len(dataset))
    for loader_val in loader:
        inputs, targets = loader_val
        total_len = len(targets)
        ## How many difff groups?
        group_col = inputs[:,group_index]
        unique_vals = torch.unique(group_col)
        num_groups= unique_vals.size(0)
    
    
        prob_vector =[]   
        group_stats = {}
    
        for i,val in enumerate(unique_vals):
            indices = torch.where(group_col == val)
            targets[indices]
            prob_vector.append(len(targets[indices])/len(targets))
            group_stats[i] = len(targets[indices])
    
    assert len(prob_vector) == num_groups
    
    return prob_vector,group_stats

def euclidean_proj_simplex(v, s=1):
  
    assert s > 0, "Radius s must be strictly positive (%d <= 0)" % s
    n, = v.shape  # will raise ValueError if v is not 1-D
    # check if we are already on the simplex
    
    if v.sum() == s and np.alltrue(v >= 0):
        # best projection: itself!
        return v
    # get the array of cumulative sums of a sorted (decreasing) copy of v
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    # get the number of > 0 components of the optimal solution
    rho = np.nonzero(u * np.arange(1, n+1) > (cssv - s))[0][-1]
    # compute the Lagrange multiplier associated to the simplex constraint
    #theta = float(cssv[rho] - s) / rho
    theta = float(cssv[rho] - s) / (rho+1)
    # compute the projection by thresholding v using theta
    w = (v - theta).clip(min=0)
    return w

def global_risk(client_risks,group_stats):
    
    total_clients = len(client_risks)
    
    # get the list of row and column keys
    clients = list(client_risks.keys())
    group_losses = list(next(iter(client_risks.values())).keys())
    
    # create an empty matrix with the right dimensions
    num_clients = len(clients)
    num_group_losses = len(group_losses)
    
    #group_stats = np.array(group_stats)
   

    matrix = np.zeros((num_clients,num_group_losses))
    #print(matrix.shape)
    # fill in the matrix with the values from the dictionary
    for i, c in enumerate(clients):
        for j, g in enumerate(group_losses):
            matrix[i,j] = client_risks[c][g]
    ## How to ensure that the jth column of group_stats  is the same as  jth column of matrix -- ??
    for j, g in enumerate(group_losses):
        #print(group_stats[j])
        matrix[:,j] = matrix[:,j]/group_stats[j]
    
    sum = np.sum(matrix, axis=0)

   # print(matrix)
   # print(sum)
    return sum


def split_dataset_group_ff(train_dataset:Dataset , group_index:int):
    
    loader = DataLoader(train_dataset, batch_size=len(train_dataset))
    group_ds = {}
    lengths = {}
    for loader_val in loader:
        
        inputs, targets, = loader_val
        group_col = inputs[:,group_index]
        
        group_unique_vals = torch.unique(group_col)
        group_num_unique_vals = group_unique_vals.size(0)
        
        targets_unique_vals = torch.unique(targets)
        targets__num_unique_vals = targets_unique_vals.size(0)
        
        for i in group_unique_vals:
            indices = torch.where(group_col == i)
            lengths["Group " + str(i)] = inputs[ indices[0]].shape[0]
            #print(inputs[ indices[0]].shape)
            
            group_ds[str(int(i))]= TensorDataset(inputs[indices[0]], targets[indices[0]])
        
    return group_ds,lengths,targets_unique_vals

def get_stats_target(train_dataset:Dataset,group_index:int):
    
    loader = DataLoader(train_dataset, batch_size=len(train_dataset))
    group_dist = {}
    for loader_val in loader:
        
        inputs, targets, = loader_val
        targets_unique_vals = torch.unique(targets)
        targets__num_unique_vals = targets_unique_vals.size(0)
        
        for i in targets_unique_vals:
            indices = torch.where(targets == i)
            group_dist["Y" + str(int(i))] = targets[indices].shape[0]
            
    return group_dist

def get_stats(train_dataset:Dataset,group_index:int):
    
    group_ds,lengths,unique_targets = split_dataset_group_ff(train_dataset ,group_index)
    num_samples = sum(lengths.values())
    
    #print("Every group length",lengths)
    groups_stats = {}
    prob_vector = {}
    for group_num,dataset in group_ds.items():
        temp = get_stats_target(dataset,group_index)
        grp_name = 'A' + str(group_num)
        groups_stats[grp_name] = temp
        
        for label,samples in temp.items():
            prob_vector[grp_name +" "+ label] = samples/num_samples
            
    
    
    return groups_stats,prob_vector


def fairbatch_dataset(dataset:Dataset,protected_idx:int):
    # Split the dataset into feature_tensors and target_tensors
    feature_tensors = [batch[0] for batch in dataset]
    target_tensors = [batch[1] for batch in dataset]
    
    # Create tensors from the lists
    feature_tensor = torch.stack(feature_tensors)
    non_sens_feature ,sens_feature = get_attribute_tensor(feature_tensor , protected_idx )
    target_tensor = torch.stack(target_tensors)
    ## Replace labels
    
    z_train = sens_feature.squeeze(1)
    xz_train = non_sens_feature.squeeze(1)
    y_train =  target_tensor.squeeze(1)
    

    
    
    xz_train = torch.FloatTensor(xz_train)
    y_train = torch.FloatTensor(y_train).float()
    z_train = torch.FloatTensor(z_train)

    
    
    y_train = torch.where(y_train == 0.0, -1.0, y_train.double()).float()
    
    
    return xz_train,z_train,y_train
    
    
        