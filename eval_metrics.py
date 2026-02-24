import torch
from torch.utils.data import Dataset, DataLoader
import utilites as utils

def spd(model, dataset, protected_attribute_index, weighted=False) -> torch.tensor:
    '''
    Calculates the statistical parity difference for a binary protected attribute and 
    binary classification problem. Assumes values 0 and 1 for protected attribute and label. 
    Statistical parity difference is the difference of the success rate between group 1 and group 0.
    '''

    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)

    for inputs, targets in loader:

        #prevents potential issue with long vs int when compared to predictions later
        inputs_mod = utils.drop_attribute_tensor(inputs,protected_attribute_index)
        targets = targets.int()
        outputs = model(inputs_mod)
        
        # if outputs.dim() > targets.dim():
        #     outputs = outputs.squeeze(1)
        
        #print(f'Output shape {outputs.shape}')
        #print(f'Labels shape {targets.shape}')
        
        if len(outputs.shape) == 1 or outputs.shape[1] == 1:
            predictions = torch.round(outputs).int()
        else:
            predictions = torch.argmax(outputs, dim=1, keepdim=True).int()
        
        # true_positives = torch.where(targets == 1, True, False)
        # true_negatives = torch.where(targets == 0, True, False)
        #import pdb;pdb.set_trace()
        
        group1_samples = torch.where(inputs[:,protected_attribute_index].int() == 1, True, False)
        group0_samples = torch.where(inputs[:,protected_attribute_index].int() == 0, True, False)
        
        group0_size = torch.sum(group0_samples)
        group1_size = torch.sum(group1_samples)

        positive_preds = torch.where(predictions == 1, True, False).flatten()
        negative_preds = torch.where(predictions == 0, True, False).flatten()

        group0_positive_preds = torch.logical_and(group0_samples, positive_preds)
        group1_positive_preds = torch.logical_and(group1_samples, positive_preds)

        group0_total_positives = torch.sum(group0_positive_preds)
        group1_total_positives = torch.sum(group1_positive_preds)

        group0_positive_rate = group0_total_positives / group0_size
        group1_positive_rate = group1_total_positives / group1_size

        assert  ((0 <= group0_positive_rate and group0_positive_rate <= 1) and (0 <= group1_positive_rate and group1_positive_rate <= 1))
        statistical_parity_difference = group1_positive_rate - group0_positive_rate

    assert -1 <= statistical_parity_difference and statistical_parity_difference <= 1
    return statistical_parity_difference.item(), group1_positive_rate.item(), group0_positive_rate.item()


def eoo_binary_attribute(model, dataset, protected_attribute_index):
    '''
    calculates the difference between true positive rates for two groups.
    Assumes groups are identified by a single one hot coordinate at protected_attribute_index.
    Also assumes a binary classification task.
    '''

    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)

    for inputs, targets in loader:
        group0_samples = torch.where((inputs[:,protected_attribute_index]).int() == 0, True, False)
        group1_samples = torch.where((inputs[:,protected_attribute_index]).int() == 1, True, False)

        inputs_mod = utils.drop_attribute_tensor(inputs,protected_attribute_index)
        outputs = model(inputs_mod)
        # if outputs.dim() > targets.dim():
        #     outputs = outputs.squeeze(1)
        #outputs = model(inputs[:,:-1])
        if len(outputs.shape) == 1 or outputs.shape[1] == 1:
            predictions = torch.round(outputs).int()
        else:
            predictions = torch.argmax(outputs, dim=1, keepdim=True).int()

        positive_labels = torch.where(targets.int() == 1, True, False).flatten()
        positive_predictions = torch.where(predictions == 1, True, False).flatten()
        true_positives = torch.logical_and(positive_labels, positive_predictions)


        group0_true_positives = torch.sum(torch.logical_and(group0_samples, true_positives))
        group1_true_positives = torch.sum(torch.logical_and(group1_samples, true_positives))

        # print(f"positives predictions = {torch.sum(positive_predictions)}\ngroup 0 true positives: {group0_true_positives}, group 1 true positives: {group1_true_positives}\n\n")

        group0_positives = torch.sum(torch.logical_and(group0_samples, positive_labels))
        group1_positives = torch.sum(torch.logical_and(group1_samples, positive_labels))
        group0_true_positive_rate = group0_true_positives.float() / group0_positives.float()
        group1_true_positive_rate = group1_true_positives.float() / group1_positives.float()

        equalized_opportunity_odds = group1_true_positive_rate - group0_true_positive_rate
        assert -1 <= equalized_opportunity_odds and equalized_opportunity_odds <= 1
        return equalized_opportunity_odds.item(), group1_true_positive_rate.item(), group0_true_positive_rate.item()


def model_accuracy(model, dataset, binary : bool = False):
    
    testloader = torch.utils.data.DataLoader(dataset, batch_size=1000,
                                                shuffle=False)
    with torch.no_grad():
        num_correct = 0.0
        num_wrong = 0.0
        device = 'cpu' # torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        for data in testloader:
            inputs, labels= data
          
            
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            output = model(inputs)
            
           
            
            # if output.dim() > labels.dim():
            #     output = output.squeeze(1)
            
         
            if binary:
                predictions = torch.round(output)
            else:
                predictions = torch.argmax(output, 1)
                
            num_correct += torch.sum(torch.where(predictions == labels, 1.0, 0.0))
        return float(num_correct)/(float(len(dataset)))

