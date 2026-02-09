import os,sys,io
import numpy as np
import pickle
from tqdm import tqdm
import torch
import torch.nn as nn
from typing import Dict, Tuple

def format_params(num: int) -> str:
    """Format parameter count in terms of K (1000)"""
    if num >= 1000:
        return f"{num/1000:.2f}K"
    return str(num)

def count_parameters(model: nn.Module, verbose: bool = False) -> Dict[str, float]:
    """
    Count total, classifier, and non-classifier trainable parameters in a PyTorch model.
    Returns values in terms of K (1000) parameters.
    
    Args:
        model: PyTorch model
        verbose: If True, print parameter counts for each layer
        
    Returns:
        Dictionary containing parameter counts in K (1000s)
    """
    def is_classifier_layer(name: str) -> bool:
        """Check if the layer is part of classifier based on common naming patterns"""
        classifier_keywords = ['classifier', 'fc', 'linear', 'head']
        return any(keyword in name.lower() for keyword in classifier_keywords)
    
    total_params = 0
    classifier_params = 0
    non_classifier_params = 0
    
    # Iterate through all parameters
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            param_count = parameter.numel()
            total_params += param_count
            
            if is_classifier_layer(name):
                classifier_params += param_count
            else:
                non_classifier_params += param_count
                
            if verbose:
                print(f"{name}: {format_params(param_count)} parameters "
                      f"{'(Classifier)' if is_classifier_layer(name) else '(Non-classifier)'}")
    
    results = {
        'total_trainable_params': total_params / 1000,  # Convert to K
        'classifier_params': classifier_params / 1000,   # Convert to K
        'non_classifier_params': non_classifier_params / 1000  # Convert to K
    }
    
    print("\nSummary:")
    print(f"Total trainable parameters (K): {format_params(total_params)}")
    print(f"Classifier parameters (K): {format_params(classifier_params)}")
    print(f"Non-classifier parameters (K): {format_params(non_classifier_params)}")
    try:
        print(f"Classifier parameters percentage (K): {(classifier_params/total_params)*100:.2f}%")
    except ZeroDivisionError:
        print(f"Classifier parameters percentage (K): 0%")
    
    return results

def compute_mas_wgts(model, train, sbatch, args, calc_imp_wrt):
    mas={}
    for n,p in model.named_parameters():
        mas[n]=0*p.data
    
    model.train()
    for bid,batch in tqdm(enumerate(train)): # TODO: refactor to be distributed inference
        batch = {k: v.to(args.device) for k, v in batch.items()}
        # Forward and backward
        model.zero_grad()
        output = model(**batch)
        logits = output["logits"]
        # print(logits.shape)
        tokenwise_l2_norm = logits.pow(2).sum(dim=-1)
        # print(tokenwise_l2_norm.shape)
        seqwise_l2_norm = tokenwise_l2_norm.mean(dim=-1) # TODO: Other options? sum?
        # print(seqwise_l2_norm.shape)
        batchwise_l2_norm = seqwise_l2_norm.sum()
        batchwise_l2_norm.backward()
        # Get gradients
        for n,p in model.named_parameters():
            if p.grad is not None:
                mas[n]+=sbatch*torch.abs(p.grad.data)
        if bid==(20000//sbatch):
            break # TODO: Remove
    
    # Mean importance across all samples
    for n,_ in model.named_parameters():
        mas[n]=mas[n]/len(train)
    # Save
    with open(args.base_dir+'/'+calc_imp_wrt+'_mas_wgts.pkl', 'wb') as fp:
        pickle.dump(mas, fp)
    
    model.zero_grad()

    # return logits
    return

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cuda')
        else:
            return super().find_class(module, name)

def overall_pt_mas_wgts(model, args):
    mas={}
    for n,p in model.named_parameters():
        mas[n]=0*p.data
    
    num_tasks = 8
    for calc_imp_wrt in tqdm(range(1,num_tasks+1,1)):
        with open(args.base_dir+'/pt'+str(calc_imp_wrt)+'_mas_wgts.pkl', 'rb') as handle:
            taskwise_mas = CPU_Unpickler(handle).load()
        for n,p in model.named_parameters():
            mas[n] += taskwise_mas[n].to(args.device)
    for n,p in model.named_parameters():
        mas[n] /= num_tasks
    
    # Save
    with open(args.base_dir+'/pt_mas_wgts.pkl', 'wb') as fp:
        pickle.dump(mas, fp)
    
    return

def compute_rel_imp(args):
    with open(args.base_dir+'/pt_mas_wgts.pkl', 'rb') as handle:
        pt_mas = CPU_Unpickler(handle).load()
    with open(args.base_dir+'/sft_mas_wgts.pkl', 'rb') as handle:
        sft_mas = CPU_Unpickler(handle).load()
    
    epsilon = 0.0000000001
    rel_imp_counter,rel_imp_weighted_mas = {},{}
    for n in sft_mas.keys():
        rel_imp = pt_mas[n]/(pt_mas[n]+sft_mas[n]+epsilon)
        rel_imp_counter[n] = rel_imp
        # Get distribution to set threshold
        tau = args.tau_multiplier*torch.nan_to_num(torch.mean(rel_imp.flatten())).item()
        # Check lamb_up bounding
        lamb_up_lower_bound = np.ceil(1/max(tau,0.05))
        lamb_up_upper_bound = max(args.lamb_max/args.lamb,lamb_up_lower_bound)
        assert args.lamb_up >= lamb_up_lower_bound and args.lamb_up <= lamb_up_upper_bound
        # Compute the new importance weighting
        rel_imp_weighted_mas[n][rel_imp>tau] = args.lamb_up*rel_imp[rel_imp>tau]*pt_mas[n][rel_imp>tau]
        rel_imp_weighted_mas[n][rel_imp<=tau] = args.lamb_down*rel_imp[rel_imp<=tau]*pt_mas[n][rel_imp<=tau]
    
    # Save
    with open(args.base_dir+'/alpha_rel.pkl', 'wb') as fp:
        pickle.dump(rel_imp_counter, fp)
    with open(args.base_dir+'/alpha_dash.pkl', 'wb') as fp:
        pickle.dump(rel_imp_weighted_mas, fp)
    
    return