import pandas as pd
import torch
# from datasets import load_dataset
from torch.utils.data import DataLoader
import os
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from dataset_log_AD import multi_task_dataset_AD,CustomDataCollator_AD



def log_AD(tokenizer, model, device, log_file, max_length,
                        dataset_name="BGL"):
    model.to(device)
    model.eval()
    data_collator = CustomDataCollator_AD(tokenizer, pad_to_multiple_of=None)
    logdata_test = multi_task_dataset_AD(log_file, [dataset_name], tokenizer, train=False)
    test_dataloader = DataLoader(logdata_test, shuffle=False,collate_fn=data_collator, batch_size=60)

    y_true = []
    y_pred = []
    for step, batch in enumerate(test_dataloader):
        input_ids = batch['input_ids'].to(device)
        atten_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].squeeze(1).detach().cpu().numpy()
        y_true.extend(labels.tolist())
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=atten_mask)
            predictions = outputs.logits.argmax(dim=-1)
            pre = predictions.detach().cpu().numpy().tolist()
            y_pred.extend(pre)

    acc = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    p = precision_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    eval_metric = {'acc': acc, 'recall': recall, 'prec': p, 'f1': f1}
    return eval_metric

def log_AD_dual(tokenizer, model_dual,model_key, device, log_file, max_length,
                        dataset_name="BGL"):
    model_key.to(device)
    model_key.eval()
    model_dual.to(device)
    model_dual.eval()
    data_collator = CustomDataCollator_AD(tokenizer, pad_to_multiple_of=None)
    logdata_test = multi_task_dataset_AD(log_file, [dataset_name], tokenizer, train=False)
    test_dataloader = DataLoader(logdata_test, shuffle=False,collate_fn=data_collator, batch_size=60)

    y_true = []
    y_pred = []
    for step, batch in enumerate(test_dataloader):
        input_ids = batch['input_ids'].to(device)
        atten_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].squeeze(1).detach().cpu().numpy()
        y_true.extend(labels.tolist())
        with torch.no_grad():
            outputs_key = model_key(input_ids=input_ids, attention_mask=atten_mask)
            cls_features = outputs_key[0][:, 0, :]
            outputs = model_dual(input_ids=input_ids, attention_mask=atten_mask,cls_features=cls_features,train=False)
            predictions = outputs.logits.argmax(dim=-1)
            pre = predictions.detach().cpu().numpy().tolist()
            y_pred.extend(pre)

    acc = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    p = precision_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    eval_metric = {'acc': acc, 'recall': recall, 'prec': p, 'f1': f1}
    return eval_metric

def log_AD_dual_idselection(tokenizer, model_dual,model_key, device, log_file, max_length,type_id,system_id,
                        dataset_name="BGL",):
    model_key.to(device)
    model_key.eval()
    model_dual.to(device)
    model_dual.eval()
    data_collator = CustomDataCollator_AD(tokenizer, pad_to_multiple_of=None)
    logdata_test = multi_task_dataset_AD(log_file, [dataset_name], tokenizer, train=False)
    test_dataloader = DataLoader(logdata_test, shuffle=False,collate_fn=data_collator, batch_size=60)

    y_true = []
    y_pred = []
    type_id_true_list = [type_id]*len(logdata_test.datas)
    system_id_true_list = [system_id]*len(logdata_test.datas)
    type_id_pre_list=[]
    system_id_pre_list=[]
    for step, batch in enumerate(test_dataloader):
        input_ids = batch['input_ids'].to(device)
        atten_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].squeeze(1).detach().cpu().numpy()
        y_true.extend(labels.tolist())
        with torch.no_grad():
            outputs_key = model_key(input_ids=input_ids, attention_mask=atten_mask)
            cls_features = outputs_key[0][:, 0, :]
            outputs,type_id_pre,system_id_pre = model_dual(input_ids=input_ids, attention_mask=atten_mask,cls_features=cls_features,train=False)
            predictions = outputs.logits.argmax(dim=-1)
            pre = predictions.detach().cpu().numpy().tolist()
            y_pred.extend(pre)
            type_id_pre=type_id_pre.squeeze().detach().cpu().numpy().tolist()
            type_id_pre_list.extend(type_id_pre)
            system_id_pre=system_id_pre.squeeze().detach().cpu().numpy().tolist()
            system_id_pre_list.extend(system_id_pre)

    count_of_correct_type_id = type_id_pre_list.count(int(type_id))
    count_of_correct_system_id = system_id_pre_list.count(int(system_id))

    acc_t = accuracy_score(type_id_true_list, type_id_pre_list)
    recall_t = recall_score(type_id_true_list, type_id_pre_list)
    p_t = precision_score(type_id_true_list, type_id_pre_list)
    f1_t = f1_score(type_id_true_list, type_id_pre_list)


    acc_s = accuracy_score(system_id_true_list, system_id_pre_list)
    recall_s = recall_score(system_id_true_list, system_id_pre_list,average='macro')
    p_s = precision_score(system_id_true_list, system_id_pre_list,average='macro')
    f1_s = f1_score(system_id_true_list, system_id_pre_list,average='macro')
    eval_metric_ts = {'acc_t': acc_t, 'recall_t': recall_t, 'prec_t': p_t, 'f1_t': f1_t,'acc_s': acc_s, 'recall_s': recall_s, 'prec_s': p_s, 'f1_s': f1_s}

    acc = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    p = precision_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    eval_metric = {'acc': acc, 'recall': recall, 'prec': p, 'f1': f1}
    return eval_metric,eval_metric_ts,type_id_pre_list,system_id_pre_list

def write_pd():
    pass

def metric_avg(metrics:dict):
    acc = 0
    rec = 0
    pre = 0
    f1 = 0
    for key in metrics:
        acc += metrics[key]['acc']
        rec += metrics[key]['recall']
        pre += metrics[key]['prec']
        f1 += metrics[key]['f1']
    acc_avg = acc/len(metrics)
    rec_avg = rec/len(metrics)
    pre_avg = pre/len(metrics)
    f1_avg = f1/len(metrics)
    metrics['Average'] = {'acc': acc_avg, 'recall': rec_avg, 'prec': pre_avg, 'f1': f1_avg}
    return metrics