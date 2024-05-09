import logging
import math
import os
import random
import datasets
import json
import torch
import numpy as np
import transformers
from datasets import load_metric

from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    HfArgumentParser,
    AutoConfig,
    default_data_collator,
    get_scheduler,
    set_seed,
    RobertaForTokenClassification,
    AutoTokenizer,
    BertModel,
    RobertaModel,
)
transformers.logging.set_verbosity_error()

from models.continual_model_zmj import HintLoss,RobertaPrefixForLogParsing_type_meta_prompt

from dataset_log import multi_task_dataset,CustomDataCollator,multi_task_dataset_buffer
from small_eval import evaluate,evaluate_dual
from log_parsing_zmj import template_extraction,template_extraction_dual,template_extraction_prompt_selection
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
device = 'cuda:0'
random.seed(2023)
torch.cuda.manual_seed(2023)
np.random.seed(2023)

def save_json(data, file):
    dict_json = json.dumps(data,indent=1)
    with open(file,'w+',newline='\n') as file:
        file.write(dict_json)

def read_json(file):
    with open(file,'r+') as file:
        content = file.read()
        content = json.loads(content)
        return content



system_type= {
        'Hadoop':0,'HDFS':0,'Spark':0,'Zookeeper':0,'OpenStack':0,
        'BGL':1,'HPC':1,'Thunderbird':1,
        'Windows':2,'Linux':2,'Mac':2,
        'Android':3,'HealthApp':3,
        'Apache':4,'OpenSSH':4,
        'Proxifier':5
    }

ori_data_path = './dataset_cl/datasets_LP'
shot = 1000
order_list = read_json('./dataset_cl/LP_order_16.json')

system2id = {}
id2system = {}
# for i,x in enumerate(files_SmallParse):
#     system2id[x] = i
#     id2system[i] = x


def dual_prefix_w_KD_type_meta(files_SmallParse,order_id):
    task_output_dir = f'./output/LP/dual_prefix_o{order_id}'
    os.makedirs(task_output_dir, exist_ok=True)
    with open(task_output_dir + '/order.txt', 'w') as files:
        files.write(str(files_SmallParse))
    model_path = "roberta-base"
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, add_prefix_space=True, do_lower_case=False)
    model_key = RobertaModel.from_pretrained(model_path)
    for param in model_key.parameters():
        param.requires_grad = False
    model_key.to(device)
    model_dual = RobertaPrefixForLogParsing_type_meta_prompt(num_labels=2, model_path=model_path, batchwise_prompt=True,
                                                             g_prompt_length=10, s_prompt_length=10, t_prompt_length=10,
                                                             t_pool_size=6, t_top_k=1,
                                                             s_prompt_key=True, s_prompt_pool=True, s_pool_size=16,
                                                             s_top_k=1, m_prompt_length=10,
                                                             m_pool_size=40, m_top_k=4, s_prompt_layer_idx=[0],
                                                             use_prefix_tune_for_s_prompt=True, same_key_value=False)

    model_dual.to(device)
    tokenizer.model_max_length = model_dual.roberta.config.max_position_embeddings - 30
    metric = load_metric("./evaluation/seqeval_metric.py")
    label_to_id = {'o': 0, 'i-val': 1}
    id_to_label = {0: 'o', 1: 'i-val'}
    data_collator = CustomDataCollator(
        tokenizer, pad_to_multiple_of=None
    )
    for idx, data_name in enumerate(files_SmallParse):
        type_id = system_type[data_name]
        print(f'idx={idx} data_name={data_name}')
        logdata_train = multi_task_dataset(ori_data_path, [data_name], tokenizer, label_to_id)
        train_dataloader = DataLoader(logdata_train, shuffle=True, collate_fn=data_collator, batch_size=30)

        logdata_eval = multi_task_dataset(ori_data_path, [data_name], tokenizer, label_to_id,
                                          train_eval_test='eval')
        eval_dataloader = DataLoader(logdata_eval, shuffle=True, collate_fn=data_collator, batch_size=30)

        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model_dual.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
            {
                "params": [p for n, p in model_dual.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=9e-3,
            betas=(0.9, 0.999),
            eps=1e-8,
        )

        num_train_epochs = 30
        lr_scheduler = get_scheduler(
            name='polynomial',
            optimizer=optimizer,
            num_warmup_steps=20,
            num_training_steps=num_train_epochs * len(train_dataloader),
        )

        max_train_steps = 3200
        completed_steps = 0

        if idx == 0:
            for epoch in range(num_train_epochs):
                model_key.eval()
                model_dual.train()
                total_loss = []
                for step, batch in enumerate(train_dataloader):
                    batch = {k: v.to(device) for k, v in batch.items()}

                    with torch.no_grad():
                        outputs_key = model_key(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
                    cls_features = outputs_key[0][:, 0, :]

                    outputs = model_dual(**batch,cls_features=cls_features,task_id=idx,type_id=type_id,train=True)
                    loss = outputs.loss
                    total_loss.append(float(loss))
                    loss.backward()
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    completed_steps += 1
                    # print('step=',step,'   loss=',loss)
                    # if completed_steps >= max_train_steps:
                    #     break
                print('epoch = ', epoch, '   loss = ', sum(total_loss) / len(total_loss))

                if epoch % 10 == 0:
                    best_metric = evaluate_dual(metric, model_key, model_dual, tokenizer, eval_dataloader, device,
                                                pad_to_max_length=False, id_to_label=id_to_label)
                    print("Finish training, best metric: ")
                    print(best_metric)
        else:
            for param_cls in model_dual.classifier.parameters():
                param_cls.requires_grad = False
            old_g_prompt = torch.load(task_output_dir + f'/save_models/model_{idx-1}_g_prompt.pth')
            old_g_prompt.requires_grad = False
            old_g_prompt = old_g_prompt.view(-1)
            alpha = 1.5
            compute_kl = HintLoss().to(device)
            for epoch in range(num_train_epochs):
                model_key.eval()
                model_dual.train()
                total_loss = []
                for step, batch in enumerate(train_dataloader):
                    batch = {k: v.to(device) for k, v in batch.items()}

                    with torch.no_grad():
                        outputs_key = model_key(input_ids=batch['input_ids'],attention_mask=batch['attention_mask'])
                    cls_features = outputs_key[0][:,0,:]

                    outputs = model_dual(**batch,cls_features=cls_features,task_id=idx,type_id=type_id,train=True)
                    loss = outputs.loss
                    new_g_prompt = model_dual.g_prefix.view(-1)
                    KDloss = compute_kl(new_g_prompt,old_g_prompt)
                    loss = loss + alpha*KDloss
                    total_loss.append(float(loss))
                    loss.backward()
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    completed_steps += 1
                    # print('step=',step,'   loss=',loss)
                    # if completed_steps >= max_train_steps:
                    #     break
                print('epoch = ', epoch, '   loss = ', sum(total_loss) / len(total_loss))

                if epoch % 10 == 0:
                    best_metric = evaluate_dual(metric, model_key,model_dual, tokenizer, eval_dataloader, device,
                                           pad_to_max_length=False, id_to_label=id_to_label)
                    print("Finish training, best metric: ")
                    print(best_metric)

        os.makedirs(task_output_dir + f'/save_models', exist_ok=True)
        torch.save(model_dual.g_prefix, task_output_dir + f'/save_models/model_{idx}_g_prompt.pth')

        sub_task_output_dir = task_output_dir+f'/Num_{idx}_{data_name}'
        os.makedirs(sub_task_output_dir, exist_ok=True)
        for file in files_SmallParse:
            print(f"Now parse {file}")
            log_file = f'./datasets_LP/{file}/{file}_2k.log_structured.csv'
            template_extraction_dual(tokenizer, model_key,model_dual, device, log_file, max_length=256,
                                model_name='roberta', shot=shot, dataset_name=file,
                                o_dir=sub_task_output_dir, mode='fine-tuning')



if __name__ == '__main__':

    for key in order_list:
        print(key)
        order_id = key.split('_')[-1]
        files_SmallParse = order_list[key]
        dual_prefix_w_KD_type_meta(files_SmallParse,order_id)
