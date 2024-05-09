import numpy as np
import json

def save_json(data, file):
    dict_json = json.dumps(data,indent=1)
    with open(file,'w+',newline='\n') as file:
        file.write(dict_json)

def read_json(file):
    with open(file,'r+') as file:
        content = file.read()
        content = json.loads(content)
        return content

def load_data(file):

    forget = {}
    transfer = {}
    zero = {}
    final = {}
    results = [forget,transfer,zero,final]

    with open(file,'r') as f:
        for i,line in enumerate(f):
            line_list = line.split()
            Acc = line_list[-7]
            Rec = line_list[-5]
            Pre = line_list[-3]
            F1 = line_list[-1]

            results[i]['Acc'] = Acc
            results[i]['Rec'] = Rec
            results[i]['Pre'] = Pre
            results[i]['F1'] = F1
    return results

orders = read_json('D:\ZMJ\pythonProject\Log_continual_learning\Anomaly_Detection\order_4.json')

methods = [
        'sequential_fine-tuning',
        # 'incremental_joint_learning',
        'sequential_keep_head',
        'sequential_keep_body',
        'sequential_keep_body_wo_1_9',
        'sequential_ewc',
        'sequential_er',
        'simple_knowledge_distill',
        'hint_knowledge_distill',
    ]


methods_results = {}
for method in methods:
    forget_Acc_metric = np.zeros(10)
    forget_F1_metric = np.zeros(10)
    transfer_Acc_metric = np.zeros(10)
    transfer_F1_metric = np.zeros(10)
    zero_Acc_metric = np.zeros(10)
    zero_F1_metric = np.zeros(10)
    final_Acc_metric = np.zeros(10)
    final_F1_metric = np.zeros(10)

    for i,order in enumerate(orders):
        # order_num = len(order)

        path = f'D:\ZMJ\pythonProject\Log_continual_learning\\Anomaly_Detection\o{i+1}\{method}\evalution_results'

        results = load_data(path)
        forget_Acc_metric[i]=results[0]['Acc']
        forget_F1_metric[i]=results[0]['F1']
        transfer_Acc_metric[i]=results[1]['Acc']
        transfer_F1_metric[i]=results[1]['F1']
        zero_Acc_metric[i]=results[2]['Acc']
        zero_F1_metric[i]=results[2]['F1']
        final_Acc_metric[i]=results[3]['Acc']
        final_F1_metric[i] = results[3]['F1']

    forget_Acc_metric_avg = sum(forget_Acc_metric)/len(forget_Acc_metric)
    forget_F1_metric_avg = sum(forget_F1_metric)/len(forget_F1_metric)
    transfer_Acc_metric_avg = sum(transfer_Acc_metric)/len(transfer_Acc_metric)
    transfer_F1_metric_avg = sum(transfer_F1_metric)/len(transfer_F1_metric)
    zero_Acc_metric_avg = sum(zero_Acc_metric)/len(zero_Acc_metric)
    zero_F1_metric_avg = sum(zero_F1_metric)/len(zero_F1_metric)
    final_Acc_metric_avg = sum(final_Acc_metric)/len(final_Acc_metric)
    final_F1_metric_avg = sum(final_F1_metric)/len(final_F1_metric)
    print(1)

