import json
import pandas as pd
import os
import numpy as np

def load_results(root_path,order):
    all_results = {}
    for idx,data_name in enumerate(order):
        result_path = root_path+f'Num_{idx}_{data_name}/metrics.csv'
        df = pd.read_csv(result_path)
        column_names = (df.columns)[1:]
        acc_row = df.iloc[0, 1:]
        rec_row = df.iloc[1, 1:]
        pre_row = df.iloc[2, 1:]
        f1_row = df.iloc[3, 1:]
        # GA_order = []
        # PA_order = []
        # ED_order = []
        # for name in order:
        #     ga = ga_row[name]
        #     pa = pa_row[name]
        #     ed = ed_row[name]
        #     GA_order.append(ga)
        #     PA_order.append(pa)
        #     ED_order.append(ed)
        all_results[data_name]={'Accuracy': acc_row, 'Recall': rec_row, 'Precision': pre_row, 'F1': f1_row}

    return all_results

def load_results_onefile(path):

    df = pd.read_csv(path)
    acc_row = df.iloc[0, 1:]
    rec_row = df.iloc[1, 1:]
    pre_row = df.iloc[2, 1:]
    f1_row = df.iloc[3, 1:]
    all_results = {'Accuracy': acc_row, 'Recall': rec_row, 'Precision': pre_row, 'F1': f1_row}
    return all_results


def compute_forget(metric,order):
    F_list = []
    for jdx in range(1,len(order)):
        F_j_list = []
        for idx in range(0,jdx):
            R_i_j = metric[idx,jdx]
            max_1_j__1 = max(metric[idx,:jdx])
            F_i_j = max_1_j__1-R_i_j
            F_j_list.append(F_i_j)
        F_j = sum(F_j_list)/jdx
        F_list.append(F_j)
    F = sum(F_list)/(len(order)-1)
    return F


def Forgetting(results_cl,order):

    R_acc_metric = np.zeros((len(order),len(order)))
    R_rec_metric = np.zeros((len(order),len(order)))
    R_pre_metric = np.zeros((len(order),len(order)))
    R_f1_metric = np.zeros((len(order), len(order)))
    # convert to metric
    for idx, dataname in enumerate(order):
        CL_acc = results_cl[dataname]['Accuracy']
        CL_rec = results_cl[dataname]['Recall']
        CL_pre = results_cl[dataname]['Precision']
        CL_f1 = results_cl[dataname]['F1']
        for jdx, dataname_c in enumerate(order):
            R_acc_metric[jdx,idx] = CL_acc[dataname_c]
            R_rec_metric[jdx,idx] = CL_rec[dataname_c]
            R_pre_metric[jdx,idx] = CL_pre[dataname_c]
            R_f1_metric[jdx, idx] = CL_f1[dataname_c]
    F_acc = compute_forget(R_acc_metric,order)
    F_rec = compute_forget(R_rec_metric,order)
    F_pre = compute_forget(R_pre_metric,order)
    F_f1 = compute_forget(R_f1_metric,order)

    return F_acc,F_rec,F_pre,F_f1

def compute_Transfer(CL,FT,name):
    R_i_i = CL[name]
    R_i = FT[name]
    T = R_i_i - R_i
    return T

def Transfer(results_cl,results_ft,order):
    T_acc_all = []
    T_rec_all = []
    T_pre_all = []
    T_f1_all = []
    for i in range(1,len(order)):
        dataname = order[i]
        CL_acc = results_cl[dataname]['Accuracy']
        CL_rec = results_cl[dataname]['Recall']
        CL_pre = results_cl[dataname]['Precision']
        CL_f1 = results_cl[dataname]['F1']

        FT_acc = results_ft['Accuracy']
        FT_rec = results_ft['Recall']
        FT_pre = results_ft['Precision']
        FT_f1 = results_ft['F1']

        T_acc = compute_Transfer(CL_acc,FT_acc,dataname)
        T_rec = compute_Transfer(CL_rec,FT_rec,dataname)
        T_pre = compute_Transfer(CL_pre,FT_pre,dataname)
        T_f1 = compute_Transfer(CL_f1, FT_f1, dataname)

        T_acc_all.append(T_acc)
        T_rec_all.append(T_rec)
        T_pre_all.append(T_pre)
        T_f1_all.append(T_f1)

    T_acc_avg = sum(T_acc_all)/(len(order)-1)
    T_rec_avg = sum(T_rec_all)/(len(order)-1)
    T_pre_avg = sum(T_pre_all)/(len(order)-1)
    T_f1_avg = sum(T_f1_all)/(len(order)-1)
    return T_acc_avg,T_rec_avg,T_pre_avg,T_f1_avg

def compute_ZeroTransfer(cl_metric,zs_metric,order):
    T0_list=[]
    for idx in range(1,len(order)):
        R0_i = zs_metric[idx]
        T0_i_list = []
        for jdx in range(0,idx):
            R_i_j = cl_metric[idx,jdx]
            T0_i_list.append(R_i_j)
        T0_i = sum(T0_i_list)/(idx)-R0_i
        T0_list.append(T0_i)
    T0 = sum(T0_list)/(len(order)-1)
    return T0

def zero_Transfer(results_cl,results_zs,order):
    R_acc_metric = np.zeros((len(order), len(order)))
    R_rec_metric = np.zeros((len(order), len(order)))
    R_pre_metric = np.zeros((len(order), len(order)))
    R_f1_metric = np.zeros((len(order), len(order)))
    ZS_acc_metric = np.zeros(len(order))
    ZS_rec_metric = np.zeros(len(order))
    ZS_pre_metric = np.zeros(len(order))
    ZS_f1_metric = np.zeros(len(order))
    # convert to metric
    for idx, dataname in enumerate(order):
        ZS_acc_metric[idx] = results_zs['Accuracy'][dataname]
        ZS_rec_metric[idx] = results_zs['Recall'][dataname]
        ZS_pre_metric[idx] = results_zs['Precision'][dataname]
        ZS_f1_metric[idx] = results_zs['F1'][dataname]

        CL_acc = results_cl[dataname]['Accuracy']
        CL_rec = results_cl[dataname]['Recall']
        CL_pre = results_cl[dataname]['Precision']
        CL_f1 = results_cl[dataname]['F1']
        for jdx, dataname_c in enumerate(order):
            R_acc_metric[jdx, idx] = CL_acc[dataname_c]
            R_rec_metric[jdx, idx] = CL_rec[dataname_c]
            R_pre_metric[jdx, idx] = CL_pre[dataname_c]
            R_f1_metric[jdx, idx] = CL_f1[dataname_c]

    ZT_acc = compute_ZeroTransfer(R_acc_metric,ZS_acc_metric,order)
    ZT_rec = compute_ZeroTransfer(R_rec_metric, ZS_rec_metric, order)
    ZT_pre = compute_ZeroTransfer(R_pre_metric, ZS_pre_metric, order)
    ZT_f1 = compute_ZeroTransfer(R_f1_metric, ZS_f1_metric, order)
    return ZT_acc,ZT_rec,ZT_pre,ZT_f1


def final_performance(results_cl,order):
    R_acc_metric = np.zeros((len(order), len(order)))
    R_rec_metric = np.zeros((len(order), len(order)))
    R_pre_metric = np.zeros((len(order), len(order)))
    R_f1_metric = np.zeros((len(order), len(order)))
    # convert to metric
    for idx, dataname in enumerate(order):
        CL_acc = results_cl[dataname]['Accuracy']
        CL_rec = results_cl[dataname]['Recall']
        CL_pre = results_cl[dataname]['Precision']
        CL_f1 = results_cl[dataname]['F1']
        for jdx, dataname_c in enumerate(order):
            R_acc_metric[jdx, idx] = CL_acc[dataname_c]
            R_rec_metric[jdx, idx] = CL_rec[dataname_c]
            R_pre_metric[jdx, idx] = CL_pre[dataname_c]
            R_f1_metric[jdx, idx] = CL_f1[dataname_c]
    FP_acc = R_acc_metric.sum(axis=0)[-1] / len(order)
    FP_rec = R_rec_metric.sum(axis=0)[-1] / len(order)
    FP_pre = R_pre_metric.sum(axis=0)[-1] / len(order)
    FP_f1 = R_f1_metric.sum(axis=0)[-1] / len(order)
    return FP_acc,FP_rec,FP_pre,FP_f1


def evalution_CL():
    order= [
            "Spirit",
  "BGL",
  "Thunderbird",
  "HDFS"
    ]
    order_num = len(order)
    # order.reverse()
    methods = [
        # 'sequential_fine-tuning',
        # 'incremental_joint_learning',
        # 'sequential_keep_head',
        # 'sequential_keep_body',
        # 'sequential_keep_body_wo_1_9',
        # 'sequential_ewc',
        # 'sequential_er',
        # 'simple_knowledge_distill',
        # 'hint_knowledge_distill',
        'dual_o10'
    ]
    for method in methods:
        root_path = f'/home/zmj/task/Log_continual_learning_Bigserver/Anomaly_Detection/CSLS_baselines/Dual_prompt/{method}/'
        results = load_results(root_path,order)
        fine_tuning_results = load_results_onefile('/home/zmj/task/Log_continual_learning_Bigserver/Anomaly_Detection/CSLS_baselines/prefix_tuning_woCL/metrics.csv')
        zero_shot_results = load_results_onefile('/home/zmj/task/Log_continual_learning_Bigserver/Anomaly_Detection/o1/zero-shot/metrics.csv')
        F_Acc,F_Rec,F_Pre,F_F1 = Forgetting(results,order)
        T_Acc,T_Rec,T_Pre,T_F1 = Transfer(results,fine_tuning_results,order)
        ZT_Acc,ZT_Rec,ZT_Pre,ZT_F1 = zero_Transfer(results,zero_shot_results,order)
        FP_Acc,FP_Rec,FP_Pre,FP_F1 = final_performance(results,order)
        print(f'Forgetting Accuracy: {F_Acc} Recall: {F_Rec} Precision: {F_Pre} F1: {F_F1}')
        print(f'Transfer Accuracy: {T_Acc} Recall: {T_Rec} Precision: {T_Pre} F1: {T_F1}')
        print(f'Zero-shot Transfer Accuracy: {ZT_Acc} Recall: {ZT_Rec} Precision: {ZT_Pre} F1: {ZT_F1}')
        print(f'Final performance Accuracy: {FP_Acc} Recall: {FP_Rec} Precision: {FP_Pre} F1: {FP_F1}')
        with open(root_path+f'evalution_results','w+') as file:
            file.write(f'Forgetting Accuracy: {F_Acc} Recall: {F_Rec} Precision: {F_Pre} F1: {F_F1} \n')
            file.write(f'Transfer Accuracy: {T_Acc} Recall: {T_Rec} Precision: {T_Pre} F1: {T_F1} \n')
            file.write(f'Zero-shot Transfer Accuracy: {ZT_Acc} Recall: {ZT_Rec} Precision: {ZT_Pre} F1: {ZT_F1} \n')
            file.write(f'Final performance Accuracy: {FP_Acc} Recall: {FP_Rec} Precision: {FP_Pre} F1: {FP_F1} \n')


if __name__ == '__main__':
    evalution_CL()