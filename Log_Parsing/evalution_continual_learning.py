import json
import pandas as pd
import os
import numpy as np

def load_results(root_path,order,shot):
    all_results = {}
    for idx,data_name in enumerate(order):
        result_path = root_path+f'Num_{idx}_{data_name}\\{shot}shot\\Num_{idx}_{data_name}_benchmark_result.csv'
        df = pd.read_csv(result_path)
        column_names = (df.columns)[1:]
        ga_row = df.iloc[0,1:]
        pa_row = df.iloc[1,1:]
        ed_row = df.iloc[2,1:]
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
        all_results[data_name]={'GA':ga_row,'PA':pa_row,'ED':ed_row}

    return all_results

def load_results_onefile(path):

    df = pd.read_csv(path)
    ga_row = df.iloc[0, 1:]
    pa_row = df.iloc[1, 1:]
    ed_row = df.iloc[2, 1:]
    all_results = {'GA': ga_row, 'PA': pa_row, 'ED': ed_row}
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

    R_GA_metric = np.zeros((len(order),len(order)))
    R_PA_metric = np.zeros((len(order),len(order)))
    R_ED_metric = np.zeros((len(order),len(order)))

    # convert to metric
    for idx, dataname in enumerate(order):
        CL_ga = results_cl[dataname]['GA']
        CL_pa = results_cl[dataname]['PA']
        CL_ed = results_cl[dataname]['ED']
        for jdx, dataname_c in enumerate(order):
            R_GA_metric[jdx,idx] = CL_ga[dataname_c]
            R_PA_metric[jdx,idx] = CL_pa[dataname_c]
            R_ED_metric[jdx,idx] = CL_ed[dataname_c]

    F_GA = compute_forget(R_GA_metric,order)
    F_PA = compute_forget(R_PA_metric,order)
    F_ED = compute_forget(R_ED_metric,order)

    return F_GA,F_PA,F_ED

def compute_Transfer(CL,FT,name):
    R_i_i = CL[name]
    R_i = FT[name]
    T = R_i_i - R_i
    return T

def Transfer(results_cl,results_ft,order):
    T_GA_all = []
    T_PA_all = []
    T_ED_all = []
    for i in range(1,len(order)):
        dataname = order[i]
        CL_ga = results_cl[dataname]['GA']
        CL_pa = results_cl[dataname]['PA']
        CL_ed = results_cl[dataname]['ED']

        FT_ga = results_ft['GA']
        FT_pa = results_ft['PA']
        FT_ed = results_ft['ED']

        T_GA = compute_Transfer(CL_ga,FT_ga,dataname)
        T_PA = compute_Transfer(CL_pa,FT_pa,dataname)
        T_ED = compute_Transfer(CL_ed,FT_ed,dataname)

        T_GA_all.append(T_GA)
        T_PA_all.append(T_PA)
        T_ED_all.append(T_ED)

    T_GA_avg = sum(T_GA_all)/(len(order)-1)
    T_PA_avg = sum(T_PA_all)/(len(order)-1)
    T_ED_avg = sum(T_ED_all)/(len(order)-1)
    return T_GA_avg,T_PA_avg,T_ED_avg

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
    R_GA_metric = np.zeros((len(order), len(order)))
    R_PA_metric = np.zeros((len(order), len(order)))
    R_ED_metric = np.zeros((len(order), len(order)))
    ZS_GA_metric = np.zeros(len(order))
    ZS_PA_metric = np.zeros(len(order))
    ZS_ED_metric = np.zeros(len(order))

    # convert to metric
    for idx, dataname in enumerate(order):
        ZS_GA_metric[idx] = results_zs['GA'][dataname]
        ZS_PA_metric[idx] = results_zs['PA'][dataname]
        ZS_ED_metric[idx] = results_zs['ED'][dataname]

        CL_ga = results_cl[dataname]['GA']
        CL_pa = results_cl[dataname]['PA']
        CL_ed = results_cl[dataname]['ED']
        for jdx, dataname_c in enumerate(order):
            R_GA_metric[jdx, idx] = CL_ga[dataname_c]
            R_PA_metric[jdx, idx] = CL_pa[dataname_c]
            R_ED_metric[jdx, idx] = CL_ed[dataname_c]

    ZT_GA = compute_ZeroTransfer(R_GA_metric,ZS_GA_metric,order)
    ZT_PA = compute_ZeroTransfer(R_PA_metric, ZS_PA_metric, order)
    ZT_ED = compute_ZeroTransfer(R_ED_metric, ZS_ED_metric, order)
    return ZT_GA,ZT_PA,ZT_ED


def final_performance(results_cl,order):
    R_GA_metric = np.zeros((len(order), len(order)))
    R_PA_metric = np.zeros((len(order), len(order)))
    R_ED_metric = np.zeros((len(order), len(order)))

    # convert to metric
    for idx, dataname in enumerate(order):
        CL_ga = results_cl[dataname]['GA']
        CL_pa = results_cl[dataname]['PA']
        CL_ed = results_cl[dataname]['ED']
        for jdx, dataname_c in enumerate(order):
            R_GA_metric[jdx, idx] = CL_ga[dataname_c]
            R_PA_metric[jdx, idx] = CL_pa[dataname_c]
            R_ED_metric[jdx, idx] = CL_ed[dataname_c]

    FP_GA = R_GA_metric.sum(axis=0)[-1] / len(order)
    FP_PA = R_PA_metric.sum(axis=0)[-1] / len(order)
    FP_ED = R_ED_metric.sum(axis=0)[-1] / len(order)
    return FP_GA,FP_PA,FP_ED


def evalution_CL():
    order= [
        "Android",
  "Apache",
  "BGL",
  "Hadoop",
  "HDFS",
  "HealthApp",
  "HPC",
  "Linux",
  "Mac",
  "OpenSSH",
  "OpenStack",
  "Proxifier",
  "Spark",
  "Thunderbird",
  "Windows",
  "Zookeeper"
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
    'Meanprompt\dual_prefix_o1'
    ]
    for method in methods:
        root_path = f'D:\ZMJ\pythonProject\Log_continual_learning\CSLS\\baselines\{method}\\'
        results = load_results(root_path,order,1000)
        fine_tuning_results = load_results_onefile('D:\ZMJ\pythonProject\Log_continual_learning\CSLS\\baselines\prompt_tuning_woCL\\1000shot\\benchmark_result.csv')
        # fine_tuning_results = load_results_onefile('D:\ZMJ\pythonProject\Log_continual_learning\\fine_tuning_zmj_1k\\fine_tuning_woCL\\1000shot\\benchmark_result.csv')
        zero_shot_results = load_results_onefile('D:\ZMJ\pythonProject\Log_continual_learning\\fine_tuning_zmj_1k\zero-shot\\1000shot\\benchmark_result.csv')
        F_GA,F_PA,F_ED = Forgetting(results,order)
        T_GA,T_PA,T_ED = Transfer(results,fine_tuning_results,order)
        ZT_GA,ZT_PA,ZT_ED = zero_Transfer(results,zero_shot_results,order)
        FP_GA,FP_PA,FP_ED = final_performance(results,order)
        print(f'Forgetting GA: {F_GA} PA: {F_PA} ED: {F_ED}')
        print(f'Transfer GA: {T_GA} PA: {T_PA} ED: {T_ED}')
        print(f'Zero-shot Transfer GA: {ZT_GA} PA: {ZT_PA} ED: {ZT_ED}')
        print(f'Final performance GA: {FP_GA} PA: {FP_PA} ED: {FP_ED}')
        with open(root_path+f'evalution_results','w+') as file:
            file.write(f'Forgetting GA: {F_GA} PA: {F_PA} ED: {F_ED} \n')
            file.write(f'Transfer GA: {T_GA} PA: {T_PA} ED: {T_ED} \n')
            file.write(f'Zero-shot Transfer GA: {ZT_GA} PA: {ZT_PA} ED: {ZT_ED} \n')
            file.write(f'Final performance GA: {FP_GA} PA: {FP_PA} ED: {FP_ED} \n')


if __name__ == '__main__':
    evalution_CL()