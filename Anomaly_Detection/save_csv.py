import pandas as pd

# 定义字典
metrics = {'BGL': {'acc': 0.985, 'recall': 0.9032258064516129, 'prec': 1.0, 'f1': 0.9491525423728813},

           'Spirit': {'acc': 0.985, 'recall': 0.9032258064516129, 'prec': 1.0, 'f1': 0.9491525423728813},
'HDFS': {'acc': 0.985, 'recall': 0.9032258064516129, 'prec': 1.0, 'f1': 0.9491525423728813},
           'Thunderbird': {'acc': 0.985, 'recall': 0.9032258064516129, 'prec': 1.0, 'f1': 0.9491525423728813}}

sorted_dict_by_key = dict(sorted(metrics.items()))
# 将字典转换为DataFrame
df = pd.DataFrame(sorted_dict_by_key)



# 写入CSV文件
df.to_csv('metrics.csv')