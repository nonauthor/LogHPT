from logppt.evaluation.utils.evaluator_main import evaluator
import os
import pandas as pd
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

order_list = read_json('D:\ZMJ\pythonProject\Log_continual_learning\order_generate\order_16.json')
file_list = [
        "Thunderbird",
  "Apache",
  "Mac",
  "Linux",
  "Android",
  "OpenSSH",
  "OpenStack",
  "BGL",
  "Proxifier",
  "Hadoop",
  "Spark",
  "HPC",
  "HDFS",
  "HealthApp",
  "Windows",
  "Zookeeper"
]
# file_list.reverse()
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
    # f'Wo_meta\dual_prefix_o10'
    'Meanprompt\dual_prefix_o6'
    ]
for method in methods:
    print(f'method:{method}')
    for idx,data_name in enumerate(file_list):
        print(f'idx={idx} data_name={data_name}')
        output_dir = f'D:\ZMJ\pythonProject\Log_continual_learning\CSLS\\baselines\{method}\\Num_{idx}_{data_name}\\1000shot'

        input_dir = '.\logs_1k\\'  # The input directory of log file
        benchmark_settings = {
            'HDFS': {
                'log_file': 'HDFS/HDFS_2k.log',
                'log_format': '<Date> <Time> <Pid> <Level> <Component>: <Content>',
            },

            'Hadoop': {
                'log_file': 'Hadoop/Hadoop_2k.log',
                'log_format': '<Date> <Time> <Level> \[<Process>\] <Component>: <Content>',
            },
            #
            'Spark': {
                'log_file': 'Spark/Spark_2k.log',
                'log_format': '<Date> <Time> <Level> <Component>: <Content>',
            },

            'Zookeeper': {
                'log_file': 'Zookeeper/Zookeeper_2k.log',
                'log_format': '<Date> <Time> - <Level>  \[<Node>:<Component>@<Id>\] - <Content>',
            },
            #
            'BGL': {
                'log_file': 'BGL/BGL_2k.log',
                'log_format': '<Label> <Timestamp> <Date> <Node> <Time> <NodeRepeat> <Type> <Component> <Level> <Content>',
            },

            'HPC': {
                'log_file': 'HPC/HPC_2k.log',
                'log_format': '<LogId> <Node> <Component> <State> <Time> <Flag> <Content>',
            },
            #
            'Thunderbird': {
                'log_file': 'Thunderbird/Thunderbird_2k.log',
                'log_format': '<Label> <Timestamp> <Date> <User> <Month> <Day> <Time> <Location> <Component>(\[<PID>\])?: <Content>',
            },

            'Windows': {
                'log_file': 'Windows/Windows_2k.log',
                'log_format': '<Date> <Time>, <Level>                  <Component>    <Content>',
            },

            'Linux': {
                'log_file': 'Linux/Linux_2k.log',
                'log_format': '<Month> <Date> <Time> <Level> <Component>(\[<PID>\])?: <Content>',
            },

            'Android': {
                'log_file': 'Android/Android_2k.log',
                'log_format': '<Date> <Time>  <Pid>  <Tid> <Level> <Component>: <Content>',
            },

            'HealthApp': {
                'log_file': 'HealthApp/HealthApp_2k.log',
                'log_format': '<Time>\|<Component>\|<Pid>\|<Content>',
            },

            'Apache': {
                'log_file': 'Apache/Apache_2k.log',
                'log_format': '\[<Time>\] \[<Level>\] <Content>',
            },

            'Proxifier': {
                'log_file': 'Proxifier/Proxifier_2k.log',
                'log_format': '\[<Time>\] <Program> - <Content>',
            },

            'OpenSSH': {
                'log_file': 'OpenSSH/OpenSSH_2k.log',
                'log_format': '<Date> <Day> <Time> <Component> sshd\[<Pid>\]: <Content>',
            },

            'OpenStack': {
                'log_file': 'OpenStack/OpenStack_2k.log',
                'log_format': '<Logrecord> <Date> <Time> <Pid> <Level> <Component> \[<ADDR>\] <Content>',
            },
            #
            'Mac': {
                'log_file': 'Mac/Mac_2k.log',
                'log_format': '<Month>  <Date> <Time> <User> <Component>\[<PID>\]( \(<Address>\))?: <Content>',
            }
        }

        bechmark_result = []
        avg_f1, avg_acc = 0, 0
        avg_ga, avg_pa, avg_ed = 0, 0, 0
        avg_unseen_pa = 0
        avg_no_unseen = 0
        unseen_datasets = 0
        for dataset, setting in benchmark_settings.items():
            print('\n=== Evaluation on %s ===' % dataset)
            indir = os.path.join(input_dir, os.path.dirname(setting['log_file']))
            log_file = os.path.basename(setting['log_file'])

            try:
                GA, PA, ED, _, _, _, _, _, _, unseen_PA, no_unseen = evaluator(
                    groundtruth=os.path.join(indir, log_file + '_structured_corrected.csv'),
                    parsedresult=os.path.join(output_dir, log_file + '_structured.csv')
                )
                bechmark_result.append([dataset, GA, PA, ED, unseen_PA, no_unseen])  # , _, _, _, _, _, _])
                avg_ga += GA
                avg_pa += PA
                avg_ed += ED
                avg_unseen_pa += unseen_PA
                avg_no_unseen += no_unseen
                if no_unseen > 0:
                    unseen_datasets += 1
            except Exception as _:
                pass

        bechmark_result.append(["Average", avg_ga / len(benchmark_settings.keys()), avg_pa / len(benchmark_settings.keys()),
                                avg_ed / len(benchmark_settings.keys()), avg_unseen_pa / unseen_datasets,
                                avg_no_unseen / unseen_datasets])

        print('\n=== Overall evaluation results ===')
        df_result = pd.DataFrame(bechmark_result,
                                 columns=['Dataset', 'Group Accuracy', 'Parsing Accuracy', 'Edit distance', 'unseen_PA',
                                          'no_unseen'])
        df_result.set_index('Dataset', inplace=True)
        print(df_result)
        df_result.T.to_csv(os.path.join(output_dir, f'Num_{idx}_{data_name}_benchmark_result.csv'))
