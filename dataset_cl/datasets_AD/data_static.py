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

ori_data_path = 'D:\ZMJ\pythonProject\Log_continual_learning\Anomaly_Detection\data_AD'

files_SmallParse = [
            'BGL',
            'HDFS',
            'Spirit',
            'Thunderbird']
for data_name in files_SmallParse:
    path_train = ori_data_path+ f'\{data_name}\\train.json'
    path_test = ori_data_path + f'\{data_name}\\test.json'
    train_data = read_json(path_train)
    test_data = read_json(path_test)
    train_label = {'0':0,'1':0}
    test_label = {'0':0,'1':0}
    for i in train_data:
        if i[-1]==0:
            train_label['0']+=1
        else:
            train_label['1']+=1
    for j in test_data:
        if j[-1]==0:
            test_label['0']+=1
        else:
            test_label['1']+=1
    print(data_name)
    print(train_label)
    print(test_label)
    print('+++++++++++++++++')