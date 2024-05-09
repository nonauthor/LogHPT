import datasets
import torch
from torch.utils.data import Dataset,DataLoader
import json
import re
import random
import pandas as pd
import copy
import transformers
from transformers import DataCollatorForTokenClassification
from transformers import DataCollatorWithPadding
from transformers import (
    # HfArgumentParser,
    # AutoConfig,
    default_data_collator,
    # get_scheduler,
    # set_seed,
    RobertaForTokenClassification,
    RobertaForSequenceClassification,
    AutoTokenizer,
    RobertaTokenizer
)
random.seed(2023)
torch.cuda.manual_seed(2023)



def save_json(data, file):
    dict_json = json.dumps(data,indent=1)
    with open(file,'w+',newline='\n') as file:
        file.write(dict_json)

def read_json(file):
    with open(file,'r+') as file:
        content = file.read()
        content = json.loads(content)
        return content


class CustomDataCollator(DataCollatorForTokenClassification):
    def __call__(self, features):

        ori_labels = [feature['labels'] for feature in features] if 'labels' in features[0].keys() else None
        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            # max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            # Conversion to tensors will fail if we have labels as they are not of the same length yet.
            return_tensors=None,
        )

        sequence_length = torch.tensor(batch["input_ids"]).shape[1]
        padding_side = self.tokenizer.padding_side
        if padding_side == "right":
            batch['labels'] = [label + [self.label_pad_token_id] * (sequence_length - len(label)) for label in
                                   ori_labels]
        else:
            batch["labels"] = [[self.label_pad_token_id] * (sequence_length - len(label)) + label for label in
                                   ori_labels]
        batch = {k: torch.tensor(v, dtype=torch.int64) for k, v in batch.items()}
        return batch

class CustomDataCollator_AD(DataCollatorForTokenClassification):
    def __call__(self, features):

        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            # max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            # Conversion to tensors will fail if we have labels as they are not of the same length yet.
            return_tensors=None,
        )


        batch = {k: torch.tensor(v, dtype=torch.int64) for k, v in batch.items()}
        return batch

def get_parameter_list(s, template_regex):
    """
    :param s: log message
    :param template_regex: template regex with <*> indicates parameters
    :return: list of parameters
    """
    # template_regex = re.sub(r"<.{1,5}>", "<*>", template_regex)
    if "<*>" not in template_regex:
        return []
    template_regex = re.sub(r'([^A-Za-z0-9])', r'\\\1', template_regex)
    template_regex = re.sub(r'\\ +', r'\\s+', template_regex)
    template_regex = "^" + template_regex.replace("\<\*\>", "(.*?)") + "$"
    parameter_list = re.findall(template_regex, s)
    parameter_list = parameter_list[0] if parameter_list else ()
    parameter_list = list(parameter_list) if isinstance(parameter_list, tuple) else [parameter_list]
    return parameter_list


def tokenize_and_align_labels(examples,tokenizer,max_length,padding,label_to_id):

        examples['text'] = " ".join(examples['text'].strip().split())
        tokenized_inputs = tokenizer(
            examples['text'],
            max_length=max_length,
            padding=padding,
            truncation=True,
            is_split_into_words=False,
        )

        t_token = "i-val"
        label = examples['label']
        content = examples['text']
        label = " ".join(label.strip().split())
        variable_list = get_parameter_list(content, label)
        input_ids = tokenized_inputs.input_ids
        input_tokens = tokenizer.convert_ids_to_tokens(input_ids)

        label_ids = []

        processing_variable = False
        variable_token = ""
        input_tokens = [tokenizer.convert_tokens_to_string([x]) for x in input_tokens]
        # pos = 0
        for ii, (input_idx, input_token) in enumerate(zip(input_ids, input_tokens)):
            if input_idx in tokenizer.all_special_ids:
                label_ids.append(-100)
                continue
            # Set target token for the first token of each word.
            if (label[:3] == "<*>" or label[:len(input_token.strip())] != input_token.strip()) \
                    and processing_variable is False:
                processing_variable = True
                variable_token = variable_list.pop(0).strip()
                pos = label.find("<*>")
                label = label[label.find("<*>") + 3:].strip()
                input_token = input_token.strip()[pos:]
            if processing_variable:
                input_token = input_token.strip()
                if input_token == variable_token[:len(input_token)]:
                    label_ids.append(label_to_id[t_token])
                    variable_token = variable_token[len(input_token):].strip()
                    # print(variable_token, "+++", input_token)
                elif len(input_token) > len(variable_token):
                    label_ids.append(label_to_id[t_token])
                    label = label[len(input_token) - len(variable_token):].strip()
                    variable_token = ""
                else:
                    raise ValueError(f"error at {variable_token} ---- {input_token}")
                if len(variable_token) == 0:
                    processing_variable = False
            else:
                input_token = input_token.strip()
                if input_token == label[:len(input_token)]:
                    label_ids.append(label_to_id['o'])
                    label = label[len(input_token):].strip()
                else:
                    raise ValueError(f"error at {content} ---- {input_token}")

        tokenized_inputs['labels'] = label_ids
        return tokenized_inputs



class multi_task_dataset(Dataset):
    def __init__(self,root_path,dataname_list:list,tokenizer,label_to_id,max_length=256,padding=False,train_eval_test='train',shot=1000):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = padding
        self.label_to_id = label_to_id

        self.datas = []

        for dataname in dataname_list:
            if train_eval_test=='train':
                path = root_path+f'\{dataname}\\{shot}shot\\1.json'
            else:
                path = root_path+f'\{dataname}\\test.json'
            with open(path,'r') as file:
                for line in file:
                    example = json.loads(line)
                    self.datas.append(example)

        if train_eval_test =='eval':
            random.shuffle(self.datas)
            self.datas = self.datas[:200]

        # print('finish initial')


    def __len__(self):
        return len(self.datas)

    def __getitem__(self, item):
        examples = self.datas[item]

        model_input=tokenize_and_align_labels(examples,self.tokenizer,self.max_length,self.padding,self.label_to_id)

        return model_input.data


class multi_task_dataset_buffer(Dataset):
    def __init__(self,root_path,dataname_list:list,tokenizer,label_to_id,max_length=256,padding=False,train_eval_test='train',shot=1000):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = padding
        self.label_to_id = label_to_id
        self.datas = []

        for dataname in dataname_list:
            if train_eval_test=='train':
                path = root_path+f'\{dataname}\\{shot}shot\\1.json'
            else:
                path = root_path+f'\{dataname}\\test.json'
            with open(path,'r') as file:
                for line in file:
                    example = json.loads(line)
                    self.datas.append(example)

        if train_eval_test =='eval':
            random.shuffle(self.datas)
            self.datas = self.datas[:200]


    # def buffer(self,sample_K):
    #     random.shuffle(self.datas)
    #     if sample_K <= len(self.datas):
    #         self.buffer_data = self.datas[:sample_K]
    #     else:
    #         self.buffer_data = self.datas

    def get_buffer(self):
        self.buffer_data = copy.deepcopy(self.datas)

    def add_buffer(self,buffer_data):
        self.datas.extend(buffer_data)


    def __len__(self):
        return len(self.datas)

    def __getitem__(self, item):
        examples = self.datas[item]

        model_input=tokenize_and_align_labels(examples,self.tokenizer,self.max_length,self.padding,self.label_to_id)

        return model_input.data#['input_ids'],model_input['attention_mask'],model_input['ori_labels']

class multi_task_dataset_AD(Dataset):
    def __init__(self,root_path,dataname_list:list,tokenizer,max_length=256,padding=False,train=True,use_log=True,train_eval_test='train',):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length=max_length
        self.padding = padding
        self.datas = []
        for dataname in dataname_list:
            if train==True:
                path = root_path+f'\\{dataname}\\train.json'
            else:
                path = root_path+f'\\{dataname}\\test.json'
            data = read_json(path)
            if use_log==True:
                for i in data:
                    self.datas.append([i[0],i[2]])
            else:
                for i in data:
                    self.datas.append([i[1],i[2]])

        if train_eval_test =='eval':
            random.shuffle(self.datas)
            self.datas = self.datas[:200]

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):
        content = self.datas[index][0]
        data_encoding = self.tokenizer(
            self.datas[index][0],
            max_length=self.max_length,
            padding=self.padding,
            truncation=True,
            is_split_into_words=False,
        )
        label = [self.datas[index][1]]
        data_encoding['labels'] = label
        return data_encoding.data

class multi_task_dataset_buffer_AD(Dataset):
    def __init__(self,root_path,dataname_list:list,tokenizer,max_length=256,padding=False,train=True,use_log=True,train_eval_test='train',):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = padding
        self.datas = []

        for dataname in dataname_list:
            if train == True:
                path = root_path + f'\\{dataname}\\train.json'
            else:
                path = root_path + f'\\{dataname}\\test.json'
            data = read_json(path)
            if use_log == True:
                for i in data:
                    self.datas.append([i[0], i[2]])
            else:
                for i in data:
                    self.datas.append([i[1], i[2]])

        if train_eval_test =='eval':
            random.shuffle(self.datas)
            self.datas = self.datas[:200]


    # def buffer(self,sample_K):
    #     random.shuffle(self.datas)
    #     if sample_K <= len(self.datas):
    #         self.buffer_data = self.datas[:sample_K]
    #     else:
    #         self.buffer_data = self.datas

    def get_buffer(self):
        self.buffer_data = copy.deepcopy(self.datas)

    def add_buffer(self,buffer_data):
        self.datas.extend(buffer_data)


    def __len__(self):
        return len(self.datas)

    def __getitem__(self, item):
        examples = self.datas[item]

        model_input=tokenize_and_align_labels(examples,self.tokenizer,self.max_length,self.padding,self.label_to_id)

        return model_input.data#['input_ids'],model_input['attention_mask'],model_input['ori_labels']

if __name__ == '__main__':
    model_path = "D:\ZMJ\Local-model\\roberta-base"
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, add_prefix_space=True, do_lower_case=False)
    model = RobertaForTokenClassification.from_pretrained(model_path)
    tokenizer.model_max_length = model.config.max_position_embeddings - 2
    label_to_id = {'o': 0, 'i-val': 1}
    logdata = multi_task_dataset('D:\ZMJ\pythonProject\Log_continual_learning\datasets',['Apache'],tokenizer,label_to_id)
    data_collator = CustomDataCollator(
        tokenizer, pad_to_multiple_of=None
    )
    train_loader = DataLoader(logdata,collate_fn=data_collator,batch_size=8)
    for batch in train_loader:
        print(batch)