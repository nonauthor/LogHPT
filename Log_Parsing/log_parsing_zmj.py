import pandas as pd
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
import hashlib
import os
import time
from logppt.data import CustomDataCollator
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def map_template_xlnet(tokenizer, c, t, mode="prompt-tuning"):
    val_token = tokenizer.convert_tokens_to_ids('i-val') if mode == "prompt-tuning" else 1
    tokens = tokenizer.convert_ids_to_tokens(c)
    # print(tokens)
    res = [" "]
    # print(t)
    for i in range(1, len(c)):
        if c[i] == tokenizer.sep_token_id:
            break
        if c[i] in tokenizer.all_special_ids:
            continue
        if t[i] < val_token:
            res.append(tokens[i])
        else:
            if "▁" in tokens[i]:
                if "<*>" not in res[-1]:
                    # print(tokens[i])
                    res.append("▁<*>")
            elif "<*>" not in res[-1]:
                res.append("<*>")
    r = "".join(res)
    r = r.replace("▁", " ")
    return r.strip()


def map_template_gpt2(tokenizer, c, t, mode="prompt-tuning"):
    val_token = tokenizer.convert_tokens_to_ids('i-val') if mode == "prompt-tuning" else 1
    tokens = tokenizer.convert_ids_to_tokens(c)
    # print(tokens)
    res = [" "]
    # print(t)
    for i in range(1, len(c)):
        if c[i] == tokenizer.pad_token_id:
            continue
        if c[i] == tokenizer.sep_token_id:
            break
        if t[i] < val_token:
            res.append(tokens[i])
        else:
            if "Ġ" in tokens[i]:
                if "<*>" not in res[-1]:
                    # print(tokens[i])
                    res.append("Ġ<*>")
            elif "<*>" not in res[-1]:
                res.append("<*>")
    r = "".join(res)
    r = r.replace("Ġ", " ")
    return r.strip()


def map_template_bert(tokenizer, c, t, m, mode="prompt-tuning"):
    val_token = tokenizer.convert_tokens_to_ids('i-val') if mode == "prompt-tuning" else 1
    tokens = tokenizer.convert_ids_to_tokens(c)
    # print(tokens)
    res = [""]
    for i in range(1, len(c)):
        if c[i] == tokenizer.sep_token_id:
            break
        if t[i] < val_token:
            if "##" not in tokens[i]:
                if m[0] == " ":
                    res.append(" ")
                    m = m.lstrip()
                res.append(tokens[i])
                m = m[len(tokens[i]):]
            else:
                res.append(tokens[i][2:])
                m = m[len(tokens[i][2:]):]
        else:
            if "##" not in tokens[i]:
                if "<*>" not in res[-1]:
                    if m[0] == " ":
                        res.append(" ")
                        m = m.lstrip()
                    res.append("<*>")
                m = m[len(tokens[i]):]
            elif "<*>" not in res[-1]:
                res.append("<*>")
                m = m[len(tokens[i][2:]):]
            else:
                m = m[len(tokens[i][2:]):]
    r = "".join(res)
    return r.strip()


def map_template_roberta(tokenizer, c, t, mode="prompt-tuning"):
    val_token = tokenizer.convert_tokens_to_ids('i-val') if mode == "prompt-tuning" else 1
    tokens = tokenizer.convert_ids_to_tokens(c)
    # print(tokens)
    res = [" "]
    # print(t)
    for i in range(1, len(c)):
        if c[i] == tokenizer.sep_token_id:
            break
        if t[i] < val_token:
            res.append(tokens[i])
        else:
            if "Ġ" in tokens[i]:
                if "<*>" not in res[-1]:
                    # print(tokens[i])
                    res.append("Ġ<*>")
            elif "<*>" not in res[-1]:
                res.append("<*>")
    r = "".join(res)
    r = r.replace("Ġ", " ")
    return r.strip()


def template_extraction(tokenizer, model, device, log_file, max_length, model_name='bert', shot=5,
                        dataset_name="BGL", o_dir="outputs", mode="prompt-tuning"):

    model.to(device)
    model.eval()

    t0 = time.time()

    def tokenize_and_align_labels(examples):
        examples['Content'] = [" ".join(x.split()) for x in examples['Content']]
        tokenized_inputs = tokenizer(
            examples['Content'],
            max_length=256,
            padding=False,
            truncation=True,
            # We use this argument because the texts in our dataset are lists of words (with a label for each word).
            is_split_into_words=False,
        )
        return tokenized_inputs

    dataset = load_dataset('csv', data_files=log_file)
    remove_columns = list(dataset['train'].features.keys())
    remove_columns.remove("LineId")
    test_dataset = dataset.map(
        tokenize_and_align_labels,
        batched=True,
        remove_columns=remove_columns,
        desc="Running tokenizer on dataset",
    )
    test_dataset = test_dataset['train']
    data_collator = CustomDataCollator(
        tokenizer, pad_to_multiple_of=None
    )
    test_loader = DataLoader(test_dataset, collate_fn=data_collator, batch_size=32, pin_memory=True)
    events = []
    # print(tokenizer.eos_token_id)
    end_token = tokenizer.sep_token_id
    if end_token is None:
        end_token = tokenizer.eos_token_id
    # model, test_loader = accelerator.prepare(
    #     model, test_loader
    # )

    for batch in tqdm(test_loader, desc='Parsing'):
        line_id = batch.pop("LineId")
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch, output_hidden_states=True)
        # print(batch)
        predictions = outputs.logits.argmax(dim=-1)

        # predictions_gathered = accelerator.gather(predictions)
        res = predictions.detach().cpu().clone().tolist()
        inp = batch['input_ids'].detach().cpu().clone().tolist()
        # print(inp[1])
        for i in range(len(inp)):
            try:
                p = inp[i].index(end_token) + 1
            except Exception as _:
                p = len(inp[i])
            res[i] = res[i][:p]
            inp[i] = inp[i][:p]
            events.append((inp[i], res[i], line_id[i]))

    df = pd.read_csv(log_file)
    content = df['Content'].tolist()
    if 'roberta' in model_name:
        event_templates = [(map_template_roberta(tokenizer, x[0], x[1], mode=mode), x[2]) for i, x in enumerate(events)]
        event_list = [""] * len(test_dataset)
        for (e, idx) in event_templates:
            event_list[int(idx) - 1] = e.strip()
    elif 'bert' in model_name:
        event_templates = [map_template_bert(tokenizer, x[0], x[1], content[i], mode=mode) for i, x in
                           enumerate(events)]
        event_list = [x.strip() for x in event_templates]
    elif 'xlnet' in model_name:
        event_templates = [(map_template_xlnet(tokenizer, x[0], x[1], mode=mode), x[2]) for i, x in enumerate(events)]
        event_list = [""] * len(test_dataset)
        for (e, idx) in event_templates:
            event_list[int(idx) - 1] = e.strip()
    elif 'gpt2' in model_name:
        event_templates = [(map_template_gpt2(tokenizer, x[0], x[1], mode=mode), x[2]) for i, x in enumerate(events)]
        event_list = [""] * len(test_dataset)
        for (e, idx) in event_templates:
            event_list[int(idx) - 1] = e.strip()
    else:
        raise NotImplementedError

    templates = {}
    for i in range(len(test_dataset)):
        event_id = hashlib.md5(event_list[i].encode('utf-8')).hexdigest()
        df.at[i, 'EventTemplate'] = event_list[i]
        df.at[i, 'EventId'] = event_id
        if event_id not in templates.keys():
            templates[event_id] = {}
            templates[event_id]['EventTemplate'] = event_list[i]
            templates[event_id]['Count'] = 1
        else:
            templates[event_id]['Count'] += 1
    print("parsing time:", time.time() - t0)
    os.makedirs(os.path.join(o_dir, f"{shot}shot"), exist_ok=True)
    pd.DataFrame.from_dict(templates, orient='index').to_csv(
        os.path.join(os.path.join(o_dir, f"{shot}shot", dataset_name + "_2k.log_templates.csv")))
    df.to_csv(os.path.join(o_dir, f"{shot}shot/" + dataset_name + "_2k.log_structured.csv"))


def template_extraction_dual(tokenizer, model_key,model_dual, device, log_file, max_length, model_name='bert', shot=5,
                        dataset_name="BGL", o_dir="outputs", mode="prompt-tuning"):

    model_key.to(device)
    model_key.eval()
    model_dual.to(device)
    model_dual.eval()
    t0 = time.time()

    def tokenize_and_align_labels(examples):
        examples['Content'] = [" ".join(x.split()) for x in examples['Content']]
        tokenized_inputs = tokenizer(
            examples['Content'],
            # max_length=480,
            padding=False,
            truncation=True,
            # We use this argument because the texts in our dataset are lists of words (with a label for each word).
            is_split_into_words=False,
        )
        return tokenized_inputs

    dataset = load_dataset('csv', data_files=log_file)
    remove_columns = list(dataset['train'].features.keys())
    remove_columns.remove("LineId")
    test_dataset = dataset.map(
        tokenize_and_align_labels,
        batched=True,
        remove_columns=remove_columns,
        desc="Running tokenizer on dataset",
    )
    test_dataset = test_dataset['train']
    data_collator = CustomDataCollator(
        tokenizer, pad_to_multiple_of=None
    )
    test_loader = DataLoader(test_dataset, collate_fn=data_collator, batch_size=100, pin_memory=True)
    events = []
    # print(tokenizer.eos_token_id)
    end_token = tokenizer.sep_token_id
    if end_token is None:
        end_token = tokenizer.eos_token_id
    # model, test_loader = accelerator.prepare(
    #     model, test_loader
    # )

    for batch in tqdm(test_loader, desc='Parsing'):
        line_id = batch.pop("LineId")
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs_key = model_key(**batch)
            cls_features = outputs_key[0][:, 0, :]
            outputs = model_dual(**batch, cls_features=cls_features)
        # print(batch)
        predictions = outputs.logits.argmax(dim=-1)

        # predictions_gathered = accelerator.gather(predictions)
        res = predictions.detach().cpu().clone().tolist()
        inp = batch['input_ids'].detach().cpu().clone().tolist()
        # print(inp[1])
        for i in range(len(inp)):
            try:
                p = inp[i].index(end_token) + 1
            except Exception as _:
                p = len(inp[i])
            res[i] = res[i][:p]
            inp[i] = inp[i][:p]
            events.append((inp[i], res[i], line_id[i]))

    df = pd.read_csv(log_file)
    content = df['Content'].tolist()
    if 'roberta' in model_name:
        event_templates = [(map_template_roberta(tokenizer, x[0], x[1], mode=mode), x[2]) for i, x in enumerate(events)]
        event_list = [""] * len(test_dataset)
        for (e, idx) in event_templates:
            event_list[int(idx) - 1] = e.strip()
    elif 'bert' in model_name:
        event_templates = [map_template_bert(tokenizer, x[0], x[1], content[i], mode=mode) for i, x in
                           enumerate(events)]
        event_list = [x.strip() for x in event_templates]
    elif 'xlnet' in model_name:
        event_templates = [(map_template_xlnet(tokenizer, x[0], x[1], mode=mode), x[2]) for i, x in enumerate(events)]
        event_list = [""] * len(test_dataset)
        for (e, idx) in event_templates:
            event_list[int(idx) - 1] = e.strip()
    elif 'gpt2' in model_name:
        event_templates = [(map_template_gpt2(tokenizer, x[0], x[1], mode=mode), x[2]) for i, x in enumerate(events)]
        event_list = [""] * len(test_dataset)
        for (e, idx) in event_templates:
            event_list[int(idx) - 1] = e.strip()
    else:
        raise NotImplementedError

    templates = {}
    for i in range(len(test_dataset)):
        event_id = hashlib.md5(event_list[i].encode('utf-8')).hexdigest()
        df.at[i, 'EventTemplate'] = event_list[i]
        df.at[i, 'EventId'] = event_id
        if event_id not in templates.keys():
            templates[event_id] = {}
            templates[event_id]['EventTemplate'] = event_list[i]
            templates[event_id]['Count'] = 1
        else:
            templates[event_id]['Count'] += 1
    print("parsing time:", time.time() - t0)
    os.makedirs(os.path.join(o_dir, f"{shot}shot"), exist_ok=True)
    pd.DataFrame.from_dict(templates, orient='index').to_csv(
        os.path.join(os.path.join(o_dir, f"{shot}shot", dataset_name + "_2k.log_templates.csv")))
    df.to_csv(os.path.join(o_dir, f"{shot}shot/" + dataset_name + "_2k.log_structured.csv"))

def template_extraction_prompt_selection(type_id,system_id,tokenizer, model_key,model_dual, device, log_file, max_length, model_name='bert', shot=5,
                        dataset_name="BGL", o_dir="outputs", mode="prompt-tuning"):

    model_key.to(device)
    model_key.eval()
    model_dual.to(device)
    model_dual.eval()
    t0 = time.time()

    def tokenize_and_align_labels(examples):
        examples['Content'] = [" ".join(x.split()) for x in examples['Content']]
        tokenized_inputs = tokenizer(
            examples['Content'],
            # max_length=480,
            padding=False,
            truncation=True,
            # We use this argument because the texts in our dataset are lists of words (with a label for each word).
            is_split_into_words=False,
        )
        return tokenized_inputs

    dataset = load_dataset('csv', data_files=log_file)
    remove_columns = list(dataset['train'].features.keys())
    remove_columns.remove("LineId")
    test_dataset = dataset.map(
        tokenize_and_align_labels,
        batched=True,
        remove_columns=remove_columns,
        desc="Running tokenizer on dataset",
    )
    test_dataset = test_dataset['train']
    data_collator = CustomDataCollator(
        tokenizer, pad_to_multiple_of=None
    )
    test_loader = DataLoader(test_dataset, collate_fn=data_collator, batch_size=100, pin_memory=True)
    events = []
    # print(tokenizer.eos_token_id)
    end_token = tokenizer.sep_token_id
    if end_token is None:
        end_token = tokenizer.eos_token_id
    # model, test_loader = accelerator.prepare(
    #     model, test_loader
    # )
    type_id_true_list = [type_id] * 1000
    system_id_true_list = [system_id] * 1000
    type_id_pre_list = []
    system_id_pre_list = []
    for batch in tqdm(test_loader, desc='Parsing'):
        line_id = batch.pop("LineId")
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs_key = model_key(**batch)
            cls_features = outputs_key[0][:, 0, :]
            outputs,type_id_pre,system_id_pre = model_dual(**batch, cls_features=cls_features)
            type_id_pre = type_id_pre.squeeze().detach().cpu().numpy().tolist()
            type_id_pre_list.extend(type_id_pre)
            system_id_pre = system_id_pre.squeeze().detach().cpu().numpy().tolist()
            system_id_pre_list.extend(system_id_pre)
        # print(batch)
        predictions = outputs.logits.argmax(dim=-1)

        # predictions_gathered = accelerator.gather(predictions)
        res = predictions.detach().cpu().clone().tolist()
        inp = batch['input_ids'].detach().cpu().clone().tolist()
        # print(inp[1])
        for i in range(len(inp)):
            try:
                p = inp[i].index(end_token) + 1
            except Exception as _:
                p = len(inp[i])
            res[i] = res[i][:p]
            inp[i] = inp[i][:p]
            events.append((inp[i], res[i], line_id[i]))

    acc_t = accuracy_score(type_id_true_list, type_id_pre_list)
    recall_t = recall_score(type_id_true_list, type_id_pre_list, average='macro')
    p_t = precision_score(type_id_true_list, type_id_pre_list, average='macro')
    f1_t = f1_score(type_id_true_list, type_id_pre_list, average='macro')

    acc_s = accuracy_score(system_id_true_list, system_id_pre_list)
    recall_s = recall_score(system_id_true_list, system_id_pre_list, average='macro')
    p_s = precision_score(system_id_true_list, system_id_pre_list, average='macro')
    f1_s = f1_score(system_id_true_list, system_id_pre_list, average='macro')
    eval_metric_ts = {'acc_t': acc_t, 'recall_t': recall_t, 'prec_t': p_t, 'f1_t': f1_t, 'acc_s': acc_s,
                      'recall_s': recall_s, 'prec_s': p_s, 'f1_s': f1_s}

    df = pd.read_csv(log_file)
    content = df['Content'].tolist()
    if 'roberta' in model_name:
        event_templates = [(map_template_roberta(tokenizer, x[0], x[1], mode=mode), x[2]) for i, x in enumerate(events)]
        event_list = [""] * len(test_dataset)
        for (e, idx) in event_templates:
            event_list[int(idx) - 1] = e.strip()
    elif 'bert' in model_name:
        event_templates = [map_template_bert(tokenizer, x[0], x[1], content[i], mode=mode) for i, x in
                           enumerate(events)]
        event_list = [x.strip() for x in event_templates]
    elif 'xlnet' in model_name:
        event_templates = [(map_template_xlnet(tokenizer, x[0], x[1], mode=mode), x[2]) for i, x in enumerate(events)]
        event_list = [""] * len(test_dataset)
        for (e, idx) in event_templates:
            event_list[int(idx) - 1] = e.strip()
    elif 'gpt2' in model_name:
        event_templates = [(map_template_gpt2(tokenizer, x[0], x[1], mode=mode), x[2]) for i, x in enumerate(events)]
        event_list = [""] * len(test_dataset)
        for (e, idx) in event_templates:
            event_list[int(idx) - 1] = e.strip()
    else:
        raise NotImplementedError

    templates = {}
    for i in range(len(test_dataset)):
        event_id = hashlib.md5(event_list[i].encode('utf-8')).hexdigest()
        df.at[i, 'EventTemplate'] = event_list[i]
        df.at[i, 'EventId'] = event_id
        if event_id not in templates.keys():
            templates[event_id] = {}
            templates[event_id]['EventTemplate'] = event_list[i]
            templates[event_id]['Count'] = 1
        else:
            templates[event_id]['Count'] += 1
    print("parsing time:", time.time() - t0)
    os.makedirs(os.path.join(o_dir, f"{shot}shot"), exist_ok=True)
    pd.DataFrame.from_dict(templates, orient='index').to_csv(
        os.path.join(os.path.join(o_dir, f"{shot}shot", dataset_name + "_2k.log_templates.csv")))
    df.to_csv(os.path.join(o_dir, f"{shot}shot/" + dataset_name + "_2k.log_structured.csv"))

    return eval_metric_ts,type_id_pre_list,system_id_pre_list