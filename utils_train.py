import torch
import json
import math
import random
import os
import numpy as np

def text2token(tokenizer, text):
    text_list = tokenizer.tokenize(text)
    token_list = tokenizer.convert_tokens_to_ids(text_list)
    token = torch.tensor(token_list)[None, :]
    return token

def list2token(tokenizer, text_list, max_length):
    token_list = tokenizer.convert_tokens_to_ids(text_list)
    token_list_padded = token_list + [0] * (max_length - len(token_list))
    token = torch.tensor(token_list_padded)[None, :]
    return token

def read_data(json_file, max_length=0):
    data_list = []
    with open(json_file) as f:
        for jsonObj in f:
            record = json.loads(jsonObj)
            if max_length != 0:
                if len(record['words']) > max_length:
                    record['words'] = record['words'][:max_length]
                    record['ner'] = record['ner'][:max_length]
            data_list.append(record)
    return data_list

def cat2digit(classes, cat_text, max_length):
    label_digit = [classes.get(item, item) for item in cat_text]
    label_digit_padded = label_digit + [len(classes)] * (max_length - len(label_digit))
    att_mask = [1] * len(label_digit) + [0] * (max_length - len(label_digit))
    return torch.tensor(label_digit_padded), torch.tensor(att_mask)

def to_batches(x, batch_size):
    num_batches = math.ceil(x.size()[0] / batch_size)
    return [x[batch_size * y: batch_size * (y+1),:] for y in range(num_batches)]

def accuracy(index_other, index_pad, y_pred, y):
    indices = ((index_other < y) & (y < index_pad)).nonzero(as_tuple=True)  # words with entity
    _, predicted_classes = y_pred[indices[0], :, indices[1]].max(dim=1)
    true_classes = y[indices[0], indices[1]]
    accuracy = torch.eq(predicted_classes, true_classes).sum() / true_classes.shape[0]
    return accuracy, predicted_classes, true_classes

def preprocess(json_file, classes, tokenizer, n_data, batch_size, max_length, test=False):
    if n_data == 0:
        data_list = read_data(json_file, max_length)
    else:
        data_list = read_data(json_file, max_length)[:n_data]
    token_tensors_all_list = [list2token(tokenizer, d['words'], max_length) for d in data_list]
    data = torch.cat(token_tensors_all_list, dim=0)
    if test:
        batch_size = data.shape[0]
    data_batches = to_batches(data, batch_size)
    target_tensors_all_list = [cat2digit(classes, d['ner'], max_length)[0] for d in data_list]
    target = torch.stack(target_tensors_all_list, dim=0)
    target_batches = to_batches(target, batch_size)
    att_mask_all_list = [cat2digit(classes, d['ner'], max_length)[1] for d in data_list]
    att_mask = torch.stack(att_mask_all_list, dim=0)
    att_mask_batches = to_batches(att_mask, batch_size)
    if test:
        return data_batches[0], target_batches[0], att_mask_batches[0], data_list
    else:
        c = list(zip(data_batches, target_batches, att_mask_batches))
        random.shuffle(c)
        data_batches, target_batches, att_mask_batches = zip(*c)
        return data_batches, target_batches, att_mask_batches

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    