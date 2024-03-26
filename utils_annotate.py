import re
import json
import random
import os
import utils_train

def load_json(file):
    with open(file, "r") as f:
        data = json.loads(f.read())
    return data

def split_para(para):
    return re.findall(r"[\w']+|[-.,!?;/\(\)\[\]]", para)

def to_dict(word_list, categories):
    return {"words": word_list, "ner": categories}

def annotate(para, class_list, name=None):
    print(para)
    word_list = split_para(para)
    categories = []
    
    for i, word in enumerate(word_list):
        c = input(f"What's the category for '{word}'? ")
        try:
            categories.append(class_list[int(c)])
        except:
            c = input(f"What's the category for '{word}'? ")
        if (i + 1) % 10 == 0:
            print()
            print(' '.join(word_list[i:]))
    if name != None:
        ner_dict = to_dict(word_list, categories)
        with open(f'individual_ner/{name}.json', 'w') as f:
            json.dump(ner_dict, f)
    return word_list, categories

def reannotate(file, class_list):
    record = load_json(file)
    word_list = record['words']
    categories = []
    for i, word in enumerate(word_list):
        c = input(f"What's the category for '{word}'? ")
        try:
            categories.append(class_list[int(c)])
        except:
            c = input(f"What's the category for '{word}'? ")
        if i % 10 == 0:
            print()
            print(' '.join(word_list[i:]))
    ner_dict = to_dict(word_list, categories)
    with open(file, 'w') as f:
        json.dump(ner_dict, f)
    return word_list, categories

def combine_records(folder):
    data_list = []
    for jsonfile in os.listdir(folder):
        with open(folder + '/'+ jsonfile) as f:
            data_list.append(json.load(f))
    return data_list

def json_train_test(folder, data_list, n_test, shuffle=True):
    if shuffle:
        random.shuffle(data_list)
    json_train = data_list[:-n_test]
    json_test = data_list[-n_test:]
    with open(f'{folder}/data_train.json', 'w') as f_train:
        f_train.write('')
    with open(f'{folder}/data_train.json', 'a') as f_train:
        for rec in json_train:
            json.dump(rec, f_train)
            f_train.write('\n')
    with open(f'{folder}/data_test.json', 'w') as f_test:
        f_test.write('')
    with open(f'{folder}/data_test.json', 'a') as f_test:
        for rec in json_test:
            json.dump(rec, f_test)
            f_test.write('\n')

def check(file, prt=False):
    record = load_json(file)
    try:
        record4check = [(i, word, record['ner'][i]) for i, word in enumerate(record['words'])]
        if prt:
            print(record4check)
        return record4check
    except:
        print("words length: ", len(record['words']))
        print("ner length: ", len(record['ner']))
        return record['words']

def check_combined(file, index, prt=False):
    record = utils_train.read_data(file)[index]
    try:
        record4check = [(i, word, record['ner'][i]) for i, word in enumerate(record['words'])]
        if prt:
            print(record4check)
        return record4check
    except:
        print("words length: ", len(record['words']))
        print("ner length: ", len(record['ner']))

def revise(file, indices, new_class, overwrite=False):
    record = load_json(file)
    for i in indices:
        record['ner'][i] = new_class
    if overwrite:
        with open(file, 'w') as f:
           json.dump(record, f)
    record4check = [(i, word, record['ner'][i]) for i, word in enumerate(record['words'])]
    return record, record4check