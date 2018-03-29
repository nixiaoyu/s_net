# coding=utf-8

import json
# from nltk.tokenize import WordPunctTokenizer
# import sys
# from Levenshtein import distance
import os

# train_path = 'ms_marco/train_v1.1.json/train_v1.1.json'
# dev_path = 'ms_marco/dev_v1.1.json/dev_v1.1.json'
# test_path = 'ms_marco/test_public_v1.1.json/test_public_v1.1.json'

processed_dir = 'ms_marco_processed'
train_processed_path = 'ms_marco_processed/train_v1.1_processed.json'
dev_processed_path = 'ms_marco_processed/dev_v1.1_processed.json'
test_processed_path = 'ms_marco_processed/test_public_v1.1_processed.json'

processed_to_r_net_dir = 'ms_marco_processed_to_r_net'
train_processed_to_r_net_path = 'ms_marco_processed_to_r_net/train-v1.1.json'
dev_processed_to_r_net_path = 'ms_marco_processed_to_r_net/dev-v1.1.json'
test_processed_to_r_net_path = 'ms_marco_processed_to_r_net/test_public-v1.1.json'


# 读文件
def read_json(file_path):
    data_list = []
    for line in open(file_path):
        data = json.loads(line)
        data_list.append(data)
    return data_list


def processing(data_list, is_test=False):
    data = {}
    data['data'] = []

    for i in range(len(data_list)):
        record = data_list[i]
        concat_passage = record['concat_passage']

        if is_test is False:
            if len(record['answers']) == 0:
                print i, 'no answer'
                continue

            start = record['start']
            end = record['end']
            min_ed_span = concat_passage[start: end]

            r4 = {}
            r4['answer_start'] = start
            r4['text'] = min_ed_span

            r3 = {}
            r3['answers'] = [r4]  # 放r4；虽然多个答案，但span的话找最小一个即可，所以不影响；那么train、dev格式同，test少answer部分信息即可
            r3['question'] = record['query']
            r3['id'] = record['query_id']

        else:
            r3 = {}
            r3['question'] = record['query']
            r3['id'] = record['query_id']

        r2 = {}
        r2['context'] = concat_passage
        r2['qas'] = [r3]  # 放r3

        r1 = {}
        r1['title'] = ""
        r1['paragraphs'] = [r2]  # 放r2

        data['data'].append(r1)

    return data


def write_json(data, file_path):
    with open(file_path, 'w') as json_file:
        json_file.write(json.dumps(data))


if os.path.exists(processed_to_r_net_dir) is False:
    os.mkdir(processed_to_r_net_dir)

print 'train data'
train_data_list = read_json(train_processed_path)
print len(train_data_list)
train_data = processing(train_data_list)
write_json(train_data, train_processed_to_r_net_path)

print 'dev data'
dev_data_list = read_json(dev_processed_path)
dev_data = processing(dev_data_list)
write_json(dev_data, dev_processed_to_r_net_path)

print 'test data'
test_data_list = read_json(test_processed_path)
test_data = processing(test_data_list, True)
write_json(test_data, test_processed_to_r_net_path)
