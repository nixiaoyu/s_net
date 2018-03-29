# coding=utf-8

import json
from nltk.tokenize import WordPunctTokenizer
from Levenshtein import distance
import sys
import os

# 读
# 每个问题，取所有span
# 选rouge-l最高的span作为target，并记录所在passage

train_path = '../ms_marco/train_v1.1.json/train_v1.1.json'
dev_path = '../ms_marco/dev_v1.1.json/dev_v1.1.json'
test_path = '../ms_marco/test_public_v1.1.json/test_public_v1.1.json'

processed_dir = '../ms_marco_processed'
train_processed_path = '../ms_marco_processed/train_v1.1_processed.json'
dev_processed_path = '../ms_marco_processed/dev_v1.1_processed.json'
test_processed_path = '../ms_marco_processed/test_public_v1.1_processed.json'


# 读文件
def read_json(file_path):
    data_list = []
    for line in open(file_path):
        data = json.loads(line)
        data_list.append(data)
    return data_list


def processing(data_list, is_test=False):
    tokenizer = WordPunctTokenizer()
    processed_data_list = []
    for i in range(len(data_list)):  # len(data_list)

        if i % 2000 == 0:
            print 'prcoces:', i

        record = data_list[i]

        passages_tokens = []
        concat_passage = ''
        passage_word_numbers = []
        passage_starts = []
        for passage in record['passages']:
            passage_text = passage['passage_text']
            passage_tokens = tokenizer.tokenize(passage_text)
            passages_tokens.append(passage_tokens)
            passage_start = len(concat_passage)  # 未拼前的长度来算
            passage_starts.append(passage_start)
            concat_passage += ' '.join(passage_tokens) + ' '
            passage_word_number = len(passage_tokens)
            passage_word_numbers.append(passage_word_number)
            
        passage_number = len(passage_word_numbers)

        query_words = tokenizer.tokenize(record['query'])
        query = ' '.join(query_words)  # 都分词拼了

        precord = {'concat_passage': concat_passage, 'passage_number': passage_number, 
                   'passage_word_numbers': passage_word_numbers, 'passage_starts': passage_starts,
                   'query_id': record['query_id'], 'query': query}

        if is_test is False:
            if len(record['answers']) == 0:
                print i, 'no answer'
                continue
            min_ed = sys.maxint
            min_ed_span = ''
            min_len = sys.maxint
            max_len = 0
            answers = []
            for answer in record['answers']:
                answer_tokens = tokenizer.tokenize(answer)
                answer = ' '.join(answer_tokens)
                answers.append(answer)
                alen = len(answer_tokens)
                min_len = min(min_len, max(alen - 10, 1))  # 注意是按词算的
                max_len = max(max_len, alen + 10)
            # 对每个passage，得搞出所有span来才行
            span_passage_no = 0
            for no in range(len(passages_tokens)):
                passage_tokens = passages_tokens[no]
                # spans = passage_tokens
                plen = len(passage_tokens)
                for k in range(min_len, min(max_len+1, plen+1)):
                    for j in range(plen-k+1):
                        k_gram = ' '.join(passage_tokens[j: j+k])
                        # k_gram与answer的编辑距离
                        for answer in answers:  # 是不是因为有没有答案的，所以有问题；看看是不是直接把没答案的去了
                            # print i_gram, '|', answer
                            ed = distance(unicode(k_gram), unicode(answer))
                            if min_ed > ed:
                                min_ed = ed
                                min_ed_span = k_gram
                                span_passage_no = no
            # 确定起止
            start = concat_passage.find(min_ed_span)
            end = start + len(min_ed_span)  # 是包含的
            print i, start, end, span_passage_no
            if min_ed_span == '' or start == -1:  # 这种样本就不要了；还可加强条件，再筛一下ed大的样本，算了先这样
                print min_ed, '|', min_ed_span, '|', answers, '|', concat_passage
                continue

            precord['start'] = start
            precord['end'] = end
            precord['answers'] = answers

        processed_data_list.append(precord)

    return processed_data_list


def write_json(data_list, file_path):
    with open(file_path, 'w') as json_file:
        for data in data_list:
            json_file.write(json.dumps(data))
            json_file.write('\n')


if os.path.exists(processed_dir) is False:
    os.mkdir(processed_dir)

print 'train data'
train_data_list = read_json(train_path)
train_processed_data_list = processing(train_data_list)
del train_data_list
write_json(train_processed_data_list, train_processed_path)
del train_processed_data_list

print 'dev data'
dev_data_list = read_json(dev_path)
dev_processed_data_list = processing(dev_data_list)
del dev_data_list
write_json(dev_processed_data_list, dev_processed_path)
del dev_processed_data_list

print 'test data'
test_data_list = read_json(test_path)
test_processed_data_list = processing(test_data_list, True)
del test_data_list
write_json(test_processed_data_list, test_processed_path)
del test_processed_data_list


# 其实可以转化的，两个文件跑一个，另一个写个代码转过去就好了；不对，两个怎么不一致了