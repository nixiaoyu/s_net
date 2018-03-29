# coding=utf-8

import json
from nltk.tokenize import WordPunctTokenizer
from nltk.stem import WordNetLemmatizer
import random
import numpy as np
import tensorflow as tf
from extraction_model import *


# 选项：extract；ext+ranking；生成；三者的组合等；不同输入
options = 'extract'

glove_filepath = '../glove.6B/glove.6B.300d.txt'
train_processed_path = '../ms_marco_processed/train_v1.1_processed.json'
dev_processed_path = '../ms_marco_processed/dev_v1.1_processed.json'
test_processed_path = '../ms_marco_processed/test_public_v1.1_processed.json'

train_middle_filepath = '../ms_marco_middle/train_v1.1_middle.json'
dev_middle_filepath = '../ms_marco_middle/dev_v1.1_middle.json'
test_middle_filepath = '../ms_marco_middle/test_v1.1_middle.json'

epoch_num = 1
batch_size = 128
hidden_size = 75
embedding_dim = 300
grad_clip = 5
initial_learning_rate = 1e-3
syn_vocab_size = 200
dropout_rate = 0.1
max_anslen = 220
beam_search_size = 12
# 上述参数是在dev上调的，后续加；dev应是test方法生成，但有target去算accuacy，然后网格搜索，用acc最好的超参；有dev集应该就不需要再交叉验证了吧，用dev集验证即可
# 不对，dev集不是用来调参的，是线下评价用，调参还是可交叉验证；也可dev；另外dev是test方法预测，但由于知道target，可在概率上自己再算些指标，看看放模型里还是模型外
# PAD_ID = 0
# UNK_ID = 1
# SOS_ID = 2
# EOS_ID = 3

# 看要不要都搞一个param文件里


def read_json(file_path):
    data_list = []
    for line in open(file_path):
        data = json.loads(line)
        data_list.append(data)
    return data_list


def write_json(data_list, file_path):
    with open(file_path, 'w') as json_file:
        for data in data_list:
            json_file.write(json.dumps(data))
            json_file.write('\n')


def to_one_hot(num, no):
    y = [0]*num
    y[no] = 1
    return y


def get_inputs_texts(data_list):
    queries = []
    concat_passages = []
    passage_numbers = []
    passage_word_numbers = []
    start_position = []
    end_position = []
    passage_ys = []
    query_ids = []
    
    for record in data_list:
        queries.append(record['query'])
        concat_passages.append(record['concat_passage'])
        passage_numbers.append(record['passage_number'])
        passage_word_numbers.append(record['passage_word_numbers'])
        start_position.append(record['start'])
        end_position.append(record['end'])
        passage_y = to_one_hot(record['passage_number'], record['span_passage_no'])
        passage_ys.append(passage_y)
        query_ids.append(record['query_id'])

    # print passage_word_numbers
    
    return queries, concat_passages, passage_numbers, passage_word_numbers, start_position, \
    end_position, passage_ys, query_ids


def load_glove(filename):
    vocab_embd = {}
    file = open(filename, 'r')
    for line in file.readlines():
        row = line.strip().split(' ')
        vocab_embd[row[0]] = row[1:]
    print('Loaded GloVe!')
    file.close()
    embedding_dim = len(vocab_embd.values()[0])
    vocab_embd['<unk>'] = [0.0]*embedding_dim
    return vocab_embd, embedding_dim


def get_embeddings(vocab_embd, texts):  # 好像很慢，考虑还是搞index往里放
    words_embeddings = []

    # tokenizer = WordPunctTokenizer()
    lemmatizer = WordNetLemmatizer()

    # maxlen = 0

    for text in texts:
        words = text.split(' ')  # 之前分了词了，就这么写了
        # print words
        lemmas = [lemmatizer.lemmatize(w) for w in words]
        # maxlen = max(maxlen, len(lemmas))
        words_embedding = []
        for lemma in lemmas:
            if lemma in vocab_embd:
                words_embedding.append(vocab_embd[lemma])
            else:
                words_embedding.append(vocab_embd['<unk>'])  # <unk>
        words_embeddings.append(words_embedding)

    # print maxlen

    return words_embeddings


def padding_embeddding(feed):
    embedding_dim = len(feed[0])
    maxlen = 0
    actual_lengths = []
    for i in range(len(feed)):
        actual_length = len(feed[i])
        actual_lengths.append(actual_length)
        maxlen = max(maxlen, actual_length)
    for i in range(len(feed)):
        feed[i] += [[0.0]*embedding_dim] * (maxlen - actual_lengths[i])
    # print maxlen
    return feed, actual_lengths


def padding_number(feed):
    maxlen = 0
    actual_lengths = []
    for i in range(len(feed)):
        actual_length = len(feed[i])
        actual_lengths.append(actual_length)
        maxlen = max(maxlen, actual_length)
    for i in range(len(feed)):
        feed[i] += [0] * (maxlen - actual_lengths[i])
    # print maxlen
    return feed


def get_batch(q_embeddings, p_embeddings, passage_numbers, passage_word_numbers,
              is_test=False, starts=None, ends=None, passage_ys=None):
    # print passage_word_numbers  # 这还正常

    batch_pad_inputs_q = []
    batch_inputs_q_actual_lengths = []
    batch_pad_inputs_p = []
    batch_inputs_p_actual_lengths = []
    batch_passage_numbers = []
    batch_passage_word_numbers = []
    batch_starts = []
    batch_ends = []
    batch_passage_ys = []

    data_num = len(q_embeddings)
    batch_num = data_num / batch_size
    batch_res = data_num % batch_size
    if batch_res != 0:  # 最后一个暂时不要了；不对啊，test不能不要，咋办；补个齐？？
        batch_pad = batch_size - batch_res
        # print batch_pad
        batch_num += 1
        # 弄两个随机的，到最后拼上去即可
        if is_test is False:
            input_q_padding_batch = []
            input_p_padding_batch = []
            passage_numbers_padding_batch = []
            passage_word_numbers_padding_batch = []
            starts_padding_batch = []
            ends_padding_batch = []
            passage_ys_padding_batch = []          
            for i in range(batch_pad):
                no = random.randint(0, data_num)
                input_q_padding_batch.append(q_embeddings[no])
                input_p_padding_batch.append(p_embeddings[no])
                passage_numbers_padding_batch.append(passage_numbers[no])
                passage_word_numbers_padding_batch.append(passage_word_numbers[no])
                starts_padding_batch.append(starts[no])
                ends_padding_batch.append(ends[no])
                passage_ys_padding_batch.append(passage_ys[no])
        else:
            input_q_padding_batch = []
            input_p_padding_batch = []
            passage_numbers_padding_batch = []
            passage_word_numbers_padding_batch = []
            for i in range(batch_pad):
                input_q_padding_batch.append([0])
                input_p_padding_batch.append([0])
                passage_numbers_padding_batch.append(0)
                passage_word_numbers_padding_batch.append([0])

    # 打乱一下
    if is_test is False:
        data = []
        for i in range(data_num):
            record = [q_embeddings[i], p_embeddings[i], passage_numbers[i], passage_word_numbers[i],
                      starts[i], ends[i], passage_ys[i]]
            data.append(record)
        random.shuffle(data)
        for i in range(data_num):
            q_embeddings[i] = data[i][0]
            p_embeddings[i] = data[i][1]
            passage_numbers[i] = data[i][2]
            passage_word_numbers[i] = data[i][3]
            starts[i] = data[i][4]
            ends[i] = data[i][5]
            passage_ys[i] = data[i][6]

    for batch_no in range(batch_num):
        q_embeddings_feed = q_embeddings[batch_no * batch_size: batch_no * batch_size + batch_size]  # 右边越界会自动到最后的
        if len(q_embeddings_feed) < batch_size:
            # print len(encoder_inputs_q_feed)
            q_embeddings_feed += input_q_padding_batch
        q_embeddings_feed, inputs_q_actual_lengths = padding_embeddding(q_embeddings_feed)  # padding
        q_embeddings_feed = np.asarray(q_embeddings_feed)  # 转成数组
        inputs_q_actual_lengths = np.asarray(inputs_q_actual_lengths)
        batch_pad_inputs_q.append(q_embeddings_feed)  # 该batch加入列表
        batch_inputs_q_actual_lengths.append(inputs_q_actual_lengths)

        p_embeddings_feed = p_embeddings[batch_no * batch_size: batch_no * batch_size + batch_size]  # 右边越界会自动到最后的
        if len(p_embeddings_feed) < batch_size:
            # print len(encoder_inputs_q_feed)
            p_embeddings_feed += input_p_padding_batch
        p_embeddings_feed, inputs_p_actual_lengths = padding_embeddding(p_embeddings_feed)  # padding
        p_embeddings_feed = np.asarray(p_embeddings_feed)  # 转成数组
        inputs_p_actual_lengths = np.asarray(inputs_p_actual_lengths)
        batch_pad_inputs_p.append(p_embeddings_feed)  # 该batch加入列表
        batch_inputs_p_actual_lengths.append(inputs_p_actual_lengths)
        
        passage_number_feed = passage_numbers[batch_no * batch_size: batch_no * batch_size + batch_size]
        if len(passage_number_feed) < batch_size:
            passage_number_feed += passage_numbers_padding_batch
        passage_number_feed = np.asarray(passage_number_feed)
        batch_passage_numbers.append(passage_number_feed)
        
        # passage_words也需要padding
        # print passage_word_numbers  # 这就不对了？？？什么鬼啊。。哦写错下标
        passage_word_number_feed = passage_word_numbers[batch_no * batch_size: batch_no * batch_size + batch_size]
        # print 'y ', passage_word_number_feed
        if len(passage_word_number_feed) < batch_size:
            passage_word_number_feed += passage_word_numbers_padding_batch
        # print passage_word_number_feed
        passage_word_number_feed = padding_number(passage_word_number_feed)  # padding
        passage_word_number_feed = np.asarray(passage_word_number_feed)
        batch_passage_word_numbers.append(passage_word_number_feed)
        
        if is_test is False:
            starts_feed = starts[batch_no * batch_size: batch_no * batch_size + batch_size]
            if len(starts_feed) < batch_size:
                starts_feed += starts_padding_batch
            starts_feed = np.asarray(starts_feed)
            batch_starts.append(starts_feed)
            
            ends_feed = ends[batch_no * batch_size: batch_no * batch_size + batch_size]
            if len(ends_feed) < batch_size:
                ends_feed += ends_padding_batch
            ends_feed = np.asarray(ends_feed)
            batch_ends.append(ends_feed)
            
            # passage_ys也要padding
            passage_ys_feed = passage_ys[batch_no * batch_size: batch_no * batch_size + batch_size]
            if len(passage_ys_feed) < batch_size:
                passage_ys_feed += passage_ys_padding_batch
            passage_ys_feed = padding_number(passage_ys_feed)  # padding
            passage_ys_feed = np.asarray(passage_ys_feed)
            batch_passage_ys.append(passage_ys_feed)

    return batch_pad_inputs_q, batch_inputs_q_actual_lengths,  batch_pad_inputs_p, batch_inputs_p_actual_lengths,\
           batch_passage_numbers, batch_passage_word_numbers, batch_starts, batch_ends, batch_passage_ys


def train(model, sess, train_q_embeddings, train_p_embeddings, train_passage_numbers, train_passage_word_numbers, 
      train_starts, train_ends, train_passage_ys):
    losses = []

    # print train_passage_word_numbers

    batch_pad_inputs_q, batch_inputs_q_actual_lengths, batch_pad_inputs_p, batch_inputs_p_actual_lengths, \
    batch_passage_numbers, batch_passage_word_numbers, batch_starts, batch_ends, batch_passage_ys\
        = get_batch(train_q_embeddings, train_p_embeddings, train_passage_numbers, train_passage_word_numbers, False,
      train_starts, train_ends, train_passage_ys)

    tvars = tf.trainable_variables()
    print tvars
    grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), grad_clip)
    optimizer = tf.train.AdamOptimizer(initial_learning_rate)
    # grads_and_vars = optimizer.compute_gradients(self.loss)
    # self.train_op = optimizer.apply_gradients(grads_and_vars)
    train_op = optimizer.apply_gradients(zip(grads, tvars))
    # print train_op

    for epoch in range(epoch_num):
        for batch_no in range(len(batch_pad_inputs_q)):  # len(batch_pad_inputs_q)
            feed_dict = {model.inputs_embedded_q: batch_pad_inputs_q[batch_no],
                         model.inputs_actual_length_q: batch_inputs_q_actual_lengths[batch_no],
                         model.inputs_embedded_concat_p: batch_pad_inputs_p[batch_no],
                         model.inputs_actual_length_concat_p: batch_inputs_p_actual_lengths[batch_no],
                         model.passage_numbers: batch_passage_numbers[batch_no],
                         model.passage_word_numbers: batch_passage_word_numbers[batch_no],
                         model.start_position: batch_starts[batch_no],
                         model.end_position: batch_ends[batch_no],
                         model.passage_y: batch_passage_ys[batch_no],
                         }
            _, loss_value = sess.run([train_op, model.loss], feed_dict=feed_dict)
            losses.append(loss_value)
            print batch_no, ' loss: ', loss_value


def test(model, sess, test_q_embeddings, test_p_embeddings, test_passage_numbers, test_passage_word_numbers):
    starts_one_hot = []
    ends_one_hot = []

    batch_pad_inputs_q, batch_inputs_q_actual_lengths, batch_pad_inputs_p, batch_inputs_p_actual_lengths, \
    batch_passage_numbers, batch_passage_word_numbers, batch_starts, batch_ends, batch_passage_ys \
        = get_batch(test_q_embeddings, test_p_embeddings, test_passage_numbers, test_passage_word_numbers, True)
    for batch_no in range(len(batch_pad_inputs_q)):  # len(batch_pad_inputs_q)
        feed_dict = {model.inputs_embedded_q: batch_pad_inputs_q[batch_no],
                     model.inputs_actual_length_q: batch_inputs_q_actual_lengths[batch_no],
                     model.inputs_embedded_concat_p: batch_pad_inputs_p[batch_no],
                     model.inputs_actual_length_concat_p: batch_inputs_p_actual_lengths[batch_no],
                     model.passage_numbers: batch_passage_numbers[batch_no],
                     model.passage_word_numbers: batch_passage_word_numbers[batch_no],
                     }
        p1, p2 = sess.run([model.p1, model.p2], feed_dict=feed_dict)
        starts_one_hot.append(p1.tolist())
        ends_one_hot.append(p2.tolist())
        # 这是one_hot，后面要找出实际位置；可就在这找

    return starts_one_hot, ends_one_hot


def one_hot_to_num(a):
    return a.index(1)


def one_hot_to_num_batch(arrs):
    ps = []
    for a in arrs:
        ps.append(one_hot_to_num(a))
    return ps


def generate_candidates(starts, ends, concat_passages, query_ids):
    candidates = []
    ids = {}
    j = 0
    for i in range(len(query_ids)):
        answer = concat_passages[i][starts[i]:ends[i]+1]
        # 如果是输出结果文件，可以后处理下，把标点前的空格删掉之类的
        candi = {'answers': [answer], 'query_id': query_ids[i]}
        candidates.append(candi)
        # 可以这里去一下重复元素？
        if query_ids[i] in ids:
            continue
        ids[query_ids[i]] = j
        j += 1
    return candidates


def find_span_passages(start, end, passage_starts, passage_number, len_concat_passage):
    p_start = 0
    p_end = 0
    for i in range(passage_number-1):  # 还有可能最后一篇文章
        if passage_starts[i] <= start <= passage_starts[i+1]:
            p_start = passage_starts[i]
        if passage_starts[i] <= end <= passage_starts[i+1]:
            p_end = passage_starts[i+1]-1
    if start >= passage_starts[passage_number-1]:
        p_start = passage_starts[passage_number-1]
    if end >= passage_starts[passage_number-1]:
        p_end = len_concat_passage
    return p_start, p_end


def generate_middle_data(data_list, starts_one_hot, ends_one_hot, starts, ends):
    middle_data_list = []

    for i in range(len(data_list)):
        record = data_list[i]
        start = starts[i]
        end = ends[i]
        concat_passage = record['concat_passage']
        span = concat_passage[start:end+1] # 不对，要的不是这个

        passage_starts = record['passage_starts']
        passage_number = record['passage_number']
        if 'span_passag_no' in record:
            passage_no = record['span_passag_no']
            p_start = passage_starts[passage_no]
            if passage_no < passage_number-1:
                p_end = passage_starts[passage_no+1]-1
            else:
                p_end = len(concat_passage)
        else:
            # passage_starts是个列表；找start，end了
            # 这样，只要跨越的文章都保留下来
            p_start, p_end = find_span_passages(start, end, passage_starts, passage_number, len(concat_passage))
        passage = concat_passage[p_start, p_end + 1]
        start_one_hot_p = starts_one_hot[i][p_start, p_end + 1]
        end_one_hot_p = ends_one_hot[i][p_start, p_end + 1]

        mid_record = {'query_id': record['query_id'], 'query': record['query'],
                      'starts_one_hot': start_one_hot_p, 'ends_one_hot': end_one_hot_p, 'span': span,  # 留一下吧
                      'passage': passage}
        if 'answers' in record:
            mid_record['answers'] = record['answers']
        middle_data_list.append(mid_record)

    return middle_data_list
# 需要：q, span所在p，每个位置是不是开始和结束的pos_one_hot，实际答案；为了比较，也可要所有passages等，不过这个可读原来文件即可


def pos_to_one_hot_pos(concat_passages, positions):
    positions_one_hot = []
    for i in range(len(positions)):
        num = len(concat_passages[i])
        pos = positions[i]
        pos_one_hot = to_one_hot(num, pos)
        positions_one_hot.append(pos_one_hot)
    return positions_one_hot


print 'build extraction model...'
train_model = ExtractionModel(batch_size, hidden_size, embedding_dim, dropout_rate, grad_clip, initial_learning_rate)
test_model = ExtractionModel(batch_size, hidden_size, embedding_dim, dropout_rate, grad_clip, 'test')

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

print 'load glove...'
vocab_embd, embedding_dim = load_glove(glove_filepath)

print 'read train data...'
train_data_list = read_json(train_processed_path)
# 要搞出需要的；挺多的，是不是一个函数搞定拉倒
train_queries, train_concat_passages, train_passage_numbers, train_passage_word_numbers, train_start_positions, \
train_end_positions, train_passage_ys, train_query_ids = get_inputs_texts(train_data_list)
print 'generate train data...'
train_q_embeddings = get_embeddings(vocab_embd, train_queries)
train_p_embeddings = get_embeddings(vocab_embd, train_concat_passages)
print 'train...'
train(train_model, sess, train_q_embeddings, train_p_embeddings, train_passage_numbers, train_passage_word_numbers, 
      train_start_positions, train_end_positions, train_passage_ys)

print 'read dev texts...'
dev_data_list = read_json(dev_processed_path)
print 'generate dev data...'
dev_queries, dev_concat_passages, dev_passage_numbers, dev_passage_word_numbers, dev_start_positions, \
dev_end_positions, dev_passage_ys, dev_query_ids = get_inputs_texts(dev_data_list)
print 'test (with dev data)...'
dev_starts_one_hot, dev_ends_one_hot = test(test_model, sess, dev_queries, dev_concat_passages, dev_passage_numbers,
                                    dev_passage_word_numbers)
dev_starts = one_hot_to_num_batch(dev_starts_one_hot)
dev_ends = one_hot_to_num_batch(dev_ends_one_hot)
print 'output candidates json...'
generate_candidates(dev_starts, dev_ends, dev_concat_passages, dev_query_ids)

print 'read test texts...'
test_data_list = read_json(test_processed_path)
print 'generate test data...'
test_queries, test_concat_passages, test_passage_numbers, test_passage_word_numbers, test_start_positions, \
test_end_positions, test_passage_ys, test_query_ids = get_inputs_texts(test_data_list)
print 'test...'
test_starts_one_hot, test_ends_one_hot = test(test_model, sess, test_queries, test_concat_passages, test_passage_numbers,
                                    test_passage_word_numbers)
test_starts = one_hot_to_num_batch(test_starts_one_hot)
test_ends = one_hot_to_num_batch(test_ends_one_hot)
print 'output candidates json...'
generate_candidates(test_starts, test_ends, test_concat_passages, test_query_ids)


# 生成中间文件
# train需要生成个one_hot
train_starts_one_hot = pos_to_one_hot_pos(train_concat_passages, train_start_positions)
train_ends_one_hot = pos_to_one_hot_pos(train_concat_passages, train_end_positions)
train_middle_data_list = generate_middle_data(train_data_list, train_starts_one_hot, train_ends_one_hot, 
                                              train_start_positions, train_end_positions)
write_json(train_middle_data_list, train_middle_filepath)

dev_middle_data_list = generate_middle_data(dev_data_list, dev_starts_one_hot, dev_ends_one_hot, 
                                              dev_start_positions, dev_end_positions)
write_json(dev_middle_data_list, dev_middle_filepath)

test_middle_data_list = generate_middle_data(test_data_list, test_starts_one_hot, test_ends_one_hot, 
                                              test_start_positions, test_end_positions)
write_json(test_middle_data_list, test_middle_filepath)