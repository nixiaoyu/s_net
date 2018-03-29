# coding=utf-8

import json
from synthesis_model import *
from nltk.tokenize import WordPunctTokenizer
from nltk.stem import WordNetLemmatizer
import collections
import random
import numpy as np
import time

# 选项：多种不同输入方式：q；qp；qsp；span

# 读入中间文件；还可读入原来文件；先写读入中间文件的
# 抽取所需数据
# 搞成输入格式
# 训练，测试，输出

batch_size = 128
vocab_size = 30000
embedding_dim = 300
hidden_size = 75
max_anslen = 220
beam_search_size = 12
epoch_num = 3

PAD_ID = 0
UNK_ID = 1
SOS_ID = 2
EOS_ID = 3

train_middle_filepath = '../ms_marco_middle/train_v1.1_middle.json'
dev_middle_filepath = '../ms_marco_middle/dev_v1.1_middle.json'
test_middle_filepath = '../ms_marco_middle/test_v1.1_middle.json'
vocab_filepath = "../output/vocabulary.json"
strtime = time.strftime('%Y%m%d%H%M%S',time.localtime(time.time()))
# strtime= str(vocab_size)
dev_candi_path = '../output/_dev_candidates-'+str(vocab_size)+'-'+strtime+'.json'
test_candi_path = '../output/_test_candidates-'+str(vocab_size)+'-'+strtime+'.json'

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


def get_inputs_texts(data_list,  is_test=False):
    query_ids = []
    queries = []
    passages = []
    starts_one_hot = []
    ends_one_hot = []
    if is_test is False:
        answers = []
        for record in data_list:
            # 其实也许答案应该只用当时抽的时候最相似的那个？？先用所有的吧，不管他
            if len(record['answers']) == 0:  # 这里应该已经没有了
                # queries.append(record['query'])
                # answers.append('')
                # print record['query']  # 有不少
                continue
            for answer in record['answers']:
                query_ids.append(record['query_id'])
                queries.append(record['query'])
                passages.append(record['passage'])
                answers.append(answer)
                starts_one_hot.append(record['starts_one_hot'])
                ends_one_hot.append(record['ends_one_hot'])
        return query_ids, queries, passages, starts_one_hot, ends_one_hot, answers
    else:
        for record in data_list:
            query_ids.append(record['query_id'])
            queries.append(record['query'])
            passages.append(record['passage'])
            starts_one_hot.append(record['starts_one_hot'])
            ends_one_hot.append(record['ends_one_hot'])
        return query_ids, queries, passages, starts_one_hot, ends_one_hot


def get_vocab(data_list, vocab_size=None):
    # 处理，统计词频
    tokenizer = WordPunctTokenizer()
    lemmatizer = WordNetLemmatizer()
    all_words = []
    for record in data_list:
        for passage in record['passages']:  # 都看吧，不要只看选中的
            passage_words = tokenizer.tokenize(passage['passage_text'])  # 分词；要不要全转小写，再看
            passage_lemma_words = [lemmatizer.lemmatize(w) for w in passage_words]  # 词形还原
            all_words.extend(passage_lemma_words)
    vocab_dict = collections.Counter(all_words)
    # 取词频最大的vocab_size个词，另外再加一个<unk>统计其他词,不用加，只需到时不在了index变一下
    # （对于一起训练的都是这样，对于用预训练embedding的则在配的时候不在表中的是0向量）
    if vocab_size is not None:
        vocab_list = vocab_dict.most_common(vocab_size)  # 就是倒序排的
    else:
        vocab_list = vocab_dict.most_common()  # 所有元素
    print len(vocab_list)
    vocab = {}  # 另外，'<PAD>': 0, '<UNK>': 1, '<SOS>': 2, '<EOS>': 3，没有实际替换，在index时直接用数字代替即可
    i = EOS_ID + 1
    for v in vocab_list:
        vocab[v[0]] = i  # 只留词和位置索引
        i += 1
    # print vocab  # 常用词挺多的，要不要考虑去停用词和标点
    return vocab


def to_index(vocab, texts, add_os=True):
    words_indices = []

    tokenizer = WordPunctTokenizer()
    lemmatizer = WordNetLemmatizer()

    # maxlen = 0

    for text in texts:
        words = tokenizer.tokenize(text)
        lemmas = [lemmatizer.lemmatize(w) for w in words]
        # maxlen = max(maxlen, len(lemmas))
        words_index = []
        if add_os is True:
            words_index.append(SOS_ID)  # 开头
        for lemma in lemmas:
            if lemma in vocab:
                words_index.append(vocab[lemma])
            else:
                words_index.append(UNK_ID)  # <unk>
        if add_os is True:
            words_index.append(EOS_ID)  # 结尾
        words_indices.append(words_index)

    # print maxlen

    return words_indices


def reverse_seq(inputs):
    for i in range(len(inputs)):
        inputs[i].reverse()


def generate_candidates(output_ids, query_ids, vocabulary_index):
    candidates = []
    ids = {}
    j = 0
    for i in range(len(query_ids)):
        answer_words = [vocabulary_index[index] for index in output_ids[i] if
                        index in vocabulary_index]  # 还有unknown不应该删，看看怎么办
        answer = ' '.join(answer_words)  # 这样是分词了的，如果标点怎么合？
        candi = {'answers': [answer], 'query_id': query_ids[i]}
        candidates.append(candi)
        i += 1
        # 可以这里去一下重复元素？
        if query_ids[i] in ids:
            continue
        ids[query_ids[i]] = j
        j += 1
    return candidates


def padding(feed):
    maxlen = 0
    actual_lengths = []
    for i in range(len(feed)):
        actual_length = len(feed[i])
        actual_lengths.append(actual_length)
        maxlen = max(maxlen, actual_length)
    for i in range(len(feed)):
        feed[i] += [0] * (maxlen - actual_lengths[i])
    # print maxlen
    return feed, actual_lengths


def get_batch(inputs_q, inputs_p, targets, starts_one_hot, ends_one_hot, batch_size):
    batch_pad_inputs_q = []
    batch_inputs_actual_lengths_q = []
    batch_pad_inputs_p = []
    batch_inputs_actual_lengths_p = []
    batch_pad_targets = []
    batch_targets_actual_lengths = []
    batch_starts = []
    batch_ends = []

    input_q_padding_batch = []
    input_p_padding_batch = []
    starts_padding_batch = []
    ends_padding_batch = []
    target_padding_batch = []
    data_num = len(inputs_q)
    batch_num = data_num / batch_size
    batch_res = data_num % batch_size

    if batch_res != 0:  # 最后一个暂时不要了；不对啊，test不能不要，咋办；补个齐？？
        batch_pad = batch_size - batch_res
        # print batch_pad
        batch_num += 1
        # 弄两个随机的，到最后拼上去即可
        if targets is not None:
            for i in range(batch_pad):
                no = random.randint(0, data_num)
                input_q_padding_batch.append(inputs_q[no])
                input_p_padding_batch.append(inputs_p[no])
                target_padding_batch.append(targets[no])
                starts_padding_batch.append(starts_one_hot[no])
                ends_padding_batch.append(ends_one_hot[no])
        else:
            input_q_padding_batch = []
            for i in range(batch_pad):
                input_q_padding_batch.append([0])
                input_p_padding_batch.append([0])
                starts_padding_batch.append([0])
                ends_padding_batch.append([0])

    # 打乱一下
    if targets is not None:
        data = []
        for i in range(data_num):
            record = [inputs_q[i], inputs_p[i], targets[i], starts_one_hot[i], ends_one_hot[i]]
            data.append(record)
        random.shuffle(data)
        for i in range(data_num):
            inputs_q[i] = data[i][0]
            inputs_p[i] = data[i][1]
            targets[i] = data[i][2]
            starts_one_hot[i] = data[i][3]
            ends_one_hot[i] = data[i][4]
    # else:
    #     data = []
    #     for i in range(data_num):
    #         record = [inputs_q[i], inputs_p[i]]
    #         data.append(record)
    #     random.shuffle(data)
    #     for i in range(data_num):
    #         inputs_q[i] = data[i][0]
    #         inputs_p[i] = data[i][1]

    for batch_no in range(batch_num):
        inputs_q_feed = inputs_q[batch_no * batch_size: batch_no * batch_size + batch_size]  # 右边越界会自动到最后的
        if len(inputs_q_feed) < batch_size:
            # print len(inputs_q_feed)
            inputs_q_feed += input_q_padding_batch
        inputs_q_feed, inputs_q_actual_lengths = padding(inputs_q_feed)  # padding
        inputs_q_feed = np.asarray(inputs_q_feed)  # 转成数组
        inputs_q_actual_lengths = np.asarray(inputs_q_actual_lengths)
        batch_pad_inputs_q.append(inputs_q_feed)  # 该batch加入列表
        batch_inputs_actual_lengths_q.append(inputs_q_actual_lengths)

        inputs_p_feed = inputs_p[batch_no * batch_size: batch_no * batch_size + batch_size]  # 右边越界会自动到最后的
        if len(inputs_p_feed) < batch_size:
            # print len(inputs_q_feed)
            inputs_p_feed += input_p_padding_batch
        inputs_p_feed, inputs_p_actual_lengths = padding(inputs_p_feed)  # padding
        inputs_p_feed = np.asarray(inputs_p_feed)  # 转成数组
        inputs_p_actual_lengths = np.asarray(inputs_p_actual_lengths)
        batch_pad_inputs_p.append(inputs_p_feed)  # 该batch加入列表
        batch_inputs_actual_lengths_p.append(inputs_p_actual_lengths)

        # start, end
        starts_feed = starts_one_hot[batch_no * batch_size: batch_no * batch_size + batch_size]  # 右边越界会自动到最后的
        if len(starts_feed) < batch_size:
            # print len(starts_feed)
            starts_feed += input_q_padding_batch
        starts_feed, starts_actual_lengths = padding(starts_feed)  # padding
        starts_feed = np.asarray(starts_feed)
        batch_starts.append(starts_feed)  # 该batch加入列表

        ends_feed = ends_one_hot[batch_no * batch_size: batch_no * batch_size + batch_size]  # 右边越界会自动到最后的
        if len(ends_feed) < batch_size:
            # print len(ends_feed)
            ends_feed += input_q_padding_batch
        ends_feed, ends_actual_lengths = padding(ends_feed)  # padding
        ends_feed = np.asarray(ends_feed)
        batch_ends.append(ends_feed)  # 该batch加入列表

        if targets is not None:
            decoder_targets_feed = targets[batch_no * batch_size: batch_no * batch_size + batch_size]
            if len(decoder_targets_feed) < batch_size:
                decoder_targets_feed += target_padding_batch
            decoder_targets_feed, targets_actual_lengths = padding(decoder_targets_feed)
            decoder_targets_feed = np.asarray(decoder_targets_feed)
            targets_actual_lengths = np.asarray(targets_actual_lengths)
            batch_pad_targets.append(decoder_targets_feed)
            batch_targets_actual_lengths.append(targets_actual_lengths)

            # print encoder_inputs_feed.shape

    return batch_pad_inputs_q, batch_pad_inputs_p, batch_pad_targets, batch_starts, batch_ends, \
    batch_inputs_actual_lengths_q, batch_inputs_actual_lengths_p, batch_targets_actual_lengths


def train(model, sess, train_inputs_q, train_inputs_p, train_targets, train_starts_one_hot, train_ends_one_hot,
          batch_size, epoch_num):
    losses = []

    batch_pad_inputs_q, batch_pad_inputs_p, batch_pad_targets, batch_starts, batch_ends, \
    batch_inputs_actual_lengths_q, batch_inputs_actual_lengths_p, batch_targets_actual_lengths \
        = get_batch(train_inputs_q, train_inputs_p, train_targets, train_starts_one_hot, train_ends_one_hot,
          batch_size)

    for epoch in range(epoch_num):
        for batch_no in range(len(batch_pad_inputs_q)):  # len(batch_pad_inputs_q)
            feed_dict = {model.inputs_q: batch_pad_inputs_q[batch_no],  # 论文说要倒着才好？回头再说
                         model.inputs_actual_length_q: batch_inputs_actual_lengths_q[batch_no],
                         model.inputs_p: batch_pad_inputs_p[batch_no],  # 论文说要倒着才好？回头再说
                         model.inputs_actual_length_p: batch_inputs_actual_lengths_p[batch_no],
                         model.starts: batch_starts[batch_no],
                         model.ends: batch_starts[batch_no],
                         model.targets_a: batch_pad_targets[batch_no],
                         model.targets_actual_length_a: batch_targets_actual_lengths[batch_no]
                         }
            _, loss_value = sess.run([model.train_op, model.loss], feed_dict=feed_dict)
            losses.append(loss_value)
            print batch_no, ' loss: ', loss_value

            # saver.save(sess, '/home/ujjawal/.ckpt')

    return losses


def test(model, sess, test_inputs_q, test_inputs_p, test_starts_one_hot, test_ends_one_hot,
          batch_size):
    all_answers_ids = []

    batch_pad_inputs_q, batch_pad_inputs_p, batch_pad_targets, batch_starts, batch_ends, \
    batch_inputs_actual_lengths_q, batch_inputs_actual_lengths_p, batch_targets_actual_lengths \
        = get_batch(test_inputs_q, test_inputs_p, None, test_starts_one_hot, test_ends_one_hot, batch_size)

    # print len(batch_pad_inputs_q)

    for batch_no in range(len(batch_pad_inputs_q)):  # len(batch_pad_inputs_q)
        feed_dict = {model.inputs_q: batch_pad_inputs_q[batch_no],  # 论文说要倒着才好？回头再说
                     model.inputs_actual_length_q: batch_inputs_actual_lengths_q[batch_no],
                     model.inputs_p: batch_pad_inputs_p[batch_no],  # 论文说要倒着才好？回头再说
                     model.inputs_actual_length_p: batch_inputs_actual_lengths_p[batch_no],
                     model.starts: batch_starts[batch_no],
                     model.ends: batch_starts[batch_no],
                     }
        answers_ids = sess.run([model.answers_ids], feed_dict=feed_dict)
        # print batch_no, probs.shape  # [batch_size, seq_length, vocab_size]
        print batch_no, answers_ids.shape  # [batch_size, seq_length]
        all_answers_ids.extend(answers_ids.tolist())

    return all_answers_ids


print 'read train texts...'
train_data_list = read_json(train_middle_filepath)
train_query_ids, train_queries, train_passages, train_starts_one_hot, train_ends_one_hot, train_answers \
    = get_inputs_texts(train_data_list)
print 'train datanum: ', len(train_data_list), len(train_answers)  #
print 'read vocab...'
# vocabulary = get_vocab(train_data_list)  # 看论文从训练数据收集的，也先只用训练数据？
# with open(vocab_filename, "w") as f:
#     json.dump(vocabulary, f)
with open(vocab_filepath, 'r') as load_f:
    vocabulary = json.load(load_f)
vocabulary = sorted(vocabulary.items(), key=lambda e: e[1])  # 根据词汇表生成方法，序号越小都词频越高
vocabulary = vocabulary[:vocab_size]
vocabulary = dict(vocabulary)
# print vocabulary
print 'vocab size: ', vocab_size

print 'build synthesis model...'
vocab_size = vocab_size + 4
train_model = SynthesisModel(batch_size, vocab_size, embedding_dim, hidden_size, max_anslen, beam_search_size)
test_model = SynthesisModel(batch_size, vocab_size, embedding_dim, hidden_size, max_anslen, beam_search_size, 'test')
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

print 'generate train data...'
train_inputs_q = to_index(vocabulary, train_queries, False)
train_inputs_p = to_index(vocabulary, train_queries, False)
train_targets = to_index(vocabulary, train_answers)
del train_data_list, train_queries, train_answers
print 'train...'
train(train_model, sess, train_inputs_q, train_inputs_p, train_targets, train_starts_one_hot, train_ends_one_hot, batch_size, epoch_num)
del train_inputs_q, train_inputs_p, train_targets

print 'read dev texts...'
dev_data_list = read_json(dev_middle_filepath)
dev_query_ids, dev_queries, dev_passages, dev_starts_one_hot, dev_ends_one_hot \
    = get_inputs_texts(dev_data_list, True)
print 'dev datanum: ', len(dev_data_list)
print 'generate dev data...'
dev_inputs_q = to_index(vocabulary, dev_queries, False)
dev_inputs_p = to_index(vocabulary, dev_queries, False)
# dev_targets = to_index(vocabulary, dev_answers)
del dev_data_list, dev_queries
print 'test (with dev data)...'
dev_output_ids = test(test_model, sess, dev_inputs_q, dev_inputs_p, dev_starts_one_hot, dev_ends_one_hot, batch_size)
print 'output candidates json...'
vocabulary_index = {v: k for k, v in vocabulary.items()}  # 倒过来便于索引
candidates = generate_candidates(dev_output_ids, dev_query_ids, vocabulary_index)
write_json(candidates, dev_candi_path)

print 'read test texts...'
test_data_list = read_json(test_middle_filepath)
test_query_ids, test_queries, test_passages, test_starts_one_hot, test_ends_one_hot \
    = get_inputs_texts(test_data_list, True)
print 'test datanum: ', len(test_data_list)
print 'generate test data...'
test_inputs_q = to_index(vocabulary, test_queries, False)
test_inputs_p = to_index(vocabulary, test_queries, False)
# test_targets = to_index(vocabulary, test_answers)
del test_data_list, test_queries
print 'test ...'
test_output_ids = test(test_model, sess, test_inputs_q, test_inputs_p, test_starts_one_hot, test_ends_one_hot, batch_size)
print 'output candidates json...'
vocabulary_index = {v: k for k, v in vocabulary.items()}  # 倒过来便于索引
candidates = generate_candidates(test_output_ids, test_query_ids, vocabulary_index)
write_json(candidates, test_candi_path)

# 还应该有个后处理（post_process)


