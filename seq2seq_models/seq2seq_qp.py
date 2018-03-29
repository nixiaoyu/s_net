# coding=utf-8

# niyihan

import tensorflow as tf
from tensorflow.contrib.seq2seq import *
from tensorflow.python.layers.core import Dense
from tensorflow.contrib.rnn import LSTMStateTuple
import json
from nltk.tokenize import WordPunctTokenizer
from nltk.stem import WordNetLemmatizer
import collections
import numpy as np
import time
import random

# 超参数等
vocab_size = 30000
embedding_dim = 300
hidden_size = 150
batch_size = 128
grad_clip = 5
decoder_max_iteration = 220
initial_learning_rate = 1e-3
epoch_num = 1
with_att = True  # 是否attention
input_reverse = True  # 输入是否逆向
bidirection = False  # 是否双向；暂未用
rnn_type = 'lstm'  # GRU还是LSTM等；暂未用
input_mode = 'qsp'  # q，qp，qsp
# 优化方式？
PAD_ID = 0
UNK_ID = 1
SOS_ID = 2
EOS_ID = 3

# 数据路径
train_path = '../ms_marco/train_v1.1.json/train_v1.1.json'
dev_path = '../ms_marco/dev_v1.1.json/dev_v1.1.json'
test_path = '../ms_marco/test_public_v1.1.json/test_public_v1.1.json'
vocab_filename = "../output/vocabulary.json"
strtime = time.strftime('%Y%m%d%H%M%S',time.localtime(time.time()))
# strtime= str(vocab_size)
dev_candi_path = '../output/'+input_mode+'_dev_candidates-'+str(vocab_size)+'-'+strtime+'.json'
test_candi_path = '../output/'+input_mode+'_test_candidates-'+str(vocab_size)+'-'+strtime+'.json'


# 模型
class Seq2SeqModel(object):

    def __init__(self, vocab_size, embedding_dim, hidden_size, grad_clip, decoder_max_iteration, batch_size,
                 initial_learning_rate, with_att, is_inference=False):
        # inputs
        self.encoder_inputs_q = tf.placeholder(tf.int32, shape=[batch_size, None], name='encoder_inputs_q')
        self.encoder_inputs_actual_length_q = tf.placeholder(tf.int32, [batch_size],
                                            name='encoder_inputs_actual_length_q')  # 每句输入的实际长度，除了padding
        self.encoder_inputs_p = tf.placeholder(tf.int32, shape=[batch_size, None], name='encoder_inputs_p')
        self.encoder_inputs_actual_length_p = tf.placeholder(tf.int32, [batch_size],
                                                           name='encoder_inputs_actual_length_p')  # 每句输入的实际长度，除了padding

        # embedding
        with tf.variable_scope('embedding', reuse=tf.AUTO_REUSE):
            embedding = tf.get_variable(name='embedding',
                            initializer=tf.truncated_normal(shape=[vocab_size, embedding_dim], stddev=0.1))
            # 截断的产生正态分布的函数，值与均值的差值大于两倍标准差则重新生成

        encoder_inputs_embedded_q = tf.nn.embedding_lookup(embedding, self.encoder_inputs_q)
        encoder_inputs_embedded_p = tf.nn.embedding_lookup(embedding, self.encoder_inputs_p)

        # encoder
        with tf.variable_scope('encoder', reuse=tf.AUTO_REUSE):
            with tf.variable_scope('encoder_q', reuse=tf.AUTO_REUSE):
                fcell = tf.contrib.rnn.BasicLSTMCell(hidden_size)
                encoder_outputs_q, encoder_state_q = tf.nn.dynamic_rnn(fcell, inputs=encoder_inputs_embedded_q,
                                                dtype=tf.float32, sequence_length=self.encoder_inputs_actual_length_q)
            with tf.variable_scope('encoder_p', reuse=tf.AUTO_REUSE):
                fcellp = tf.contrib.rnn.BasicLSTMCell(hidden_size)
                encoder_outputs_p, encoder_state_p = tf.nn.dynamic_rnn(fcellp, inputs=encoder_inputs_embedded_p,
                                                                   dtype=tf.float32,
                                                                   sequence_length=self.encoder_inputs_actual_length_p)
            c_q, h_q = encoder_state_q
            c_p, h_p = encoder_state_p
            c = tf.concat([c_q, c_p], 1)
            h = tf.concat([h_q, h_p], 1)
            encoder_state = LSTMStateTuple(c, h)
            # print encoder_state_q, encoder_state_p, encoder_state

        # helper
        if is_inference:
            self.start_tokens = tf.placeholder(tf.int32, shape=[batch_size], name='start_tokens')
            #self.end_token = tf.placeholder(tf.int32, name='end_token')
            helper = GreedyEmbeddingHelper(embedding, self.start_tokens, EOS_ID)
        else:
            self.target_ids = tf.placeholder(tf.int32, shape=[batch_size, None], name='target_ids')
            self.decoder_targets_actual_length = tf.placeholder(tf.int32, shape=[batch_size],
                                                                name='decoder_targets_actual_length')
            target_embedded = tf.nn.embedding_lookup(embedding, self.target_ids)
            helper = TrainingHelper(target_embedded, self.decoder_targets_actual_length)

        # decoder
        with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE):
            fc_layer = Dense(vocab_size)
            decoder_cell = tf.contrib.rnn.BasicLSTMCell(hidden_size*2)  # 拼接了，是不是这乘2；这里这么搞吧，实际自己写时不一定，可用矩阵调
            if with_att is True:
                encoder_outputs = tf.concat([encoder_outputs_q, encoder_outputs_p], axis=1)
                encoder_inputs_actual_length = self.encoder_inputs_actual_length_q + self.encoder_inputs_actual_length_p
                # Create an attention mechanism
                attention_states = tf.transpose(encoder_outputs, [0, 1, 2])
                attention_mechanism = tf.contrib.seq2seq.LuongAttention(
                    hidden_size*2, attention_states,
                    memory_sequence_length=encoder_inputs_actual_length)
                decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
                    decoder_cell, attention_mechanism,
                    attention_layer_size=hidden_size*2)  # 加个attention
                encoder_state = decoder_cell.zero_state(batch_size, tf.float32).clone(cell_state=encoder_state)
            decoder = BasicDecoder(decoder_cell, helper, encoder_state, fc_layer)

        logits, final_state, final_sequence_lengths = dynamic_decode(decoder, maximum_iterations=decoder_max_iteration)

        if not is_inference:
            mask = tf.sequence_mask(self.decoder_targets_actual_length, dtype=tf.float32)
            self.loss = tf.contrib.seq2seq.sequence_loss(logits.rnn_output, self.target_ids, weights=mask)
            # define train op
            tvars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), grad_clip)
            optimizer = tf.train.AdamOptimizer(initial_learning_rate)
            self.train_op = optimizer.apply_gradients(zip(grads, tvars))
        else:
            self.prob = tf.nn.softmax(logits.rnn_output)
            self.sample_id = logits.sample_id


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


# 分批，每批padding
def get_batch(inputs_q, inputs_p, targets, batch_size):
    batch_pad_inputs_q = []
    batch_inputs_q_actual_lengths = []
    batch_pad_inputs_p = []
    batch_inputs_p_actual_lengths = []
    batch_pad_targets = []
    batch_targets_actual_lengths = []

    data_num = len(inputs_q)
    batch_num = data_num / batch_size
    batch_res = data_num % batch_size
    if batch_res != 0:  # 最后一个暂时不要了；不对啊，test不能不要，咋办；补个齐？？
        batch_pad = batch_size - batch_res
        print batch_pad
        batch_num += 1
        # 弄两个随机的，到最后拼上去即可
        if targets is not None:
            input_q_padding_batch = []
            input_p_padding_batch = []
            target_padding_batch = []
            for i in range(batch_pad):
                no = random.randint(0, data_num)
                input_q_padding_batch.append(inputs_q[no])
                input_p_padding_batch.append(inputs_p[no])
                target_padding_batch.append(targets[no])
        else:
            input_q_padding_batch = []
            for i in range(batch_pad):
                input_q_padding_batch.append([0])
            input_p_padding_batch = input_q_padding_batch

    # 打乱一下
    if targets is not None:
        data = []
        for i in range(data_num):
            record = [inputs_q[i], inputs_p[i], targets[i]]
            data.append(record)
        random.shuffle(data)
        for i in range(data_num):
            inputs_q[i] = data[i][0]
            inputs_p[i] = data[i][1]
            targets[i] = data[i][2]
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
        encoder_inputs_q_feed = inputs_q[batch_no * batch_size: batch_no * batch_size + batch_size]  # 右边越界会自动到最后的
        if len(encoder_inputs_q_feed) < batch_size:
            # print len(encoder_inputs_q_feed)
            encoder_inputs_q_feed += input_q_padding_batch
        encoder_inputs_q_feed, inputs_q_actual_lengths = padding(encoder_inputs_q_feed)  # padding
        encoder_inputs_q_feed = np.asarray(encoder_inputs_q_feed)  # 转成数组
        inputs_q_actual_lengths = np.asarray(inputs_q_actual_lengths)
        batch_pad_inputs_q.append(encoder_inputs_q_feed)  # 该batch加入列表
        batch_inputs_q_actual_lengths.append(inputs_q_actual_lengths)

        encoder_inputs_p_feed = inputs_p[batch_no * batch_size: batch_no * batch_size + batch_size]  # 右边越界会自动到最后的
        if len(encoder_inputs_p_feed) < batch_size:
            # print len(encoder_inputs_q_feed)
            encoder_inputs_p_feed += input_p_padding_batch
        encoder_inputs_p_feed, inputs_p_actual_lengths = padding(encoder_inputs_p_feed)  # padding
        encoder_inputs_p_feed = np.asarray(encoder_inputs_p_feed)  # 转成数组
        inputs_p_actual_lengths = np.asarray(inputs_p_actual_lengths)
        batch_pad_inputs_p.append(encoder_inputs_p_feed)  # 该batch加入列表
        batch_inputs_p_actual_lengths.append(inputs_p_actual_lengths)

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

    return batch_pad_inputs_q, batch_pad_inputs_p, batch_pad_targets, \
           batch_inputs_q_actual_lengths, batch_inputs_p_actual_lengths, batch_targets_actual_lengths


# 训练
def train(model, train_inputs_q, train_inputs_p, train_targets, batch_size, epoch_num):
    losses = []

    batch_pad_inputs_q, batch_pad_inputs_p, batch_pad_targets, \
    batch_inputs_actual_lengths_q, batch_inputs_actual_lengths_p, batch_targets_actual_lengths \
        = get_batch(train_inputs_q, train_inputs_p, train_targets, batch_size)

    for epoch in range(epoch_num):
        for batch_no in range(len(batch_pad_inputs_q)):  # len(batch_pad_inputs_q)
            feed_dict = {model.encoder_inputs_q: batch_pad_inputs_q[batch_no],  # 论文说要倒着才好？回头再说
                         model.encoder_inputs_actual_length_q: batch_inputs_actual_lengths_q[batch_no],
                         model.encoder_inputs_p: batch_pad_inputs_p[batch_no],  # 论文说要倒着才好？回头再说
                         model.encoder_inputs_actual_length_p: batch_inputs_actual_lengths_p[batch_no],
                         model.target_ids: batch_pad_targets[batch_no],
                         model.decoder_targets_actual_length: batch_targets_actual_lengths[batch_no]
                         }
            _, loss_value = sess.run([model.train_op, model.loss], feed_dict=feed_dict)
            losses.append(loss_value)
            print batch_no, ' loss: ', loss_value


# 测试
def test(model, test_inputs_q, test_inputs_p, test_targets, batch_size):
    all_sample_id = []

    batch_pad_inputs_q, batch_pad_inputs_p, batch_pad_targets, \
    batch_inputs_actual_lengths_q, batch_inputs_actual_lengths_p, batch_targets_actual_lengths \
        = get_batch(test_inputs_q, test_inputs_p, test_targets, batch_size)

    print len(batch_pad_inputs_q)

    for batch_no in range(len(batch_pad_inputs_q)):  # len(batch_pad_inputs_q)
        feed_dict = {model.encoder_inputs_q: batch_pad_inputs_q[batch_no],  # 论文说要倒着才好？回头再说
                     model.encoder_inputs_actual_length_q: batch_inputs_actual_lengths_q[batch_no],
                     model.encoder_inputs_p: batch_pad_inputs_p[batch_no],  # 论文说要倒着才好？回头再说
                     model.encoder_inputs_actual_length_p: batch_inputs_actual_lengths_p[batch_no],
                     model.start_tokens: np.array([SOS_ID] * batch_pad_inputs_q[batch_no].shape[0]),
                     # model.end_token: EOS_ID
                     }
        probs, sample_id = sess.run([model.prob, model.sample_id], feed_dict=feed_dict)
        # print batch_no, probs.shape  # [batch_size, seq_length, vocab_size]
        print batch_no, sample_id.shape  # [batch_size, seq_length]
        # if test_targets is not None:
        #     targets_actual_lengths_tf = tf.convert_to_tensor(targets_actual_lengths)
        #     probs_tf = tf.convert_to_tensor(probs)
        #     targets_tf = tf.convert_to_tensor(batch_pad_targets[batch_no])
        #     mask = tf.sequence_mask(targets_actual_lengths_tf, dtype=tf.float32)
        #     loss = tf.contrib.seq2seq.sequence_loss(probs_tf, targets_tf, weights=mask)
        #     print batch_no, 'test loss: ', loss
        all_sample_id.extend(sample_id.tolist())

    return all_sample_id


# 读文件
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


# 抽出qa对或q
def get_ms_marco_dqa(data_list, input_mode, is_test=False):
    query_ids = []
    queries = []
    passages = []
    if is_test is False:
        answers = []
        for record in data_list:
            if len(record['answers']) == 0:  # 无答案的情况;加不加没什么区别
                query_ids.append(record['query_id'])
                queries.append(record['query'])
                answers.append('')
                # print record['query']  # 有不少
                continue
            for answer in record['answers']:
                query_ids.append(record['query_id'])
                answers.append(answer)
                queries.append(record['query'])
                passage_list = record['passages']
                passage_concat = ''
                for p in passage_list:
                    if input_mode == 'qsp' and p['is_selected'] == 0:
                        continue
                    passage_concat += ' ' + p['passage_text']
                passages.append(passage_concat)
        return query_ids, queries, passages, answers
    else:
        for record in data_list:
            query_ids.append(record['query_id'])
            queries.append(record['query'])
            passage_list = record['passages']
            passage_concat = ''
            for p in passage_list:
                passage_concat += ' ' + p['passage_text']  # test数据没有selected，dev有，但是为公平用于测试时也不利用该信息吧
                # 会有点慢，考虑把生成文章这里分开，搞出文件来
            passages.append(passage_concat)
        return query_ids, queries, passages


# 这个数据抽取方式不完善，不好复原，先简单改改，回头再全面改
# 评测程序要求每记录至多1个答案，不过id貌似可多次出现
# 无答案的也可评，但算法中是否分开比较好;先不分吧
# 先把dev完全当test来，不考虑调参吧


# 由词频确定词典
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


# 转换为id形式，即可输入model的形式
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


def generate_candidates(output_ids, query_ids):
    candidates = []
    for i in range(len(query_ids)):
        answer_words = [vocabulary_index[index] for index in output_ids[i] if
                        index in vocabulary_index]  # 还有unknown不应该删，看看怎么办
        answer = ' '.join(answer_words)  # 这样是分词了的，如果标点怎么合？
        candi = {'answers': [answer], 'query_id': query_ids[i]}
        candidates.append(candi)
        i += 1
    return candidates

print 'setting: ', vocab_size, embedding_dim, hidden_size, batch_size, grad_clip, decoder_max_iteration, \
    initial_learning_rate, epoch_num, with_att, input_reverse, bidirection, rnn_type, input_mode

print 'read train texts...'
train_data_list = read_json(train_path)
train_query_ids, train_queries, train_passages, train_answers = get_ms_marco_dqa(train_data_list, input_mode)
print 'train datanum: ', len(train_data_list), len(train_answers)  # 82326 92489 (原只看有答案的90306，则有2000多无答案的）

print 'read vocab...'
# vocabulary = get_vocab(train_data_list)  # 看论文从训练数据收集的，也先只用训练数据？
# with open(vocab_filename, "w") as f:
#     json.dump(vocabulary, f)
with open(vocab_filename, 'r') as load_f:
    vocabulary = json.load(load_f)
vocabulary = sorted(vocabulary.items(), key=lambda e: e[1])  # 根据词汇表生成方法，序号越小都词频越高
vocabulary = vocabulary[:vocab_size]
vocabulary = dict(vocabulary)
# print vocabulary
print 'vocab size: ', vocab_size

print 'build model...'
total_vocab_size = vocab_size + 4  # !!!!
# with tf.variable_scope('root'):
train_model = Seq2SeqModel(total_vocab_size, embedding_dim, hidden_size, grad_clip, decoder_max_iteration,
                           batch_size, initial_learning_rate, with_att)
# with tf.variable_scope('root', reuse=True):
test_model = Seq2SeqModel(total_vocab_size, embedding_dim, hidden_size, grad_clip, decoder_max_iteration,
                          batch_size, initial_learning_rate, with_att, True)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

print 'generate train data...'
train_inputs_q = to_index(vocabulary, train_queries, False)
train_inputs_p = to_index(vocabulary, train_queries, False)
train_targets = to_index(vocabulary, train_answers)
del train_data_list, train_queries, train_answers

print 'train...'
train(train_model, train_inputs_q, train_inputs_p, train_targets, batch_size, epoch_num)
del train_inputs_q, train_inputs_p, train_targets

print 'read dev texts...'
dev_data_list = read_json(dev_path)
dev_query_ids, dev_queries, dev_passages = get_ms_marco_dqa(dev_data_list, input_mode, True)
print 'dev datanum: ', len(dev_data_list), len(dev_queries)  # 10047 ?；这里多答案的也都拆了
del dev_data_list

print 'generate dev data...'
dev_inputs_q = to_index(vocabulary, dev_queries, False)
dev_inputs_p = to_index(vocabulary, dev_passages, False)
# dev_targets = to_index(vocabulary, dev_answers)
# del dev_queries, dev_answers

print 'test (with dev data)...'
dev_output_ids = test(test_model, dev_inputs_q, dev_inputs_p, None, batch_size)

# 转成文本，输出文件，官方script评价
vocabulary_index = {v: k for k, v in vocabulary.items()}  # 倒过来便于索引
print 'output candidates json...'
candidates = generate_candidates(dev_output_ids, dev_query_ids)
write_json(candidates, dev_candi_path)

print 'read test texts...'
test_data_list = read_json(test_path)
test_query_ids, test_queries, test_passages = get_ms_marco_dqa(test_data_list, input_mode, True)
print 'test datanum: ', len(test_data_list)
del test_data_list

print 'generate test data...'
test_inputs_q = to_index(vocabulary, test_queries, False)
test_inputs_p = to_index(vocabulary, test_passages, False)
del test_queries

print 'test...'
test_output_ids = test(test_model, test_inputs_q, test_inputs_p, None, batch_size)

print 'output candidates json...'
candidates = generate_candidates(test_output_ids, test_query_ids)
write_json(candidates, test_candi_path)
print test_candi_path

sess.close()
