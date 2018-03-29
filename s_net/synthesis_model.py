# coding=utf-8

import tensorflow as tf


SOS_ID=2


class SynthesisModel:  # 生成部分
    def __init__(self, batch_size, vocab_size, embedding_dim, hidden_size, max_anslen, beam_search_size, dropout_rate,
                 initial_learning_rate, mode='train'):
        # inputs
        self.inputs_q = tf.placeholder(tf.int32, shape=[batch_size, None], name='inputs_q') # index了
        self.inputs_actual_length_q = tf.placeholder(tf.int32, [batch_size], name='inputs_actual_length_q')
        self.inputs_p = tf.placeholder(tf.int32, shape=[batch_size, None], name='inputs_p')
        self.inputs_actual_length_p = tf.placeholder(tf.int32, [batch_size], name='inputs_actual_length_p')
        self.starts = tf.placeholder(tf.float32, [batch_size, None, 1])  # 论文后面说维度50是啥？？这里不应该就是一个数，即每个词0或1？
        self.ends = tf.placeholder(tf.float32, [batch_size, None, 1])

        # targets
        if mode == 'train':
            self.targets_a = tf.placeholder(tf.int32, shape=[batch_size, None], name='targets_a')
            self.targets_actual_length_a = tf.placeholder(tf.int32, [batch_size], name='targets_actual_length_a')

        # embeddings
        # 如果要标点，则不能lookup，得自己设个词汇表embedding W，来更新；这里为方便先用下吧
        with tf.variable_scope('embedding', reuse=tf.AUTO_REUSE):
            embedding = tf.get_variable(name='embedding',
                                        initializer=tf.truncated_normal(shape=[vocab_size, embedding_dim],
                                                                        stddev=0.1))
            # 截断的产生正态分布的函数，值与均值的差值大于两倍标准差则重新生成

        inputs_embedded_q = tf.nn.embedding_lookup(embedding, self.inputs_q)
        inputs_embedded_p = tf.nn.embedding_lookup(embedding, self.inputs_p)
        if mode == 'train':
            targets_embedded_a = tf.nn.embedding_lookup(embedding, self.targets_a)

        inputs_concat_pos = tf.concat([self.starts, self.ends], axis=2)
        inputs_concat_p = tf.concat([inputs_embedded_p, inputs_concat_pos], axis=2)
        # print inputs_concat_pos, inputs_concat_p

        # question encoder
        with tf.variable_scope("q_encoder", reuse=tf.AUTO_REUSE):
            fcell_q = tf.nn.rnn_cell.GRUCell(hidden_size)
            bcell_q = tf.nn.rnn_cell.GRUCell(hidden_size)
            fcell_q = tf.contrib.rnn.DropoutWrapper(fcell_q, output_keep_prob=1 - dropout_rate)  # 有3个dropout，应该用哪个？？
            bcell_q = tf.contrib.rnn.DropoutWrapper(bcell_q, output_keep_prob=1 - dropout_rate)
            (fw_outputs_q, bw_outputs_q), (fw_final_state_q, bw_final_state_q) = \
                tf.nn.bidirectional_dynamic_rnn(cell_fw=fcell_q,
                                                cell_bw=bcell_q,
                                                inputs=inputs_embedded_q,
                                                sequence_length=self.inputs_actual_length_q,
                                                dtype=tf.float32)
            h_q = tf.concat((fw_outputs_q, bw_outputs_q), 2)
            # print h_q  # 输出是root，root_1的；但前面embedding是共享的
            # print 'bw_outputs_q', bw_outputs_q

        # passage encoder
        with tf.variable_scope("p_encoder", reuse=tf.AUTO_REUSE):
            fcell_p = tf.nn.rnn_cell.GRUCell(hidden_size)
            bcell_p = tf.nn.rnn_cell.GRUCell(hidden_size)
            fcell_p = tf.contrib.rnn.DropoutWrapper(fcell_p, output_keep_prob=1 - dropout_rate)  # 有3个dropout，应该用哪个？？
            bcell_p = tf.contrib.rnn.DropoutWrapper(bcell_p, output_keep_prob=1 - dropout_rate)
            (fw_outputs_p, bw_outputs_p), (fw_final_state_p, bw_final_state_p) = \
                tf.nn.bidirectional_dynamic_rnn(cell_fw=fcell_p,
                                                cell_bw=bcell_p,
                                                inputs=inputs_concat_p,
                                                sequence_length=self.inputs_actual_length_p,
                                                dtype=tf.float32)
            h_p = tf.concat((fw_outputs_p, bw_outputs_p), 2)
            # print h_p
            # print 'bw_outputs_p', bw_outputs_p

        # decoder
        with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE):
            with tf.variable_scope("initial_state"):
                w_d = tf.get_variable(name='w_d', shape=[hidden_size * 2, hidden_size * 2])
                bias = tf.get_variable(name='bias', shape=[1, hidden_size * 2])

                def compute_d_0(i, bw_p, bw_q, d_0_b):
                    bw_p_0_i = tf.reshape(bw_p[i][0], shape=[1, hidden_size])  # 第一个是这个0的吗
                    bw_q_0_i = tf.reshape(bw_q[i][0], shape=[1, hidden_size])
                    h_concat_i = tf.concat([bw_p_0_i, bw_q_0_i], axis=1)
                    d_0_i = tf.tanh(tf.matmul(h_concat_i, w_d) + bias)  # d_0的维度也可随便定的，用矩阵w_d调即可；按论文，这里应该150吧
                    d_0_b.write(i, d_0_i)
                    i = tf.add(i, 1)
                    return i, bw_p, bw_q, d_0_b

                d_0_b = tf.TensorArray(dtype=tf.float32, size=batch_size)
                c = lambda x, y, z, w: tf.less(x, batch_size)  # batch的循环
                b = lambda x, y, z, w: compute_d_0(x, y, z, w)
                i = tf.constant(0)  # batch号
                d_res = tf.while_loop(cond=c, body=b, loop_vars=(i, bw_outputs_p, bw_outputs_q, d_0_b))
                d_0 = d_res[-1].stack()
                # print 'd_0', d_0

            with tf.variable_scope("attention_decoder_ouput", reuse=tf.AUTO_REUSE):
                w_a = tf.get_variable(name='w_a', shape=[hidden_size * 2, hidden_size * 2])
                u_a = tf.get_variable(name='u_a', shape=[hidden_size * 2, hidden_size * 2])
                v_a = tf.get_variable(name='v_a', shape=[hidden_size * 2, 1])

                cell_d = tf.nn.rnn_cell.GRUCell(hidden_size * 2)  # 这个维度？？w,c,d怎么结合的；不对啊，d就是hidden_size大小吧

                w_r = tf.get_variable(name='w_r', shape=[embedding_dim, hidden_size * 2])  # 注意要和词向量乘，维度不同
                u_r = tf.get_variable(name='u_r', shape=[hidden_size * 2, hidden_size * 2])
                v_r = tf.get_variable(name='v_r', shape=[hidden_size * 2, hidden_size * 2])
                w_o = tf.get_variable(name='w_o', shape=[hidden_size, vocab_size])  # 是这里调吗？

                def attention_step(h_p_i, h_q_i, d_t_1):  # w_t应是当前输入词的embedding，也就是上一个输出的词
                    h = tf.concat([h_p_i, h_q_i], axis=0)  # 不对啊词数都不同，这里怎么拼的？？直接第一维拼？嗯这里就按这种做法
                    # 感觉是这样的：拼是p+q，然后每个词做attention，所以还是一起乘，对的
                    # print d_t_1, h
                    sum_d = tf.matmul(d_t_1, w_a) + tf.matmul(h, u_a)  # 矩阵了；h的行数是是p+q词数，列数是向量维度
                    # print 'h, sum_d', h, sum_d
                    s_t = tf.matmul(tf.tanh(sum_d), v_a)  # 列向量
                    a_t = tf.nn.softmax(s_t)
                    a_t = tf.transpose(a_t)  # 转成行向量
                    c_t = tf.matmul(a_t, h)
                    # print 'c_t', c_t
                    return c_t

                def compute_m_t_j(j, r_t, m_t):
                    r_m = tf.maximum(r_t[0][j], r_t[0][j + 1])
                    # print r_m
                    m_t.write(j, r_m)
                    j = tf.add(j, 1)
                    return j, r_t, m_t

                def maxout_hidden_layer(r_t):
                    m_t = tf.TensorArray(dtype=tf.float32, size=hidden_size)
                    c = lambda x, y, z: tf.less(x, hidden_size * 2 - 1)
                    b = lambda x, y, z: compute_m_t_j(x, y, z)
                    j = tf.constant(0)
                    m_t_res = tf.while_loop(cond=c, body=b, loop_vars=(j, r_t, m_t))
                    m_t = m_t_res[-1].stack()
                    m_t = tf.reshape(m_t, shape=[1, hidden_size])
                    # print 'm_t', m_t
                    return m_t

                # train中两部分可分开；test中有依赖关系不可分开
                if mode == 'train':
    
                    def output_step_train(w_t_1, c_t, d_t):
                        r_t = tf.matmul(w_t_1, w_r) + tf.matmul(c_t, u_r) + tf.matmul(d_t, v_r)  # 其实w,c,d维度都不需一样， 用参数矩阵调即可；论文中说r_t是2d维，到底应该多少
                        # print 'r_t', r_t
                        m_t = maxout_hidden_layer(r_t)  # maxout hidden layer
                        prob_t = tf.nn.softmax(tf.matmul(m_t, w_o))  # 行向量
                        # w_t_index = tf.argmax(prob_t, axis=1)  # argmax??找出id，再embedding[id]
                        # print 'w_t_index', w_t_index
                        # w_t = tf.reshape(embedding[w_t_index[0]], shape=[1, -1])
                        return prob_t
    
                    def one_step_train(t, h_p_i, h_q_i, ta_i, d_t, w_t, prob_a):  # 不对，train的时候w_t是输入的真实target的词
                        c_t = attention_step(h_p_i, h_q_i, d_t)
    
                        # print 'w_t, c_t, d_t ', w_t, c_t, d_t
                        w_c_d = tf.concat([w_t, c_t, d_t], axis=1)
                        # print 'w_c_d, d_t', w_c_d, d_t
                        out, d_tp1 = cell_d(inputs=w_c_d, state=d_t)
    
                        prob_t = output_step_train(w_t, c_t, d_tp1)
                        prob_a.write(t, prob_t)
                        # print 'prob_t', prob_t

                        # print ta_i[t]
                        w_tp1 = tf.reshape(ta_i[t], shape=[1, -1])  # -1是SOS，则从0开始是答案
    
                        t = tf.add(t, 1)
    
                        return t, h_p_i, h_q_i, ta_i, d_tp1, w_tp1, prob_a
    
                    def one_answer_train(i, prob_b):
                        c = lambda x, y, z, m, n, p, q: tf.less(x, self.targets_actual_length_a[i])  # 改这里，这里不应该是文章长，应该是答案长；应该根据生成词判断终止条件，也就是这里搞个函数，但回头再说，这里先简化
                        b = lambda x, y, z, m, n, p, q: one_step_train(x, y, z, m, n, p, q)
                        t = tf.constant(0)  # batch号
                        h_p_i = h_p[i]
                        h_q_i = h_q[i]
                        d_0_i = d_0[i]
                        ta_i = targets_embedded_a[i]
                        # print 'd_0_i', d_0_i
                        w_0 = tf.reshape(embedding[SOS_ID], shape=[1, -1])  # 应是start标签的embedding；对的对的
                        prob_a = tf.TensorArray(dtype=tf.float32, size=self.targets_actual_length_a[i])
                        # print prob_a
                        res = tf.while_loop(cond=c, body=b, loop_vars=(t, h_p_i, h_q_i, ta_i, d_0_i, w_0, prob_a))
                        temp = tf.squeeze(res[-1].stack(), axis=1)  # 具体做了什么，stack和squeeze的作用？
                        # print res[-1].stack(), temp
                        prob_b.write(i, temp)
                        i = tf.add(i, 1)
                        return i, prob_b
    
                    prob_b = tf.TensorArray(dtype=tf.float32, size=batch_size)  # 应是batch，答案t长，词汇表长
                    c = lambda x, y: tf.less(x, batch_size)  # batch的循环
                    b = lambda x, y: one_answer_train(x, y)
                    i = tf.constant(0)  # batch号
                    prob_b_res = tf.while_loop(cond=c, body=b, loop_vars=(i, prob_b))
                    prob = prob_b_res[-1].stack()
                    # print 'prob', prob

                if mode == 'test':  # 要用beam search;先用最后只取概率最大的

                    def output_step_test(w_t_1, c_t, d_t, seq_w_ts_j, seq_w_ts_indices_j, seq_prob_ts_j):
                        # 这里需要考虑 输入！输入！有没有EOS之类的了
                        # 如果输入是结束符

                        # 就现在改了吧
                        def eos():
                            w_tp1s_j = tf.reshape(embedding[EOS_ID], shape=[1, -1])  # 就一直上EOS吧
                            seq_w_tp1s_j = tf.concat([seq_w_ts_j, w_tp1s_j], axis=0)
                            seq_w_tp1s_j = tf.expand_dims(seq_w_tp1s_j, 0)
                            w_tp1s_indices_j = tf.reshape([EOS_ID], shape=[1, 1])
                            seq_w_tp1s_indices_j = tf.expand_dims(seq_w_ts_indices_j, 0)
                            seq_w_tp1s_indices_j = tf.concat([seq_w_tp1s_indices_j, w_tp1s_indices_j], axis=1)
                            seq_prob_tp1s_j = tf.reshape(seq_prob_ts_j, shape=[1, 1])
                            # print seq_w_tp1s_j, seq_w_tp1s_indices_j, seq_prob_tp1s_j
                            return seq_w_tp1s_j, seq_w_tp1s_indices_j, seq_prob_tp1s_j

                        def not_eos():
                            r_t = tf.matmul(w_t_1, w_r) + tf.matmul(c_t, u_r) + tf.matmul(d_t, v_r)
                            # print 'r_t', r_t
                            m_t = maxout_hidden_layer(r_t)  # maxout hidden layer
                            prob_t = tf.nn.softmax(tf.matmul(m_t, w_o))  # 行向量

                            # 先prob乘，选最大beam_size个
                            seq_prob_tp1s_all = seq_prob_ts_j[0] * prob_t  # 应该是这么乘吧；这里取数
                            rs = tf.nn.top_k(seq_prob_tp1s_all, beam_search_size)
                            # 乘完的筛选差不多
                            # prob
                            seq_prob_tp1s = tf.reshape(rs.values, shape=[-1, 1])  # 直接替换，变列向量
                            # id
                            seq_w_tp1_indices = tf.concat([tf.reshape(seq_w_ts_indices_j, shape=[1, -1])]
                                                          * beam_search_size, axis=0)  # 搞成行向量，重复12次，拼接
                            seq_w_tp1_indices = tf.concat([seq_w_tp1_indices, tf.transpose(rs.indices)], axis=1)
                            # embedding
                            seq_w_tp1s = tf.concat([tf.expand_dims(seq_w_ts_j, 0)] * beam_search_size,
                                                   axis=0)  # 是不是有别的办法，这样写好奇怪
                            w_tp1s = tf.nn.embedding_lookup(embedding, rs.indices[0])
                            w_tp1s = tf.expand_dims(w_tp1s, 1)  # 升维，第1维加
                            seq_w_tp1s = tf.concat([seq_w_tp1s, w_tp1s], axis=1)
                            # print 'seq_w_tp1s, seq_w_tp1_indices, seq_prob_tp1s', seq_w_tp1s, seq_w_tp1_indices, seq_prob_tp1s

                            return seq_w_tp1s, seq_w_tp1_indices, seq_prob_tp1s

                        result = tf.cond(tf.equal(seq_w_ts_indices_j[-1], tf.constant(EOS_ID)),
                                         lambda: eos(), lambda: not_eos())
                        # print 'result', result
                        return result

                    def get_best_seqs(seq_w_ts, seq_w_ts_indices, seq_prob_ts):
                        # 可以转成字典，排序求最大，再取得相应下标？？
                        # 或连接，排序，再拆开
                        # 啊啊啊啊有现成函数
                        seq_prob_ts = tf.reshape(seq_prob_ts, shape=[1, -1])
                        rs = tf.nn.top_k(seq_prob_ts, beam_search_size)  # 这里得是行向量
                        # print rs.indices
                        # 下面根据indices切片
                        best_seq_prob_ts = tf.reshape(rs.values, shape=[-1, 1])  # 这是行向量，看看改不改；改列向量吧
                        best_seq_w_ts = tf.gather(seq_w_ts, rs.indices[0])  # 都是对于第0维取
                        best_seq_w_ts_indices = tf.gather(seq_w_ts_indices, rs.indices[0])
                        # print 'best_seq_w_ts, best_seq_w_ts_indices, best_seq_prob_ts', \
                        #     best_seq_w_ts, best_seq_w_ts_indices, best_seq_prob_ts
                        return best_seq_w_ts, best_seq_w_ts_indices, best_seq_prob_ts

                    def one_step_test(t, h_p_i, h_q_i, d_ts, best_seq_w_ts, best_seq_w_ts_indices, best_seq_prob_ts):
                        d_tp1s = []
                        seq_w_tp1s_list = []
                        seq_w_tp1_indices_list = []
                        seq_prob_tp1s_list = []

                        for j in range(beam_search_size):  # 这里看看要不要也改成lambda；如果用lambda似乎要arraywrite了
                            seq_w_ts_j = best_seq_w_ts[j]
                            seq_w_ts_indices_j = best_seq_w_ts_indices[j]
                            seq_prob_ts_j = best_seq_prob_ts[j]  # 先不取到数吧
                            # print 'seq_w_ts_j, seq_w_ts_indices_j, seq_prob_ts_j ', seq_w_ts_j, seq_w_ts_indices_j, seq_prob_ts_j
                            w_t = tf.reshape(seq_w_ts_j[-1], shape=[1, -1])  # 取最后一个？不对，返回的是全部啊；要么在外面拼，要么在里面拼
                            d_t = tf.reshape(d_ts[j], shape=[1, -1])  # 这也复数
                            c_t = attention_step(h_p_i, h_q_i, d_t)
                            w_c_d = tf.concat([w_t, c_t, d_t], axis=1)
                            # print 'w_t, c_t, d_t, w_c_d ', w_t, c_t, d_t, w_c_d
                            out, d_tp1 = cell_d(inputs=w_c_d, state=d_t)  # 但是连d都不一样，这咋搞
                            d_tp1s.append(d_tp1)
                            seq_w_tp1s, seq_w_tp1_indices, seq_prob_tp1s = \
                                output_step_test(w_t, c_t, d_tp1, seq_w_ts_j, seq_w_ts_indices_j, seq_prob_ts_j)
                            # 输出得12*12后再一起筛选成12；具体咋筛呢；还要这里w不能只是当前词了，得是从1至当前序列的所有词
                            # 每个12个就够了，自己都进不了前12的，其他也进不了
                            seq_w_tp1s_list.append(seq_w_tp1s)
                            seq_w_tp1_indices_list.append(seq_w_tp1_indices)
                            seq_prob_tp1s_list.append(seq_prob_tp1s)
                        # 拼起来
                        d_tp1s = tf.concat(d_tp1s, axis=0)
                        seq_w_tp1s = tf.concat(seq_w_tp1s_list, axis=0)
                        seq_w_tp1_indices = tf.concat(seq_w_tp1_indices_list, axis=0)
                        seq_prob_tp1s = tf.concat(seq_prob_tp1s_list, axis=0)  # 看如果是列向量就是这样
                        # 筛选
                        best_seq_w_tp1s, best_seq_w_tp1_indices, best_seq_prob_tp1s = \
                            get_best_seqs(seq_w_tp1s, seq_w_tp1_indices, seq_prob_tp1s)

                        # 求下not_all_eos；判断最后是不是EOS都
                        # ???不会求啊。。。。先不管他，直接搞到最后好了

                        t =tf.add(t, 1)
                        return t, h_p_i, h_q_i, d_tp1s, best_seq_w_tp1s, best_seq_w_tp1_indices, best_seq_prob_tp1s
                    # 每次应返回到目前为止概率最大的beam_size个序列（ids，embeddings）及各自概率
                    # 第一步就是一个SOS，但后面是beam_size的输入了，但也有可能有重复的词；t=0在函数里分布对待好了

                    def output_step_test_t_0(w_0, c_0, d_1):
                        r_0 = tf.matmul(w_0, w_r) + tf.matmul(c_0, u_r) + tf.matmul(d_1, v_r)
                        # 其实w,c,d维度都不需一样， 用参数矩阵调即可；论文中说r_t是2d维，到底应该多少
                        # print 'r_0', r_0
                        m_t = maxout_hidden_layer(r_0)  # maxout hidden layer
                        prob_0 = tf.nn.softmax(tf.matmul(m_t, w_o))  # 行向量

                        rs = tf.nn.top_k(prob_0, beam_search_size)
                        # prob
                        seq_prob_1s = tf.reshape(rs.values, shape=[-1, 1])
                        # print 'seq_prob_1s', seq_prob_1s
                        # id
                        seq_w_0_indices = tf.reshape(tf.constant([SOS_ID] * beam_search_size), shape=[-1, 1])  # 搞成列向量
                        # print 'seq_w_0_indices', seq_w_0_indices
                        seq_w_1_indices = tf.concat([seq_w_0_indices, tf.transpose(rs.indices)], axis=1)  # e二维的
                        # embedding
                        seq_w_0s = tf.concat([w_0] * beam_search_size, axis=0)
                        seq_w_0s = tf.expand_dims(seq_w_0s, 1)  # 升维，第1维加
                        w_1s = tf.nn.embedding_lookup(embedding, rs.indices[0])
                        w_1s = tf.expand_dims(w_1s, 1)  # 升维，第1维加
                        seq_w_1s = tf.concat([seq_w_0s, w_1s], axis=1)  # SOS 拼上相应词
                        # print 'seq_w_1', seq_w_1s

                        return seq_w_1s, seq_w_1_indices, seq_prob_1s

                    def one_step_test_t_0(w_0, d_0, h_p_i, h_q_i):
                        c_0 = attention_step(h_p_i, h_q_i, d_0)
                        w_c_d = tf.concat([w_0, c_0, d_0], axis=1)
                        # print 't==0 ', w_0, c_0, d_0, w_c_d
                        out, d_1 = cell_d(inputs=w_c_d, state=d_0)
                        best_seq_w_1s, best_seq_w_1_indices, best_seq_prob_1s \
                            = output_step_test_t_0(w_0, c_0, d_1)
                        d_1s = tf.concat([d_1] * beam_search_size, axis=0)  # 重复下；行向量拼的
                        return d_1s, best_seq_w_1s, best_seq_w_1_indices, best_seq_prob_1s

                    def stop_condition(t, best_seq_w_t_indices):  # 应是到最大长度了，或前beam_size个概率最大的序列全到结束符了
                        w_t_indices = best_seq_w_t_indices[:, -1]  # 先抽出每个序列最后一个词的indices，直接切片即可吧
                        all_eos = tf.constant([EOS_ID]*beam_search_size)  # 拼出EOS的
                        cond1 = tf.equal(w_t_indices, all_eos)  # 然后equals进行比较
                        cond1 = tf.logical_not(tf.reduce_all(cond1))  # True不终止，False终止
                        cond2 = t < max_anslen  # True不终止，False终止
                        cond = tf.logical_and(cond1, cond2)  # 都True才不终止，一个False即终止
                        # print cond
                        return cond

                    def one_answer(i, a_ids_b):
                        # t=0时单独搞一个
                        h_p_i = h_p[i]
                        h_q_i = h_q[i]
                        d_0_i = d_0[i]
                        w_0 = tf.reshape(embedding[SOS_ID], shape=[1, -1])
                        d_1s, best_seq_w_1s, best_seq_w_1_indices, best_seq_prob_1s = one_step_test_t_0(w_0, d_0_i, h_p_i, h_q_i)
                        # t=1开始
                        t = tf.constant(1)
                        c = lambda x, y, z, m, n, p, q: stop_condition(x, p)
                        b = lambda x, y, z, m, n, p, q: one_step_test(x, y, z, m, n, p, q)
                        res = tf.while_loop(cond=c, body=b, loop_vars=[t, h_p_i, h_q_i, d_1s,
                                                                       best_seq_w_1s, best_seq_w_1_indices,
                                                                       best_seq_prob_1s],
                                            shape_invariants=[t.get_shape(), h_p_i.get_shape(),
                                                              h_q_i.get_shape(), d_1s.get_shape(),
                                                              tf.TensorShape([beam_search_size, None, embedding_dim]),
                                                              tf.TensorShape([beam_search_size, None]),
                                                              tf.TensorShape([beam_search_size, 1])])
                                            # 可能还有问题，后面不加词了是不是需要padding；应该要，里面维度要一样
                        # 到最后一步取最大概率的序列，填入a_ids_b
                        seq_probs = res[-1]
                        # w_ts = res[-3].stack()
                        w_ts_indices = res[-2]
                        best_seq_no = tf.argmax(seq_probs)
                        best_seq_no = best_seq_no[0]
                        # best_w_ts = w_ts[best_seq_no]
                        best_a_ids = w_ts_indices[best_seq_no]
                        print 'best_seq_no, best_a_ids ', best_seq_no, best_a_ids
                        a_ids_b.write(i, best_a_ids)
                        i = tf.add(i, 1)
                        return i, a_ids_b

                    answers_ids_b = tf.TensorArray(dtype=tf.int32, size=batch_size)
                    c = lambda x, y: tf.less(x, batch_size)  # batch的循环
                    b = lambda x, y: one_answer(x, y)
                    i = tf.constant(0)  # batch号
                    answers_ids_b_res = tf.while_loop(cond=c, body=b, loop_vars=(i, answers_ids_b))
                    # print 'prob', prob
                    self.answers_ids = answers_ids_b_res[-1].stack()
                    print 'answers_ids', self.answers_ids

        # loss, train_op
        if mode == 'train':
            with tf.variable_scope('loss', reuse=tf.AUTO_REUSE):

                def compute_loss_step(t, ta_index_i, prob_i, loss_a):
                    ta_index_t = ta_index_i[t]
                    prob_t = prob_i[t][ta_index_t]
                    # print prob_t
                    loss_a *= prob_t
                    t = tf.add(t, 1)
                    return t, ta_index_i, prob_i, loss_a

                def compute_loss_batch(i, loss):
                    t = tf.constant(0)
                    prob_i = prob[i]
                    ta_index_i = self.targets_a[i]
                    loss_a = tf.constant(1.0)
                    c = lambda x, y, z, m: tf.less(x, self.targets_actual_length_a[i])
                    b = lambda x, y, z, m: compute_loss_step(x, y, z, m)
                    loss_res = tf.while_loop(cond=c, body=b, loop_vars=(t, ta_index_i, prob_i, loss_a))
                    loss_a = loss_res[-1]
                    loss += loss_a
                    # print loss_a, loss
                    i = tf.add(i, 1)
                    return i, loss

                loss = tf.constant(0.0)
                c = lambda x, y: tf.less(x, batch_size)
                b = lambda x, y: compute_loss_batch(x, y)
                loss_res = tf.while_loop(cond=c, body=b, loop_vars=(i, loss))
                self.loss = loss_res[-1]
                # print 'loss', self.loss

            with tf.variable_scope('train_op', reuse=tf.AUTO_REUSE):
                tvars = tf.trainable_variables()
                # print tvars
                # grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), grad_clip)
                optimizer = tf.train.AdamOptimizer(initial_learning_rate)
                grads_and_vars = optimizer.compute_gradients(self.loss)
                self.train_op = optimizer.apply_gradients(grads_and_vars)  # 还是报一样的错
                # self.train_op = optimizer.apply_gradients(zip(grads, tvars))
                print self.train_op


model = SynthesisModel(batch_size=128, vocab_size=300, embedding_dim=300, hidden_size=75, max_anslen=220,
                       beam_search_size=12, dropout_rate=0.1, initial_learning_rate=1e-3)

