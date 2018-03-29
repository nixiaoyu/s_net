# coding=utf-8

import tensorflow as tf


class ExtractionModel:
    def __init__(self, batch_size, hidden_size, embedding_dim, dropout_rate, grad_clip, initial_learning_rate,
                 vocab_size, mode='train'):
        # 还是改成输入id，lookup
        # inputs
        self.inputs_q = tf.placeholder(tf.float32, shape=[batch_size, None], name='inputs_q')
        # self.inputs_embedded_q = tf.placeholder(tf.float32, shape=[batch_size, None, embedding_dim], name='inputs_embedded_q')
        self.inputs_actual_length_q = tf.placeholder(tf.int32, [batch_size],
                                                             name='inputs_actual_length')  # 每句输入的实际长度，除了padding
        self.inputs_concat_p = tf.placeholder(tf.float32, shape=[batch_size, None], name='inputs_concat_p')  # 干脆先全部都concat吧；不对，这样这里还要padding；还是原来的搞，切片吧
        # self.inputs_embedded_concat_p = tf.placeholder(tf.float32, shape=[batch_size, None, embedding_dim], name='inputs_embedded_concat_p')  # 干脆先全部都concat吧；不对，这样这里还要padding；还是原来的搞，切片吧
        self.inputs_actual_length_concat_p = tf.placeholder(tf.int32, [batch_size],
                                                             name='inputs_actual_length_concat_p')
        # passage ranking
        self.passage_numbers = tf.placeholder(tf.int32, [batch_size], name='passage_numbers')
        self.passage_word_numbers = tf.placeholder(tf.int32, [batch_size, None], name='passage_word_numbers')

        # 还有char也没搞；我看也可以用现成的

        # targets
        if mode != 'test':
            self.start_position = tf.placeholder(tf.int32, [batch_size])
            self.end_position = tf.placeholder(tf.int32, [batch_size])
            # self.y_1 = tf.placeholder(tf.float32, [None])
            # self.y_2 = tf.placeholder(tf.float32, [None])
            # passage_ranking
            self.passage_y = tf.placeholder(tf.int32, [batch_size, None], name='passage_y')  # 也需要padding；实际长度在上面

        with tf.variable_scope('embedding', reuse=tf.AUTO_REUSE):
            self.embedding_pl = tf.placeholder(tf.float32, shape=[vocab_size, embedding_dim], name='inputs_q')
            embedding = tf.get_variable(name='embedding', trainable=False, initializer=self.embedding_pl)  # 这里改成放入

        self.inputs_embedded_q = tf.nn.embedding_lookup(embedding, self.inputs_q)
        self.inputs_embedded_concat_p = tf.nn.embedding_lookup(embedding, self.inputs_concat_p)

        with tf.variable_scope("q_encoder", reuse=tf.AUTO_REUSE):
            fcell_q = tf.nn.rnn_cell.GRUCell(hidden_size)
            bcell_q = tf.nn.rnn_cell.GRUCell(hidden_size)
            fcell_q = tf.contrib.rnn.DropoutWrapper(fcell_q, output_keep_prob=1-dropout_rate)  # 有3个dropout，应该用哪个？？
            bcell_q = tf.contrib.rnn.DropoutWrapper(bcell_q, output_keep_prob=1-dropout_rate)
            (fw_outputs_q, bw_outputs_q), (fw_final_state_q, bw_final_state_q) = \
            tf.nn.bidirectional_dynamic_rnn(cell_fw=fcell_q,
                                            cell_bw=bcell_q,
                                            inputs=self.inputs_embedded_q,
                                            sequence_length=self.inputs_actual_length_q,
                                            dtype=tf.float32)
            u_q = tf.concat((fw_outputs_q, bw_outputs_q), 2)
            # print u_q  # 输出是root，root_1的；但前面embedding是共享的

        with tf.variable_scope("p_encoder", reuse=tf.AUTO_REUSE):
            fcell_p = tf.nn.rnn_cell.GRUCell(hidden_size)
            bcell_p = tf.nn.rnn_cell.GRUCell(hidden_size)
            fcell_p = tf.contrib.rnn.DropoutWrapper(fcell_p, output_keep_prob=1-dropout_rate)  # 有3个dropout，应该用哪个？？
            bcell_p = tf.contrib.rnn.DropoutWrapper(bcell_p, output_keep_prob=1-dropout_rate)

            def p_encoder_one_p(j, start, end, inputs_embedded_concat_p_i, p_w_num_i, fw_p_i, bw_p_i):
                inputs_embedded_p = tf.expand_dims(inputs_embedded_concat_p_i[start:end, :], 0)
                p_w_num = tf.expand_dims(p_w_num_i[j], 0)
                with tf.variable_scope("p_encoder_one_p", reuse=tf.AUTO_REUSE):
                    (fw_outputs_p, bw_outputs_p), (fw_final_state_p, bw_final_state_p) = \
                        tf.nn.bidirectional_dynamic_rnn(cell_fw=fcell_p,
                                                        cell_bw=bcell_p,
                                                        inputs=inputs_embedded_p,
                                                        sequence_length=p_w_num,
                                                        dtype=tf.float32)
                    # 其实是不是一样的呢，反正也都用；可能还是不太一样，一篇文章第一个词不依赖上一篇文章最后词
                fw_p_i.write(j, fw_outputs_p)
                bw_p_i.write(j, bw_outputs_p)
                start = end
                j = tf.add(j, 1)
                end = p_w_num_i[j]
                return j, start, end, inputs_embedded_concat_p_i, p_w_num_i, fw_p_i, bw_p_i

            def p_encoder_one_q(i, fw_p_b, bw_p_b):
                j = tf.constant(0)
                p_w_num_i = self.passage_word_numbers[i]
                start = tf.constant(0)
                end = p_w_num_i[0]
                inputs_embedded_concat_p_i = self.inputs_embedded_concat_p[i]
                p_num_i = self.passage_numbers[i]
                fw_p_i = tf.TensorArray(dtype=tf.float32, size=p_num_i)
                bw_p_i = tf.TensorArray(dtype=tf.float32, size=p_num_i)
                c = lambda x, y, z, m, n, p, q: tf.less(x, p_num_i)
                b = lambda x, y, z, m, n, p, q: p_encoder_one_p(x, y, z, m, n, p, q)
                u_p_i_res = tf.while_loop(cond=c, body=b, loop_vars=(j, start, end, inputs_embedded_concat_p_i, p_w_num_i, fw_p_i, bw_p_i))
                fw_p_i = u_p_i_res[-2].stack()
                bw_p_i =u_p_i_res[-1].stack()
                # print 'fw_p_i, bw_p_i', fw_p_i, bw_p_i
                fw_p_i = tf.reshape(fw_p_i, shape=[-1, hidden_size])  # 就是降了一维
                bw_p_i = tf.reshape(bw_p_i, shape=[-1, hidden_size])  # 就是降了一维
                # print 'fw_p_i, bw_p_i', fw_p_i, bw_p_i
                fw_p_b.write(i, fw_p_i)
                bw_p_b.write(i, bw_p_i)
                i = tf.add(i, 1)
                return i, fw_p_b, bw_p_b

            i = tf.constant(0)
            fw_p_b = tf.TensorArray(dtype=tf.float32, size=batch_size)
            bw_p_b = tf.TensorArray(dtype=tf.float32, size=batch_size)
            c = lambda x, y, z: tf.less(x, batch_size)  # 不用调，切第一维即可；不对，关键每个batch的切法不同；还是分开吧
            b = lambda x, y, z: p_encoder_one_q(x, y, z)
            u_p_b_res = tf.while_loop(cond=c, body=b, loop_vars=(i, fw_p_b, bw_p_b))
            fw_p = u_p_b_res[-2].stack()
            bw_p = u_p_b_res[-2].stack()
            # print 'fw_p, bw_p', fw_p, bw_p
            u_p = tf.concat((fw_p, bw_p), 2)
            # print 'u_p', u_p
            # 要把它弄成和原来一样的形状，回头要分再切片即可

        with tf.variable_scope("q_p_attention", reuse=tf.AUTO_REUSE):
            w_q_u = tf.get_variable(name='w_q_u', shape=[hidden_size*2, hidden_size*2])
            w_p_u = tf.get_variable(name='w_p_u', shape=[hidden_size*2, hidden_size*2])
            # w_p_v = tf.get_variable(name='w_p_v', shape=[hidden_size*2, hidden_size*2])
            v = tf.get_variable(name='v', shape=[hidden_size*2, 1])
            w_g = tf.get_variable(name='w_g', shape=[hidden_size*4, hidden_size*4])  # 这里是又拼接了一把的
            cell_v = tf.nn.rnn_cell.GRUCell(hidden_size*2)

            # passage中第t个词的attention
            def attention_step(t, q_i, p_i, len_q_i, state, v_p_p):
                p_i_t = tf.reshape(p_i[t], [1, -1])  # ！！注意可用-1，怎么忘了；变1行
                q_i_t = tf.slice(q_i, begin=[0, 0], size=[len_q_i, hidden_size*2])  # 哦是为了去掉padding的部分

                # sum_t = tf.matmul(w_q_u, q_i_t) + tf.matmul(w_p_u, p_i_t)  # 是可以的！！！
                # + tf.matmul(w_p_v, tf.transpose(v_p_t_1)  # 看看加不加
                sum_t = tf.matmul(q_i_t, w_q_u) + tf.matmul(p_i_t, w_p_u)  # 少一点转置，减少计算量吧
                # print sum_t  # (?,150)
                s_t = tf.matmul(tf.tanh(sum_t), v)  # 列向量，问题长
                # print s_t # (?,1),?应该最后填了是150
                a_t = tf.nn.softmax(s_t)
                a_t = tf.reshape(a_t, [-1, 1])
                c_q_t = tf.transpose(tf.matmul(q_i_t, a_t))  # 行向量
                # print 'c_q_t', c_q_t  #  (1,?),同样应150

                p_c = tf.concat([p_i_t, c_q_t], axis=1)  # 行向量， 维度 hidden_size*4
                g_t = tf.nn.sigmoid(tf.matmul(p_c, w_g))  # 维度 hidden_size*4，行向量
                # print p_c, g_t  # (1,?), 应300；(1,300)
                # 方法：用门的输出向量按元素乘以我们需要控制的那个向量 原理：门的输出是 0到1 之间的实数向量，
                p_c_gated = g_t * p_c  # 应该直接乘就行
                # print p_c_gated  # 行向量，(1,300)
                out, next_state = cell_v(inputs=p_c_gated, state=state)  # out和state一样？？
                # print 'state', state
                # print 'out', out
                v_p_p = v_p_p.write(t, out)  # 这块，看看咋分开

                t = tf.add(t, 1)

                return t, q_i, p_i, len_q_i, state, v_p_p

            # 就是i,t->i,j,t
            def atention_one_p(j, q_i, p_i, len_q_i, p_w_num_i, v_p_q):
                state = cell_v.zero_state(batch_size=1, dtype=tf.float32)  # 不对，双向的；？
                p_w_num_i_j = p_w_num_i[j]
                v_p_p = tf.TensorArray(dtype=tf.float32, size=p_w_num_i_j)
                t = tf.constant(0)

                c = lambda a, x, y, z, s, u: tf.less(a, p_w_num_i_j)
                b = lambda a, x, y, z, s, u: attention_step(a, x, y, z, s, u)
                v_p_p_res = tf.while_loop(cond=c, body=b, loop_vars=(t, q_i, p_i, len_q_i, state, v_p_p))

                v_p_p = v_p_p_res[-1].stack()
                # print 'v_p_p', v_p_p
                v_p_q.write(j, v_p_p)

                return j, q_i, p_i, len_q_i, p_w_num_i, v_p_q

            # 整个passage的attention
            def atention_one_q(i, v_p_b):
                p_i = u_p[i]  # 一个question
                q_i = u_q[i]  # 对应的passage
                len_q_i = self.inputs_actual_length_q[i]
                # print state
                j = tf.constant(0)
                p_num_i = self.passage_numbers[i]
                p_w_num_i = self.passage_word_numbers[i]
                v_p_q = tf.TensorArray(dtype=tf.float32, size=p_num_i)

                c = lambda a, x, y, z, s, u: tf.less(a, p_num_i)
                b = lambda a, x, y, z, s, u: atention_one_p(a, x, y, z, s, u)
                v_p_q_res = tf.while_loop(cond=c, body=b, loop_vars=(j, q_i, p_i, len_q_i, p_w_num_i, v_p_q))

                v_p_q = v_p_q_res[-1].stack()
                # print 'v_p_q', v_p_q
                v_p_q = tf.reshape(v_p_q, shape=[-1, hidden_size*2])  # 应该是什么shape？
                # print 'v_p_q', v_p_q
                v_p_b.write(i, v_p_q)
                # print 'temp', temp

                i = tf.add(i, 1)

                return i, v_p_b

            v_p_b = tf.TensorArray(dtype=tf.float32, size=batch_size)  # 存放batch中每条的结果
            c = lambda x, y: tf.less(x, batch_size)  # batch的循环
            b = lambda x, y: atention_one_q(x, y)
            i = tf.constant(0)  # batch号
            v_p_b_res = tf.while_loop(cond=c, body=b, loop_vars=(i, v_p_b))  # 这个会先不循环，而后面用for则会循环
            v_p = v_p_b_res[-1].stack()  # 是v_p；应就是把多个array拼成高一维的一个array
            # print 'v_p', v_p

        # with tf.variable_scope("self-matching"): # 这里s_net似乎删掉了r_net的self-matching部分

        with tf.variable_scope("output_layer", reuse=tf.AUTO_REUSE):
            # 先算h的初始状态r_q
            # 算a，p，c
            # 用c做输入，下个h

            with tf.variable_scope("intial_state", reuse=tf.AUTO_REUSE):
                w_u_q = tf.get_variable(name='w_u_q', shape=[hidden_size * 2, hidden_size * 2])
                w_q_v = tf.get_variable(name='w_q_v', shape=[hidden_size * 2, hidden_size * 2])
                v_q_r = tf.get_variable(name='v_q_r', shape=[1, hidden_size * 2])  # 好像是向量吧，难道是矩阵？？
                v2 = tf.get_variable(name='v2', shape=[hidden_size * 2, 1])

                def attention_r_q(i, r_q_b):
                    q_i = u_q[i]
                    # print 'q_i', q_i
                    len_q_i = self.inputs_actual_length_q[i]
                    q_i = tf.slice(q_i, begin=[0, 0], size=[len_q_i, hidden_size * 2])  # 直接列表那样也可以。像下文那样；先不改了吧
                    # print 'q_i', q_i
                    sum_q = tf.matmul(q_i, w_u_q) + tf.matmul(v_q_r, w_q_v)
                    # print sum_q
                    s = tf.matmul(tf.tanh(sum_q), v2)
                    # print s
                    a = tf.nn.softmax(s)
                    a = tf.reshape(a, [-1, 1])
                    r_q_i = tf.transpose(tf.matmul(tf.transpose(q_i), a))  # 还是转成行向量
                    # print 'r_q_i', r_q_i  # 应该还是hidden*2
                    r_q_b.write(i, r_q_i)
                    i = tf.add(i, 1)
                    return i, r_q_b

                r_q_b = tf.TensorArray(dtype=tf.float32, size=batch_size)
                c = lambda x, y: tf.less(x, batch_size)  # batch的循环
                b = lambda x, y: attention_r_q(x, y)
                r_q_b_res = tf.while_loop(cond=c, body=b, loop_vars=(i, r_q_b))
                r_q = r_q_b_res[-1].stack()
                # print 'r_q', r_q  # 哦没squeeze，[b, 1, ]

            with tf.variable_scope("answer_recurrent_network", reuse=tf.AUTO_REUSE):

                w_p_h = tf.get_variable(shape=[hidden_size * 2, hidden_size * 2], name="w_p_h_s")
                w_a_h = tf.get_variable(shape=[hidden_size*2, hidden_size*2], name="w_a_h_s")
                v4 = tf.get_variable(shape=[hidden_size*2, 1], name="v4")
                cell_h = tf.nn.rnn_cell.GRUCell(hidden_size * 2)

                def pointers(i, p_1_b, p_2_b, a_1_b, a_2_b):
                    p_i = v_p[i]
                    len_p_i = self.inputs_actual_length_concat_p[i]
                    p_i_t = tf.slice(p_i, begin=[0, 0], size=[len_p_i, hidden_size * 2])

                    # t就是取1,2,开头和结尾，见论文损失那里的下标
                    # start, t=1
                    h_a_1 = r_q[i]  # 初始状态
                    sum_1 = tf.matmul(p_i_t, w_p_h) + tf.matmul(h_a_1, w_a_h)
                    s_1 = tf.matmul(tf.tanh(sum_1), v4)  # 列向量，passasge长N
                    a_1 = tf.nn.softmax(s_1)
                    a_1 = tf.reshape(a_1, [-1, 1])
                    a_1_b.write(i, tf.transpose(a_1))  # 还是转行向量
                    c_1 = tf.transpose(tf.matmul(p_i_t, a_1))  # 行向量
                    c_1 = tf.reshape(c_1, [1, hidden_size * 2])  # 必须这样固定
                    h_a_1 = tf.reshape(h_a_1, [1, hidden_size * 2])
                    # print 'c_1', c_1  # (1,?),同样应150
                    # print 'h_a_1', h_a_1
                    h_a_2, state = cell_h(inputs=c_1, state=h_a_1)
                    p_1 = tf.argmax(a_1)
                    p_1_b.write(i, p_1)

                    # end,t=2
                    sum_2 = tf.matmul(p_i_t, w_p_h) + tf.matmul(h_a_2, w_a_h)
                    s_2 = tf.matmul(tf.tanh(sum_2), v4)  # 列向量，passasge长N
                    a_2 = tf.nn.softmax(s_2)
                    a_2 = tf.reshape(a_2, [-1, 1])
                    a_2_b.write(i, tf.transpose(a_2))
                    p_2 = tf.argmax(a_2)
                    p_2_b.write(i, p_2)

                    i = tf.add(i, 1)

                    return i, p_1_b, p_2_b, a_1_b, a_2_b

                p_1_b = tf.TensorArray(dtype=tf.int32, size=batch_size)
                p_2_b = tf.TensorArray(dtype=tf.int32, size=batch_size)
                a_1_b = tf.TensorArray(dtype=tf.float32, size=batch_size)
                a_2_b = tf.TensorArray(dtype=tf.float32, size=batch_size)
                c = lambda x, y, z, m, n: tf.less(x, batch_size)  # batch的循环
                b = lambda x, y, z, m, n: pointers(x, y, z, m, n)
                b_res = tf.while_loop(cond=c, body=b, loop_vars=(i, p_1_b, p_2_b, a_1_b, a_2_b))
                p_1 = b_res[1].stack()
                p_2 = b_res[2].stack()
                # print 'p_1', p_1
                # print 'p_2', p_2
                a_1 = b_res[3].stack()
                a_2 = b_res[4].stack()
                # print 'a_1', a_1
                # print 'a_2', a_2
                # self.p = [tf.reshape(p_1, [1, -1]), tf.reshape(p_2, [1, -1])]
                self.p1 = tf.reshape(p_1, [1, -1])
                self.p2 = tf.reshape(p_2, [1, -1])
                a = [tf.reshape(a_1, [1, -1]), tf.reshape(a_2, [1, -1])]
                # print p, a

        with tf.variable_scope("passage_ranking", reuse=tf.AUTO_REUSE):
            w_v_q = tf.get_variable(name='w_v_q', shape=[hidden_size * 2, hidden_size * 2])
            w_v_p = tf.get_variable(name='w_v_p', shape=[hidden_size * 2, hidden_size * 2])
            v3 = tf.get_variable(name='v3', shape=[hidden_size * 2, 1])
            v_g = tf.get_variable(name='v_g', shape=[hidden_size * 2, 1])
            w_g_2 = tf.get_variable(name='w_g_2', shape=[hidden_size * 2, hidden_size * 2])

            def attention_r_p_one_passage(j, start, end, v_p_i, r_q_i, p_w_num_i, r_p_i):
                v_p_i_j = v_p_i[start:end, :]
                # print 'v_p_i_j', v_p_i_j
                sum_p = tf.matmul(v_p_i_j, w_v_p) + tf.matmul(r_q_i, w_v_q)
                # print 'sum_p', sum_p  # [p_w_n, hidden*2]
                s = tf.matmul(tf.tanh(sum_p), v3)
                # print 's', s  # [p_w_n, 1]
                a = tf.nn.softmax(s)
                # print 'a', a
                r_p_i_j = tf.transpose(tf.matmul(tf.transpose(v_p_i_j), a))  # 还是转成行向量
                # print 'r_p_i_j', r_p_i_j  # [1, hidden*2]
                r_p_i.write(j, r_p_i_j)
                start = p_w_num_i[j]
                j = tf.add(j, 1)
                end = p_w_num_i[j]
                return j, start, end, v_p_i, r_q_i, p_w_num_i, r_p_i

            def attention_r_p(i, r_p_b):
                v_p_i = v_p[i]  # 主要是这里了，要分开成不同文章搞
                # print 'v_p_i', v_p_i, v_p_i[:self.passage_word_numbers[i][0],:]
                r_q_i = r_q[i]
                p_num_i = self.passage_numbers[i]
                r_p_i = tf.TensorArray(dtype=tf.float32, size=p_num_i)  # 这个竟然可以！！！
                # print r_p_i
                j = tf.constant(0)
                start = tf.constant(0)
                p_w_num_i = self.passage_word_numbers[i]
                end = p_w_num_i[0]
                c = lambda x, y, z, m, n, p, q: tf.less(x, p_num_i)
                b = lambda x, y, z, m, n, p, q: attention_r_p_one_passage(x, y, z, m, n, p, q)
                res = tf.while_loop(cond=c, body=b, loop_vars=(j, start, end, v_p_i, r_q_i, p_w_num_i, r_p_i))
                r_p_i = tf.squeeze(res[-1].stack(), axis=1)
                # print 'r_p_i', r_p_i
                r_p_b.write(i, r_p_i)
                i = tf.add(i, 1)
                return i, r_p_b

            r_p_b = tf.TensorArray(dtype=tf.float32, size=batch_size)
            c = lambda x, y: tf.less(x, batch_size)  # batch的循环
            b = lambda x, y: attention_r_p(x, y)
            r_p_b_res = tf.while_loop(cond=c, body=b, loop_vars=(i, r_p_b))
            r_p = r_p_b_res[-1].stack()
            # print 'r_p', r_p

            def compute_g_b_one_passage(j, r_q_i, r_p_i, g_i):
                r_p_i_j = tf.reshape(r_p_i[j], shape=[1, -1])
                # print r_q_i, r_p_i_j
                r_p_q = tf.concat([r_q_i, r_p_i_j], axis=1)
                mul_g = tf.matmul(r_p_q, w_g_2)
                g_j = tf.matmul(tf.tanh(mul_g), v_g)  # 这个是数
                # print 'g_j', g_j
                g_i.write(j, g_j)
                j = tf.add(j, 1)
                return j, r_q_i, r_p_i, g_i

            def compute_g_b(i, g_b):
                r_q_i = r_q[i]
                r_p_i = r_p[i]
                p_num_i = self.passage_numbers[i]
                j = tf.constant(0)
                g_i = tf.TensorArray(dtype=tf.float32, size=p_num_i)
                c = lambda x, y, z, m: tf.less(x, p_num_i)
                b = lambda x, y, z, m: compute_g_b_one_passage(x, y, z, m)
                res = tf.while_loop(cond=c, body=b, loop_vars=(j, r_q_i, r_p_i, g_i))
                g_i = tf.squeeze(res[-1].stack(), axis=1)  # 向量
                # print 'g_i', g_i
                # 要归一化一下，再加到g_b里
                g_i = tf.nn.softmax(g_i)
                g_b.write(i, g_i)
                i = tf.add(i, 1)
                return i, g_b

            # 还得按batch
            g_b = tf.TensorArray(dtype=tf.float32, size=batch_size)
            c = lambda x, y: tf.less(x, batch_size)  # batch的循环
            b = lambda x, y: compute_g_b(x, y)
            g_b_res = tf.while_loop(cond=c, body=b, loop_vars=(i, g_b))
            g = tf.squeeze(g_b_res[-1].stack(), axis=2)
            # print 'g', g

        if mode == 'train':
            with tf.variable_scope("loss"):
                # 不对，train中要的不是p，是a
                # 通过两个位置先搞个y出来
                # 长度此时还未定，似乎要用lambda？？

                def write_y(j, pos, y):
                    if j != pos:
                        y.write(j, 0)
                    else:
                        y.write(j, 1)
                    return j, pos, y

                def to_one_hot(i, y1_b, y2_b):
                    len_p_i = self.inputs_actual_length_concat_p[i]
                    start = self.start_position[i]
                    end = self.end_position[i]
                    y1 = tf.TensorArray(dtype=tf.float32, size=len_p_i)
                    y2 = tf.TensorArray(dtype=tf.float32, size=len_p_i)
                    c = lambda x, y, z: tf.less(x, len_p_i)  # batch的循环
                    b = lambda x, y, z: write_y(x, y, z)
                    j = tf.constant(0)  # batch号
                    y1_res = tf.while_loop(cond=c, body=b, loop_vars=(j, start, y1))
                    j = tf.constant(0)  # batch号
                    y2_res = tf.while_loop(cond=c, body=b, loop_vars=(j, end, y2))
                    y1_i = y1_res[-1].stack()
                    y2_i = y2_res[-1].stack()
                    y1_b.write(i, y1_i)
                    y2_b.write(i, y2_i)
                    i = tf.add(i, 1)
                    return i, y1_b, y2_b

                y1_b = tf.TensorArray(dtype=tf.float32, size=batch_size)  # 存放batch中每条的结果
                y2_b = tf.TensorArray(dtype=tf.float32, size=batch_size)  # 存放batch中每条的结果
                c = lambda x, y, z: tf.less(x, batch_size)  # batch的循环
                b = lambda x, y, z: to_one_hot(x, y, z)
                i = tf.constant(0)  # batch号
                res = tf.while_loop(cond=c, body=b, loop_vars=(i, y1_b, y2_b))  # 这个会先不循环，而后面用for则会循环
                y1 = res[-2].stack()
                y2 = res[-1].stack()
                y = [tf.reshape(y1, [1, -1]), tf.reshape(y2, [1, -1])]
                # print 'y', y

                self.loss = 0.0
                for t in range(2):
                    self.loss += tf.reduce_sum(y[t] * tf.log(a[t]) + (1-y[t]) * (1-tf.log(a[t])), 1)
                # print self.loss

                # 这里loss还没加上passage_ranking的

            # train_op什么的，看写哪，写这还是外面
            # with tf.variable_scope("train_op"):
            # # define train op
            #     tvars = tf.trainable_variables()
            #     print tvars
            #     grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), grad_clip)
            #     optimizer = tf.train.AdamOptimizer(initial_learning_rate)
            #     # grads_and_vars = optimizer.compute_gradients(self.loss)
            #     # self.train_op = optimizer.apply_gradients(grads_and_vars)
            #     self.train_op = optimizer.apply_gradients(zip(grads, tvars))
            #     print self.train_op
            # 为啥报错。。。



# char_emb就用可训练的加look_up；其实word_emb也可是不可训练的加loop_up
# # 可以设多少步存一下模型
# model = ExtractionModel(batch_size=128, hidden_size=75, embedding_dim=300, dropout_rate=0.1, grad_clip=5,
#                         initial_learning_rate=1e-3)
