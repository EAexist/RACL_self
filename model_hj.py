from utils import *
import numpy as np
from math import sqrt
from evaluation import get_metric
import os
import time
import logging
import tensorflow as tf
from tensorflow.contrib import layers


class MODEL(object):

    def __init__(self, opt, word_embedding, domain_embedding, word_dict):
        with tf.name_scope('parameters'):
            self.opt = opt
            self.w2v = word_embedding
            self.w2v_domain = domain_embedding
            self.word_id_mapping = word_dict
            self.Winit = tf.random_uniform_initializer(minval=-0.01, maxval=0.01, seed=0.05)

            info = ''
            for arg in vars(opt):
                info += ('>>> {0}: {1}\n'.format(arg, getattr(opt, arg)))

            if not os.path.exists(r'./log/{}'.format(self.opt.task)):
                os.makedirs(r'./log/{}'.format(self.opt.task))
            filename = r'./log/{}/{}.txt'.format(self.opt.task, time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()))
            self.logger = logging.getLogger(filename)
            self.logger.setLevel(logging.DEBUG)

            sh = logging.StreamHandler()
            th = logging.FileHandler(filename, 'a')

            self.logger.addHandler(sh)
            self.logger.addHandler(th)

            self.logger.info('{:-^80}'.format('Parameters'))
            self.logger.info(info + '\n')

        with tf.name_scope('embeddings'):
            self.word_embedding = tf.Variable(self.w2v, dtype=tf.float32, name='word_embedding', trainable=False)
            self.domain_embedding = tf.Variable(self.w2v_domain, dtype=tf.float32, name='domain_embedding', trainable=False)

        with tf.name_scope('inputs'):
            self.x = tf.placeholder(tf.int32, [None, self.opt.max_sentence_len], name='x')
            self.aspect_y = tf.placeholder(tf.int32, [None, self.opt.max_sentence_len, self.opt.class_num], name='aspect_y')
            self.opinion_y = tf.placeholder(tf.int32, [None, self.opt.max_sentence_len, self.opt.class_num], name='opinion_y')
            self.sentiment_y = tf.placeholder(tf.int32, [None, self.opt.max_sentence_len, self.opt.class_num], name='sentiment_y')
            self.word_mask = tf.placeholder(tf.float32, [None, self.opt.max_sentence_len], name='word_mask')
            self.senti_mask = tf.placeholder(tf.float32, [None, self.opt.max_sentence_len], name='senti_mask')
            self.position = tf.placeholder(tf.float32, [None, self.opt.max_sentence_len, self.opt.max_sentence_len], name='position_att')
            self.keep_prob1 = tf.placeholder(tf.float32)
            self.keep_prob2 = tf.placeholder(tf.float32)
            self.is_training = tf.placeholder(tf.bool)
            self.drop_block1 = DropBlock2D(keep_prob=self.keep_prob2, block_size=3)
            self.drop_block2 = DropBlock2D(keep_prob=self.keep_prob2, block_size=3)
            self.drop_block3 = DropBlock2D(keep_prob=self.keep_prob2, block_size=3)


    def RACL(self, inputs, position_att):
        batch_size = tf.shape(inputs)[0]
        inputs = tf.nn.dropout(inputs, keep_prob=self.keep_prob1)

        # Shared Feature
        inputs = tf.layers.conv1d(inputs, self.opt.emb_dim, 1, padding='SAME', activation=tf.nn.relu, name='inputs')
        inputs = tf.nn.dropout(inputs, keep_prob=self.keep_prob1)

        mask256 = tf.tile(tf.expand_dims(self.word_mask, -1), [1, 1, self.opt.filter_num])
        mask70 = tf.tile(tf.expand_dims(self.word_mask, 1), [1, self.opt.max_sentence_len, 1])

        # Private Feature
        aspect_input, opinion_input, context_input = list(), list(), list()
        aspect_prob_list, opinion_prob_list, senti_prob_list = list(), list(), list()
        aspect_input.append(inputs)
        opinion_input.append(inputs)
        context_input.append(inputs)

        # We found that the SC task is more difficult than the AE and OE tasks.
        # Hence, we augment it with a memory-like mechanism by updating the aspect query with the retrieved contexts.
        # Refer to https://www.aclweb.org/anthology/D16-1021/ for for more details about the memory network.
        query = list()
        query.append(inputs)

        ## RACL num_layer = hop
        for hop in range(self.opt.hop_num):
            with tf.variable_scope('layers_{}'.format(hop)):
                # AE & OE Convolution
                aspect_conv = tf.layers.conv1d(aspect_input[-1], self.opt.filter_num, self.opt.kernel_size, padding='SAME', activation=tf.nn.relu, name='aspect_conv')
                opinion_conv = tf.layers.conv1d(opinion_input[-1], self.opt.filter_num, self.opt.kernel_size, padding='SAME', activation=tf.nn.relu, name='opinion_conv')

                # Relation R1
                aspect_see_opinion = tf.matmul(tf.nn.l2_normalize(aspect_conv, -1), tf.nn.l2_normalize(opinion_conv, -1), adjoint_b=True)
                aspect_att_opinion = softmask_2d(aspect_see_opinion, self.word_mask)
                aspect_inter = tf.concat([aspect_conv, tf.matmul(aspect_att_opinion, opinion_conv)], -1)

                opinion_see_aspect = tf.matmul(tf.nn.l2_normalize(opinion_conv, -1), tf.nn.l2_normalize(aspect_conv, -1), adjoint_b=True)
                opinion_att_aspect = softmask_2d(opinion_see_aspect, self.word_mask)
                opinion_inter = tf.concat([opinion_conv, tf.matmul(opinion_att_aspect, aspect_conv)], -1)

                # AE & OE Prediction
                aspect_p = layers.fully_connected(aspect_inter, self.opt.class_num, activation_fn=None, weights_initializer=self.Winit, biases_initializer=self.Winit, scope='aspect_p')
                opinion_p = layers.fully_connected(opinion_inter, self.opt.class_num, activation_fn=None, weights_initializer=self.Winit, biases_initializer=self.Winit, scope='opinion_p')

                # OE Confidence
                # A slight difference from the original paper.
                # For propagating R3, we calculate the confidence of each candidate opinion word.
                # Only when a word satisfies the condition Prob[B,I] > Prob[O] in OE, it can be propagated to SC.
                confidence = tf.maximum(0., 1 - 2. * tf.nn.softmax(opinion_p, -1)[:, :, 0])
                opinion_propagate = tf.tile(tf.expand_dims(confidence, 1), [1, self.opt.max_sentence_len, 1]) * mask70 * position_att

                # SC Convolution
                context_conv = tf.layers.conv1d(context_input[-1], self.opt.emb_dim, self.opt.kernel_size, padding='SAME', activation=tf.nn.relu, name='context_conv')

                # SC Aspect-Context Attention
                word_see_context = tf.matmul((query[-1]), tf.nn.l2_normalize(context_conv, -1), adjoint_b=True)  * position_att
                word_att_context = softmask_2d(word_see_context, self.word_mask, scale=True)

                # Relation R2 & R3
                word_att_context += aspect_att_opinion + opinion_propagate
                context_inter = (query[-1] + tf.matmul(word_att_context, context_conv)) # query + value
                query.append(context_inter) # update query

                # SC Prediction
                senti_p = layers.fully_connected(context_inter, self.opt.class_num, activation_fn=None, weights_initializer=self.Winit, biases_initializer=self.Winit, scope='senti_p')

                # Stacking
                aspect_prob_list.append(tf.expand_dims(aspect_p, -1))
                opinion_prob_list.append(tf.expand_dims(opinion_p, -1))
                senti_prob_list.append(tf.expand_dims(senti_p, -1))

                # We use DropBlock to enhance the learning of the private features for AE & OE & SC.
                # Refer to http://papers.nips.cc/paper/8271-dropblock-a-regularization-method-for-convolutional-networks for more details.
                aspect_inter = tf.squeeze(self.drop_block1(inputs=tf.expand_dims(aspect_inter, -1), training=self.is_training), -1)
                opinion_inter = tf.squeeze(self.drop_block2(inputs=tf.expand_dims(opinion_inter, -1), training=self.is_training), -1)
                context_conv = tf.squeeze(self.drop_block3(inputs=tf.expand_dims(context_conv, -1), training=self.is_training), -1)

                aspect_input.append(aspect_inter)
                opinion_input.append(opinion_inter)
                context_input.append(context_conv)

        # Multi-layer Short-cut
        aspect_prob = tf.reduce_mean(tf.concat(aspect_prob_list, -1), -1)
        opinion_prob = tf.reduce_mean(tf.concat(opinion_prob_list, -1), -1)
        sentiment_prob = tf.reduce_mean(tf.concat(senti_prob_list, -1), -1)

        return aspect_prob, opinion_prob, sentiment_prob

    def get_one_hot(self, targets, nb_classes):
        res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
        return res.reshape(list(targets.shape)+[nb_classes])

############################################################    
    def get_onehot_type(self, preds, type):
          # target shape: (1705, 80, 3)
          new_preds = []
          if (type == 'aspect') | (type == 'opinion'):
            for i in range(len(preds)):
              new_preds_i = []
              zero_index = preds[i].index([0,0,0])
              temp_preds = preds[i][:zero_index]
              arg_preds = np.argmax(temp_preds, axis=-1)
              new_preds_i.extend(self.get_one_hot(arg_preds, 3).tolist())
              new_preds_i.extend(preds[i][zero_index:])
              new_preds.append(new_preds_i)
            return np.array(new_preds)
          elif type == 'sentiment':
            for i in range(len(preds)):
              new_preds_i = []
              for j in range(len(preds[i])):
                if preds[i][j] == [0,0,0]:
                  new_preds_i.append([0,0,0])
                else:
                  arg_pred = np.argmax(preds[i][j], axis=-1)
                  new_preds_i.append(self.get_one_hot(arg_pred, 3).tolist())
              new_preds.append(new_preds_i)
            return np.array(new_preds)
############################################################                    

    def run(self):
        # a = tf.ones
        # print(a.numpy())
        # print(get_one_hot(a,3))

        batch_size = tf.shape(self.x)[0]
        inputs_word = tf.nn.embedding_lookup(self.word_embedding, self.x)
        inputs_domain = tf.nn.embedding_lookup(self.domain_embedding, self.x)
        inputs = tf.concat([inputs_word, inputs_domain], -1)

        aspect_prob, opinion_prob, sentiment_prob = self.RACL(inputs, self.position)
        aspect_value = tf.nn.softmax(aspect_prob, -1)
        opinion_value = tf.nn.softmax(opinion_prob, -1)
        senti_value = tf.nn.softmax(sentiment_prob, -1)

        # AE & OE Regulation Loss
        reg_cost = tf.reduce_sum(tf.maximum(0., tf.reduce_sum(aspect_value[:,:,1:], -1) + tf.reduce_sum(opinion_value[:,:,1:], -1) - 1.)) / tf.reduce_sum(self.word_mask)

        # Mask AE & OE Probabilities
        word_mask = tf.tile(tf.expand_dims(self.word_mask, -1), [1, 1, self.opt.class_num])
        aspect_prob = tf.reshape(word_mask * aspect_prob, [-1, self.opt.class_num])
        aspect_label = tf.reshape(self.aspect_y, [-1, self.opt.class_num])
        opinion_prob = tf.reshape(word_mask * opinion_prob, [-1, self.opt.class_num])
        opinion_label = tf.reshape(self.opinion_y, [-1, self.opt.class_num])

        # Relation R4 (Only in Training)
        # In training/validation, the sentiment masks are set to 1.0 only for the aspect terms.
        # In testing, the sentiment masks are set to 1.0 for all words (except padding ones).
        senti_mask = tf.tile(tf.expand_dims(self.senti_mask, -1), [1, 1, self.opt.class_num])

        # Mask SC Probabilities
        sentiment_prob = tf.reshape(tf.cast(senti_mask, tf.float32) * sentiment_prob, [-1, self.opt.class_num])
        sentiment_label = tf.reshape(self.sentiment_y, [-1, self.opt.class_num])

        with tf.name_scope('loss'):
            tv = tf.trainable_variables()
            total_para = count_parameter()
            self.logger.info('>>> total parameter: {}'.format(total_para))

            aspect_cost = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(logits=aspect_prob, labels=tf.cast(aspect_label, tf.float32)))
            opinion_cost = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(logits=opinion_prob, labels=tf.cast(opinion_label, tf.float32)))
            sentiment_cost = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(logits=sentiment_prob, labels=tf.cast(sentiment_label, tf.float32)))

            cost = aspect_cost + opinion_cost + sentiment_cost + self.opt.reg_scale * reg_cost

        with tf.name_scope('train'):
            global_step = tf.Variable(0, name="tr_global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(learning_rate=self.opt.learning_rate).minimize(cost, global_step=global_step)

        with tf.name_scope('predict'):
            true_ay = tf.reshape(aspect_label, [batch_size, self.opt.max_sentence_len, -1])
            pred_ay = tf.reshape(aspect_prob, [batch_size, self.opt.max_sentence_len, -1])

            true_oy = tf.reshape(opinion_label, [batch_size, self.opt.max_sentence_len, -1])
            pred_oy = tf.reshape(opinion_prob, [batch_size, self.opt.max_sentence_len, -1])

            true_sy = tf.reshape(sentiment_label, [batch_size, self.opt.max_sentence_len, -1])
            pred_sy = tf.reshape(sentiment_prob, [batch_size, self.opt.max_sentence_len, -1])

        saver = tf.train.Saver(max_to_keep=120)
        # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)

        ## load data tf.Session()밖에서 미리 진행한다.
        #train_sets = read_data(self.opt.train_path, self.word_id_mapping, self.opt.max_sentence_len)
        train_unlabeled_sets, train_labeled_sets = read_data(self.opt.train_path, self.word_id_mapping,
                                                             self.opt.max_sentence_len, unlabeled_ratio=self.opt.unlabeled_ratio)
        dev_sets = read_data(self.opt.dev_path, self.word_id_mapping, self.opt.max_sentence_len)
        test_sets = read_data(self.opt.test_path, self.word_id_mapping, self.opt.max_sentence_len, is_testing=True)


        ## meta epoch 0 : 먼저 labeled_sets를 이용해 학습해 unlabeled_sets에 pseudo label 입력하는 과정

        with tf.Session() as sess:
            if self.opt.load == 0:
                init = tf.global_variables_initializer()
                sess.run(init)
            else:
                ckpt = tf.train.get_checkpoint_state('checkpoint/{}-meta{}'.format(self.opt.task, 0))
                saver.restore(sess, ckpt.model_checkpoint_path)

            aspect_f1_list = []
            opinion_f1_list = []
            sentiment_acc_list = []
            sentiment_f1_list = []
            ABSA_f1_list = []
            dev_metric_list = []
            dev_loss_list = []

            pr_a_preds_list, pr_a_labels_list = [], []
            pr_o_preds_list, pr_o_labels_list = [], []
            pr_s_preds_list, pr_s_labels_list = [], []
            pr_final_mask_list = []
            pr_aspect_f1_list = []
            pr_opinion_f1_list = []
            pr_sentiment_acc_list = []
            pr_sentiment_f1_list = []
            pr_ABSA_f1_list = []       

            for i in range(self.opt.n_iter):
                'Train'
                tr_loss = 0.
                tr_aloss = 0.
                tr_oloss = 0.
                tr_sloss = 0.
                tr_rloss = 0.
                if self.opt.load == 0:
                    epoch_start = time.time()
                    ## labeled_sets 넣어서 학습하기!!
                    for train, num in self.get_batch_data(train_labeled_sets, self.opt.batch_size, self.opt.kp1, self.opt.kp2, True, True):
                        tr_eloss, tr_aeloss, tr_oeloss, tr_seloss, tr_reloss, _, step = sess.run(
                        [cost, aspect_cost, opinion_cost, sentiment_cost, reg_cost, optimizer, global_step], feed_dict=train)
                        tr_loss += tr_eloss * num
                        tr_aloss += tr_aeloss * num
                        tr_oloss += tr_oeloss * num
                        tr_sloss += tr_seloss * num
                        tr_rloss += tr_reloss * num
                    # if i >= self.opt.warmup_iter:
                    #     saver.save(sess, 'checkpoint/{}/RACL.ckpt'.format(self.opt.task), global_step=i)
                    epoch_end = time.time()
                    epoch_time = 'Epoch Time: {:.0f}m {:.0f}s'.format((epoch_end - epoch_start) // 60, (epoch_end - epoch_start) % 60)

                'Test'
                a_preds, a_labels = [], []
                o_preds, o_labels = [], []
                s_preds, s_labels = [], []
                final_mask = []
                for test, _ in self.get_batch_data(test_sets, 200, 1.0, 1.0):
                    _step, t_ay, p_ay, t_oy, p_oy, t_sy, p_sy, e_mask = sess.run(
                        [global_step, true_ay, pred_ay, true_oy, pred_oy, true_sy, pred_sy, self.word_mask], feed_dict=test)
                    a_preds.extend(p_ay)
                    a_labels.extend(t_ay)
                    o_preds.extend(p_oy)
                    o_labels.extend(t_oy)
                    s_preds.extend(p_sy)
                    s_labels.extend(t_sy)
                    final_mask.extend(e_mask)

                aspect_f1, opinion_f1, sentiment_acc, sentiment_f1, ABSA_f1 \
                    = get_metric(a_labels, a_preds, o_labels, o_preds, s_labels, s_preds, final_mask, 1)

                aspect_f1_list.append(aspect_f1)
                opinion_f1_list.append(opinion_f1)
                sentiment_acc_list.append(sentiment_acc)
                sentiment_f1_list.append(sentiment_f1)
                ABSA_f1_list.append(ABSA_f1)

                ## unlabeled 데이터 넣어서 predict
                'Predict Unlabeled'
                pr_a_preds, pr_a_labels = [], []
                pr_o_preds, pr_o_labels = [], []
                pr_s_preds, pr_s_labels = [], []
                pr_final_mask = []
                for unlabeled, _ in self.get_batch_data(train_unlabeled_sets, 200, 1.0, 1.0):
                    _step, t_ay, p_ay, t_oy, p_oy, t_sy, p_sy, e_mask = sess.run(
                        [global_step, true_ay, pred_ay, true_oy, pred_oy, true_sy, pred_sy, self.word_mask], feed_dict=unlabeled)
                    _step, t_ay, p_ay, t_oy, p_oy, t_sy, p_sy, e_mask = sess.run(
                        [global_step, true_ay, pred_ay, true_oy, pred_oy, true_sy, pred_sy, self.word_mask], feed_dict=unlabeled)                        
                    pr_a_preds.extend(p_ay)
                    pr_a_labels.extend(t_ay)
                    pr_o_preds.extend(p_oy)
                    pr_o_labels.extend(t_oy)
                    pr_s_preds.extend(p_sy)
                    pr_s_labels.extend(t_sy)
                    pr_final_mask.extend(e_mask)

                ## 각 epoch마다의 값들 저장
                pr_a_preds_list.append(pr_a_preds)
                pr_a_labels_list.append(pr_a_labels)
                pr_o_preds_list.append(pr_o_preds)
                pr_o_labels_list.append(pr_o_labels)
                pr_s_preds_list.append(pr_s_preds)
                pr_s_labels_list.append(pr_s_labels)
                pr_final_mask_list.append(pr_final_mask)

                pr_aspect_f1, pr_opinion_f1, pr_sentiment_acc, pr_sentiment_f1, pr_ABSA_f1 \
                    = get_metric(pr_a_labels, pr_a_preds, pr_o_labels, pr_o_preds, pr_s_labels, pr_s_preds, pr_final_mask, 1)

                pr_aspect_f1_list.append(pr_aspect_f1)
                pr_opinion_f1_list.append(pr_opinion_f1)
                pr_sentiment_acc_list.append(pr_sentiment_acc)
                pr_sentiment_f1_list.append(pr_sentiment_f1)
                pr_ABSA_f1_list.append(pr_ABSA_f1)

                'Dev'
                dev_loss = 0.
                dev_aloss = 0.
                dev_oloss = 0.
                dev_sloss = 0.
                dev_rloss = 0.
                dev_a_preds, dev_a_labels = [], []
                dev_o_preds, dev_o_labels = [], []
                dev_s_preds, dev_s_labels = [], []
                dev_final_mask = []
                for dev, num in self.get_batch_data(dev_sets, 200, 1.0, 1.0):
                    dev_eloss, dev_aeloss, dev_oeloss, dev_seloss, dev_reloss, _step, dev_t_ay, dev_p_ay, dev_t_oy, dev_p_oy, dev_t_sy, dev_p_sy, dev_e_mask = \
                        sess.run([cost, aspect_cost, opinion_cost, sentiment_cost, reg_cost, global_step, true_ay, pred_ay, true_oy, pred_oy, true_sy, pred_sy, self.word_mask],
                                 feed_dict=dev)
                    dev_a_preds.extend(dev_p_ay)
                    dev_a_labels.extend(dev_t_ay)
                    dev_o_preds.extend(dev_p_oy)
                    dev_o_labels.extend(dev_t_oy)
                    dev_s_preds.extend(dev_p_sy)
                    dev_s_labels.extend(dev_t_sy)
                    dev_final_mask.extend(dev_e_mask)
                    dev_loss += dev_eloss * num
                    dev_aloss += dev_aeloss * num
                    dev_oloss += dev_oeloss * num
                    dev_sloss += dev_seloss * num
                    dev_rloss += dev_reloss * num

                dev_aspect_f1, dev_opinion_f1, dev_sentiment_acc, dev_sentiment_f1, dev_ABSA_f1 \
                    = get_metric(dev_a_labels, dev_a_preds, dev_o_labels, dev_o_preds, dev_s_labels, dev_s_preds, dev_final_mask, 1)

                if i < self.opt.warmup_iter:
                    dev_metric_list.append(0.) ## metric은 클수록 좋고
                    dev_loss_list.append(1000.) ## loss는 작을수록 좋으니까
                else:
                    dev_metric_list.append(dev_ABSA_f1)
                    dev_loss_list.append(dev_loss)

                if self.opt.load == 0:
                    self.logger.info('\nMETA EPOCH 0 | {:-^80}'.format('Iter' + str(i)))

                    # self.logger.info('Train: final loss={:.6f}, aspect loss={:.6f}, opinion loss={:.6f}, sentiment loss={:.6f}, reg loss={:.6f}, step={}'.
                    #     format(tr_loss, tr_aloss, tr_oloss, tr_sloss, tr_rloss, step))
                    # self.logger.info('Dev:   final loss={:.6f}, aspect loss={:.6f}, opinion loss={:.6f}, sentiment loss={:.6f}, reg loss={:.6f}, step={}'.
                    #     format(dev_loss, dev_aloss, dev_oloss, dev_sloss, dev_rloss, step))

                    # self.logger.info('Dev:   aspect f1={:.4f}, opinion f1={:.4f}, sentiment acc=={:.4f}, sentiment f1=={:.4f}, ABSA f1=={:.4f},'
                    #     .format(dev_aspect_f1, dev_opinion_f1, dev_sentiment_acc, dev_sentiment_f1, dev_ABSA_f1))
                    # self.logger.info('Test:  aspect f1={:.4f}, opinion f1={:.4f}, sentiment acc=={:.4f}, sentiment f1=={:.4f}, ABSA f1=={:.4f},'
                    #     .format(aspect_f1, opinion_f1, sentiment_acc, sentiment_f1, ABSA_f1))
                    # ## PREDICT UNLABELED 추가
                    # self.logger.info('Predict unlabeled:  aspect f1={:.4f}, opinion f1={:.4f}, sentiment acc=={:.4f}, sentiment f1=={:.4f}, ABSA f1=={:.4f},'
                    #     .format(pr_aspect_f1, pr_opinion_f1, pr_sentiment_acc, pr_sentiment_f1, pr_ABSA_f1))

                    # self.logger.info('Current Max Metrics Index : {} Current Min Loss Index : {} {}'
                    #       .format(dev_metric_list.index(max(dev_metric_list)), dev_loss_list.index(min(dev_loss_list)), epoch_time))

                if self.opt.load == 1:
                    break

            self.logger.info('\nMETA EPOCH 0 | {:-^80}'.format('Mission Complete'))

            max_dev_index = dev_metric_list.index(max(dev_metric_list))
            self.logger.info('Dev Max Metrics Index: {}'.format(max_dev_index))
            self.logger.info('TEST | aspect f1={:.4f}, opinion f1={:.4f}, sentiment acc={:.4f}, sentiment f1={:.4f}, ABSA f1={:.4f},'
                  .format(aspect_f1_list[max_dev_index], opinion_f1_list[max_dev_index],
                          sentiment_acc_list[max_dev_index],
                          sentiment_f1_list[max_dev_index], ABSA_f1_list[max_dev_index]))
            ## PREDICT UNLABELED log 추가
            self.logger.info('PREDICT UNLABELED | aspect f1={:.4f}, opinion f1={:.4f}, sentiment acc={:.4f}, sentiment f1={:.4f}, ABSA f1={:.4f},'
                  .format(pr_aspect_f1_list[max_dev_index], pr_opinion_f1_list[max_dev_index],
                          pr_sentiment_acc_list[max_dev_index],
                          pr_sentiment_f1_list[max_dev_index], pr_ABSA_f1_list[max_dev_index]))

            min_dev_index = dev_loss_list.index(min(dev_loss_list))
            self.logger.info('Dev Min Loss Index: {}'.format(min_dev_index))
            self.logger.info('TEST aspect f1={:.4f}, opinion f1={:.4f}, sentiment acc={:.4f}, sentiment f1={:.4f}, ABSA f1={:.4f},'
                  .format(aspect_f1_list[min_dev_index], opinion_f1_list[min_dev_index],
                          sentiment_acc_list[min_dev_index],
                          sentiment_f1_list[min_dev_index], ABSA_f1_list[min_dev_index]))
            ## PREDICT UNLABELED log 추가
            self.logger.info('PREDICT UNLABELED | aspect f1={:.4f}, opinion f1={:.4f}, sentiment acc={:.4f}, sentiment f1={:.4f}, ABSA f1={:.4f},'
                  .format(pr_aspect_f1_list[min_dev_index], pr_opinion_f1_list[min_dev_index],
                          pr_sentiment_acc_list[min_dev_index],
                          pr_sentiment_f1_list[min_dev_index], pr_ABSA_f1_list[min_dev_index]))

##############################################################
            pr_a_preds = pr_a_preds_list[min_dev_index]
            pr_o_preds = pr_o_preds_list[min_dev_index]
            pr_s_preds = pr_s_preds_list[min_dev_index]
            
            pr_a_preds = get_onehot_type(pr_a_preds, 'aspect')
            pr_o_preds = get_onehot_type(pr_o_preds, 'opinion')
            pr_s_preds = get_onehot_type(pr_s_preds, 'sentiment')
            # pr_a_preds = np.argmax(pr_a_preds, axis=-1)
            # pr_a_preds = self.get_one_hot(pr_a_preds, 3)

            # pr_o_preds = np.argmax(pr_o_preds, axis=-1)
            # pr_o_preds = self.get_one_hot(pr_o_preds, 3)

            # pr_s_preds = np.argmax(pr_s_preds, axis=-1)
            # pr_s_preds = self.get_one_hot(pr_s_preds, 4)
##############################################################    

            ## predicted_y를 넣어서 new_unlabeled_data 만들기
            ## 기존 애들 불러온 다음.. 새로 구성
            unlabeled_source_data =train_unlabeled_sets[0]
            unlabeled_aspect_y =train_unlabeled_sets[1]
            unlabeled_opinion_y =train_unlabeled_sets[2]
            unlabeled_sentiment_y =train_unlabeled_sets[3]
            unlabeled_source_mask =train_unlabeled_sets[4]
            unlabeled_sentiment_mask =train_unlabeled_sets[5]
            unlabeled_position_m =train_unlabeled_sets[6]
    
            # labeled data
            labeled_source_data =train_labeled_sets[0]
            labeled_aspect_y =train_labeled_sets[1]
            labeled_opinion_y =train_labeled_sets[2]
            labeled_sentiment_y =train_labeled_sets[3]
            labeled_source_mask =train_labeled_sets[4]
            labeled_sentiment_mask =train_labeled_sets[5]
            labeled_position_m = train_labeled_sets[6]
            
            
            index_no_aspect = np.array(np.where(np.all(labeled_aspect_y == np.array([1,0,0]), axis=-1))).T ## aspect에서 0 라벨에 해당하는 단어 index 
            index_no_opinion = np.array(np.where(np.all(labeled_opinion_y == np.array([1,0,0]), axis=-1))).T ## opinion에서 0 라벨에 해당하는 단어 index 
            index_no_as_op = np.array([x for x in set(tuple(x) for x in index_no_aspect) & set(tuple(x) for x in index_no_opinion)]) ## 둘의 교집합

            chosen_rows_index = np.random.choice(labeled_source_data.shape[0], int(labeled_source_data.shape[0]*self.opt.shake_ratio), replace=False) ## 얼마나의 문장을 선택할 것인가? 선택받은 문장의 index

            for k in chosen_rows_index:
                try:
                    temp_index_no_as_op = index_no_as_op[np.where(index_no_as_op[:,0]==k)] ## 해당 row에 해당하는 index_no_as_op 불러오기
                    chosen_indexs = np.random.choice(temp_index_no_as_op.shape[0], 2, replace=False) ## temp_index_no_as_op(단어 중 ) 2개 골라낸다 2개의 temp_index_no_as_op 상에서의 index 찾아냄.
                    index1 = temp_index_no_as_op[chosen_indexs[0]] ## 실제 labeled_source_data에서의 index 값을 저장
                    index2 = temp_index_no_as_op[chosen_indexs[1]]
                    labeled_source_data[index1], labeled_source_data[index2] = labeled_source_data[index2], labeled_source_data[index1]  ## swap!!
                except:
                    continue
                    

            ## 먼저 pr_a_labels의 형태가 unlabeled_aspect_y와 동일한지 확인
            new_train_sets = (np.concatenate((labeled_source_data, unlabeled_source_data), axis=0),
                              np.concatenate((labeled_aspect_y, np.array(pr_a_preds)), axis=0),
                              np.concatenate((labeled_opinion_y, np.array(pr_o_preds)), axis=0),
                              np.concatenate((labeled_sentiment_y, np.array(pr_s_preds)), axis=0),
                              np.concatenate((labeled_source_mask, unlabeled_source_mask), axis=0),
                              np.concatenate((labeled_sentiment_mask, unlabeled_sentiment_mask), axis=0),
                              np.concatenate((labeled_position_m, unlabeled_position_m), axis=0))
            ## 위에서 predict의 결과물 중 final_mask가 있는데 이건 따로 추가를 안해줘도 되는걸까?

        ## meta epoch 1~n students 개수만큼 진행..!
        for j in range(self.opt.num_students):

            self.opt.load = 0 ## 이걸 해줘야 sess.run(init)이 작동되는거겠지.. 모델을 새롭게 시작한다는 차원에서
            with tf.Session() as sess:
                if self.opt.load == 0:
                    init = tf.global_variables_initializer()
                    sess.run(init)
                else:
                    ckpt = tf.train.get_checkpoint_state('checkpoint/{}-meta{}'.format(self.opt.task, j+1))
                    saver.restore(sess, ckpt.model_checkpoint_path)

                aspect_f1_list = []
                opinion_f1_list = []
                sentiment_acc_list = []
                sentiment_f1_list = []
                ABSA_f1_list = []
                dev_metric_list = []
                dev_loss_list = []

                pr_a_preds_list, pr_a_labels_list = [], []
                pr_o_preds_list, pr_o_labels_list = [], []
                pr_s_preds_list, pr_s_labels_list = [], []
                pr_final_mask_list = []
                pr_aspect_f1_list = []
                pr_opinion_f1_list = []
                pr_sentiment_acc_list = []
                pr_sentiment_f1_list = []
                pr_ABSA_f1_list = [] 
                
                for i in range(self.opt.n_iter):
                    'Train'
                    tr_loss = 0.
                    tr_aloss = 0.
                    tr_oloss = 0.
                    tr_sloss = 0.
                    tr_rloss = 0.
                    if self.opt.load == 0:
                        epoch_start = time.time()
                        ## labeled_sets 넣어서 학습하기!!
                        for train, num in self.get_batch_data(new_train_sets, self.opt.batch_size, self.opt.kp1, self.opt.kp2, True, True):
                            tr_eloss, tr_aeloss, tr_oeloss, tr_seloss, tr_reloss, _, step = sess.run(
                            [cost, aspect_cost, opinion_cost, sentiment_cost, reg_cost, optimizer, global_step], feed_dict=train)
                            tr_loss += tr_eloss * num
                            tr_aloss += tr_aeloss * num
                            tr_oloss += tr_oeloss * num
                            tr_sloss += tr_seloss * num
                            tr_rloss += tr_reloss * num
                        # if i >= self.opt.warmup_iter:
                        #     saver.save(sess, 'checkpoint/{}/RACL.ckpt'.format(self.opt.task), global_step=i)
                        epoch_end = time.time()
                        epoch_time = 'Epoch Time: {:.0f}m {:.0f}s'.format((epoch_end - epoch_start) // 60, (epoch_end - epoch_start) % 60)

                    'Test'
                    a_preds, a_labels = [], []
                    o_preds, o_labels = [], []
                    s_preds, s_labels = [], []
                    final_mask = []
                    for test, _ in self.get_batch_data(test_sets, 200, 1.0, 1.0):
                        _step, t_ay, p_ay, t_oy, p_oy, t_sy, p_sy, e_mask = sess.run(
                            [global_step, true_ay, pred_ay, true_oy, pred_oy, true_sy, pred_sy, self.word_mask], feed_dict=test)
                        a_preds.extend(p_ay)
                        a_labels.extend(t_ay)
                        o_preds.extend(p_oy)
                        o_labels.extend(t_oy)
                        s_preds.extend(p_sy)
                        s_labels.extend(t_sy)
                        final_mask.extend(e_mask)

                    aspect_f1, opinion_f1, sentiment_acc, sentiment_f1, ABSA_f1 \
                        = get_metric(a_labels, a_preds, o_labels, o_preds, s_labels, s_preds, final_mask, 1)

                    aspect_f1_list.append(aspect_f1)
                    opinion_f1_list.append(opinion_f1)
                    sentiment_acc_list.append(sentiment_acc)
                    sentiment_f1_list.append(sentiment_f1)
                    ABSA_f1_list.append(ABSA_f1)

                    ## unlabeled 데이터 넣어서 predict
                    'Predict Unlabeled'
                    pr_a_preds, pr_a_labels = [], []
                    pr_o_preds, pr_o_labels = [], []
                    pr_s_preds, pr_s_labels = [], []
                    pr_final_mask = []
                    for unlabeled, _ in self.get_batch_data(train_unlabeled_sets, 200, 1.0, 1.0):
                        _step, t_ay, p_ay, t_oy, p_oy, t_sy, p_sy, e_mask = sess.run(
                            [global_step, true_ay, pred_ay, true_oy, pred_oy, true_sy, pred_sy, self.word_mask], feed_dict=unlabeled)
                        pr_a_preds.extend(p_ay)
                        pr_a_labels.extend(t_ay)
                        pr_o_preds.extend(p_oy)
                        pr_o_labels.extend(t_oy)
                        pr_s_preds.extend(p_sy)
                        pr_s_labels.extend(t_sy)
                        pr_final_mask.extend(e_mask)

                    ## 각 epoch마다의 값들 저장
                    pr_a_preds_list.append(pr_a_preds)
                    pr_a_labels_list.append(pr_a_labels)
                    pr_o_preds_list.append(pr_o_preds)
                    pr_o_labels_list.append(pr_o_labels)
                    pr_s_preds_list.append(pr_s_preds)
                    pr_s_labels_list.append(pr_s_labels)
                    pr_final_mask_list.append(pr_final_mask)

                    pr_aspect_f1, pr_opinion_f1, pr_sentiment_acc, pr_sentiment_f1, pr_ABSA_f1 \
                        = get_metric(pr_a_labels, pr_a_preds, pr_o_labels, pr_o_preds, pr_s_labels, pr_s_preds, pr_final_mask, 1)

                    pr_aspect_f1_list.append(pr_aspect_f1)
                    pr_opinion_f1_list.append(pr_opinion_f1)
                    pr_sentiment_acc_list.append(pr_sentiment_acc)
                    pr_sentiment_f1_list.append(pr_sentiment_f1)
                    pr_ABSA_f1_list.append(pr_ABSA_f1)


                    'Dev'
                    dev_loss = 0.
                    dev_aloss = 0.
                    dev_oloss = 0.
                    dev_sloss = 0.
                    dev_rloss = 0.
                    dev_a_preds, dev_a_labels = [], []
                    dev_o_preds, dev_o_labels = [], []
                    dev_s_preds, dev_s_labels = [], []
                    dev_final_mask = []
                    for dev, num in self.get_batch_data(dev_sets, 200, 1.0, 1.0):
                        dev_eloss, dev_aeloss, dev_oeloss, dev_seloss, dev_reloss, _step, dev_t_ay, dev_p_ay, dev_t_oy, dev_p_oy, dev_t_sy, dev_p_sy, dev_e_mask = \
                            sess.run([cost, aspect_cost, opinion_cost, sentiment_cost, reg_cost, global_step, true_ay, pred_ay, true_oy, pred_oy, true_sy, pred_sy, self.word_mask],
                                     feed_dict=dev)
                        dev_a_preds.extend(dev_p_ay)
                        dev_a_labels.extend(dev_t_ay)
                        dev_o_preds.extend(dev_p_oy)
                        dev_o_labels.extend(dev_t_oy)
                        dev_s_preds.extend(dev_p_sy)
                        dev_s_labels.extend(dev_t_sy)
                        dev_final_mask.extend(dev_e_mask)
                        dev_loss += dev_eloss * num
                        dev_aloss += dev_aeloss * num
                        dev_oloss += dev_oeloss * num
                        dev_sloss += dev_seloss * num
                        dev_rloss += dev_reloss * num

                    dev_aspect_f1, dev_opinion_f1, dev_sentiment_acc, dev_sentiment_f1, dev_ABSA_f1 \
                        = get_metric(dev_a_labels, dev_a_preds, dev_o_labels, dev_o_preds, dev_s_labels, dev_s_preds, dev_final_mask, 1)

                    if i < self.opt.warmup_iter:
                        dev_metric_list.append(0.) ## metric은 클수록 좋고
                        dev_loss_list.append(1000.) ## loss는 작을수록 좋으니까
                    else:
                        dev_metric_list.append(dev_ABSA_f1)
                        dev_loss_list.append(dev_loss)

                    if self.opt.load == 0:
                        self.logger.info('\nMETA EPOCH {} | {:-^80}'.format(j+1, 'Iter' + str(i)))

                        self.logger.info('Train: final loss={:.6f}, aspect loss={:.6f}, opinion loss={:.6f}, sentiment loss={:.6f}, reg loss={:.6f}, step={}'.
                            format(tr_loss, tr_aloss, tr_oloss, tr_sloss, tr_rloss, step))
                        self.logger.info('Dev:   final loss={:.6f}, aspect loss={:.6f}, opinion loss={:.6f}, sentiment loss={:.6f}, reg loss={:.6f}, step={}'.
                            format(dev_loss, dev_aloss, dev_oloss, dev_sloss, dev_rloss, step))

                        self.logger.info('Dev:   aspect f1={:.4f}, opinion f1={:.4f}, sentiment acc=={:.4f}, sentiment f1=={:.4f}, ABSA f1=={:.4f},'
                            .format(dev_aspect_f1, dev_opinion_f1, dev_sentiment_acc, dev_sentiment_f1, dev_ABSA_f1))
                        self.logger.info('Test:  aspect f1={:.4f}, opinion f1={:.4f}, sentiment acc=={:.4f}, sentiment f1=={:.4f}, ABSA f1=={:.4f},'
                            .format(aspect_f1, opinion_f1, sentiment_acc, sentiment_f1, ABSA_f1))
                        ## PREDICT UNLABELED 추가
                        self.logger.info('Predict unlabeled:  aspect f1={:.4f}, opinion f1={:.4f}, sentiment acc=={:.4f}, sentiment f1=={:.4f}, ABSA f1=={:.4f},'
                            .format(pr_aspect_f1, pr_opinion_f1, pr_sentiment_acc, pr_sentiment_f1, pr_ABSA_f1))

                        self.logger.info('Current Max Metrics Index : {} Current Min Loss Index : {} {}'
                              .format(dev_metric_list.index(max(dev_metric_list)), dev_loss_list.index(min(dev_loss_list)), epoch_time))

                    if self.opt.load == 1:
                        break

                self.logger.info('\nMETA EPOCH {} | {:-^80}'.format(j+1, 'Mission Complete'))

                max_dev_index = dev_metric_list.index(max(dev_metric_list))
                self.logger.info('Dev Max Metrics Index: {}'.format(max_dev_index))
                self.logger.info('TEST | aspect f1={:.4f}, opinion f1={:.4f}, sentiment acc={:.4f}, sentiment f1={:.4f}, ABSA f1={:.4f},'
                      .format(aspect_f1_list[max_dev_index], opinion_f1_list[max_dev_index],
                              sentiment_acc_list[max_dev_index],
                              sentiment_f1_list[max_dev_index], ABSA_f1_list[max_dev_index]))
                ## PREDICT UNLABELED log 추가
                self.logger.info('PREDICT UNLABELED | aspect f1={:.4f}, opinion f1={:.4f}, sentiment acc={:.4f}, sentiment f1={:.4f}, ABSA f1={:.4f},'
                      .format(pr_aspect_f1_list[max_dev_index], pr_opinion_f1_list[max_dev_index],
                              pr_sentiment_acc_list[max_dev_index],
                              pr_sentiment_f1_list[max_dev_index], pr_ABSA_f1_list[max_dev_index]))

                min_dev_index = dev_loss_list.index(min(dev_loss_list))
                self.logger.info('Dev Min Loss Index: {}'.format(min_dev_index))
                self.logger.info('TEST aspect f1={:.4f}, opinion f1={:.4f}, sentiment acc={:.4f}, sentiment f1={:.4f}, ABSA f1={:.4f},'
                      .format(aspect_f1_list[min_dev_index], opinion_f1_list[min_dev_index],
                              sentiment_acc_list[min_dev_index],
                              sentiment_f1_list[min_dev_index], ABSA_f1_list[min_dev_index]))
                ## PREDICT UNLABELED log 추가
                self.logger.info('PREDICT UNLABELED | aspect f1={:.4f}, opinion f1={:.4f}, sentiment acc={:.4f}, sentiment f1={:.4f}, ABSA f1={:.4f},'
                      .format(pr_aspect_f1_list[min_dev_index], pr_opinion_f1_list[min_dev_index],
                              pr_sentiment_acc_list[min_dev_index], 
                              pr_sentiment_f1_list[min_dev_index], pr_ABSA_f1_list[min_dev_index]))

                    ##############################################################
                pr_a_preds = pr_a_preds_list[min_dev_index]
                pr_o_preds = pr_o_preds_list[min_dev_index]
                pr_s_preds = pr_s_preds_list[min_dev_index]
                
                pr_a_preds = get_onehot_type(pr_a_preds, 'aspect')
                pr_o_preds = get_onehot_type(pr_o_preds, 'opinion')
                pr_s_preds = get_onehot_type(pr_s_preds, 'sentiment')
                # pr_a_preds = np.argmax(pr_a_preds, axis=-1)
                # pr_a_preds = self.get_one_hot(pr_a_preds, 3)
    
                # pr_o_preds = np.argmax(pr_o_preds, axis=-1)
                # pr_o_preds = self.get_one_hot(pr_o_preds, 3)
    
                # pr_s_preds = np.argmax(pr_s_preds, axis=-1)
                # pr_s_preds = self.get_one_hot(pr_s_preds, 4)
    ##############################################################    
    
                ## predicted_y를 넣어서 new_unlabeled_data 만들기
                ## 기존 애들 불러온 다음.. 새로 구성
                unlabeled_source_data =train_unlabeled_sets[0]
                unlabeled_aspect_y =train_unlabeled_sets[1]
                unlabeled_opinion_y =train_unlabeled_sets[2]
                unlabeled_sentiment_y =train_unlabeled_sets[3]
                unlabeled_source_mask =train_unlabeled_sets[4]
                unlabeled_sentiment_mask =train_unlabeled_sets[5]
                unlabeled_position_m =train_unlabeled_sets[6]
        
                # labeled data
                labeled_source_data =train_labeled_sets[0]
                labeled_aspect_y =train_labeled_sets[1]
                labeled_opinion_y =train_labeled_sets[2]
                labeled_sentiment_y =train_labeled_sets[3]
                labeled_source_mask =train_labeled_sets[4]
                labeled_sentiment_mask =train_labeled_sets[5]
                labeled_position_m = train_labeled_sets[6]
                
                # pr_a_preds = pr_a_preds_list[max_dev_index]
                # pr_o_preds = pr_o_preds_list[max_dev_index]
                # pr_s_preds = pr_s_preds_list[max_dev_index]

                # pr_a_preds = np.argmax(pr_a_preds, axis=-1)
                # pr_a_preds = self.get_one_hot(pr_a_preds, 3)

                # pr_o_preds = np.argmax(pr_o_preds, axis=-1)
                # pr_o_preds = self.get_one_hot(pr_o_preds, 3)

                # pr_s_preds = np.argmax(pr_s_preds, axis=-1)
                # pr_s_preds = self.get_one_hot(pr_s_preds, 3)

                # ## predicted_y를 넣어서 new_unlabeled_data 만들기
                # ## 기존 애들 불러온 다음.. 새로 구성
                # labeled_source_data, labeled_aspect_y, labeled_opinion_y, labeled_sentiment_y, labeled_source_mask, labeled_sentiment_mask, labeled_position_m = train_labeled_sets
                # unlabeled_source_data, unlabeled_aspect_y, unlabeled_opinion_y, unlabeled_sentiment_y, unlabeled_source_mask, unlabeled_sentiment_mask, unlabeled_position_m = train_unlabeled_sets
                
                index_no_aspect = np.array(np.where(np.all(labeled_aspect_y == np.array([1,0,0]), axis=-1))).T ## aspect에서 0 라벨에 해당하는 단어 index 
                index_no_opinion = np.array(np.where(np.all(labeled_opinion_y == np.array([1,0,0]), axis=-1))).T ## opinion에서 0 라벨에 해당하는 단어 index 
                index_no_as_op = np.array([x for x in set(tuple(x) for x in index_no_aspect) & set(tuple(x) for x in index_no_opinion)]) ## 둘의 교집합

                chosen_rows_index = np.random.choice(labeled_source_data.shape[0], int(labeled_source_data.shape[0]*self.opt.shake_ratio), replace=False) ## 얼마나의 문장을 선택할 것인가? 선택받은 문장의 index
                for k in chosen_rows_index:
                    try:
                        temp_index_no_as_op = index_no_as_op[np.where(index_no_as_op[:,0]==k)] ## 해당 row에 해당하는 index_no_as_op 불러오기
                        chosen_indexs = np.random.choice(temp_index_no_as_op.shape[0], 2, replace=False) ## temp_index_no_as_op(단어 중 ) 2개 골라낸다 2개의 temp_index_no_as_op 상에서의 index 찾아냄.
                        index1 = temp_index_no_as_op[chosen_indexs[0]] ## 실제 labeled_source_data에서의 index 값을 저장
                        index2 = temp_index_no_as_op[chosen_indexs[1]]
                        labeled_source_data[index1], labeled_source_data[index2] = labeled_source_data[index2], labeled_source_data[index1]  ## swap!!
                    except:
                        continue
                
                ## 먼저 pr_a_labels의 형태가 unlabeled_aspect_y와 동일한지 확인
                new_train_sets = (np.concatenate((labeled_source_data, unlabeled_source_data), axis=0), \
                                  np.concatenate((labeled_aspect_y, np.array(pr_a_preds)), axis=0), \
                                  np.concatenate((labeled_opinion_y, np.array(pr_o_preds)), axis=0), \
                                  np.concatenate((labeled_sentiment_y, np.array(pr_s_preds)), axis=0), \
                                  np.concatenate((labeled_source_mask, unlabeled_source_mask), axis=0), \
                                  np.concatenate((labeled_sentiment_mask, unlabeled_sentiment_mask), axis=0), \
                                  np.concatenate((labeled_position_m, unlabeled_position_m), axis=0))
                ## 위에서 predict의 결과물 중 final_mask가 있는데 이건 따로 추가를 안해줘도 되는걸까?


    def get_batch_data(self, dataset, batch_size, keep_prob1, keep_prob2, is_training=False, is_shuffle=False):
        length = len(dataset[0])
        all_index = np.arange(length)
        if is_shuffle:
            np.random.shuffle(all_index)
        for i in range(int(length / batch_size) + (1 if length % batch_size else 0)):
            index = all_index[i * batch_size:(i + 1) * batch_size]
            feed_dict = {
                self.x: dataset[0][index],
                self.aspect_y: dataset[1][index],
                self.opinion_y: dataset[2][index],
                self.sentiment_y: dataset[3][index],
                self.word_mask: dataset[4][index],
                self.senti_mask: dataset[5][index],
                self.position: dataset[6][index],
                self.keep_prob1: keep_prob1,
                self.keep_prob2: keep_prob2,
                self.is_training: is_training,
            }
            yield feed_dict, len(index)
