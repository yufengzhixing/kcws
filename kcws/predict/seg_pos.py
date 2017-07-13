#!/usr/bin/python
#-*- coding:utf-8 -*-
import argparse   
import tensorflow as tf
import numpy as np

def read_vocabulary(input_file):
    tmp_vocab = []
    with open(input_file, "r") as f:
        tmp_vocab.extend(f.readlines())
    tmp_vocab = [line.strip().split("\t") for line in tmp_vocab]
    vocab = dict([(x,y) for (x,y) in tmp_vocab])
    return vocab

def load_graph(frozen_graph_filename):
    # We parse the graph_def file
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # We load the graph_def in the default graph
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(
            graph_def,
            input_map=None,
            return_elements=None,
            name="prefix",
            op_dict=None,
            producer_op_list=None
        )
    return graph
def getSegWeights():

    # print len(vocab_dict)
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen_model_filename", default="/home/di/pycharmProjects/segment/kcws-master/kcws/models/seg_model.pbtxt", type=str, help="Frozen model file to import")
    args = parser.parse_args()
    #加载已经将参数固化后的图
    graph = load_graph(args.frozen_model_filename)

    # for op in graph.get_operations():
    #     print(op.name,op.values())

    x = graph.get_tensor_by_name('prefix/input_placeholder:0')
    t = graph.get_tensor_by_name('prefix/transitions:0')
    y = graph.get_tensor_by_name('prefix/Reshape_7:0')

    return x,t,y,graph
def getPosWeights():

    # print len(vocab_dict)
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen_model_filename", default="/home/di/pycharmProjects/segment/kcws-master/kcws/models/pos_model.pbtxt", type=str, help="Frozen model file to import")
    args = parser.parse_args()
    #加载已经将参数固化后的图
    graph = load_graph(args.frozen_model_filename)

    # for op in graph.get_operations():
    #     print(op.name,op.values())

    wx = graph.get_tensor_by_name('prefix/input_words:0')
    cx = graph.get_tensor_by_name('prefix/input_chars:0')
    t = graph.get_tensor_by_name('prefix/transitions:0')
    y = graph.get_tensor_by_name('prefix/Reshape_9:0')

    return wx,cx,t,y,graph
def viterbi_decode(score, transition_params):
  """Decode the highest scoring sequence of tags outside of TensorFlow.

  This should only be used at test time.

  Args:
    score: A [seq_len, num_tags] matrix of unary potentials.
    transition_params: A [num_tags, num_tags] matrix of binary potentials.

  Returns:
    viterbi: A [seq_len] list of integers containing the highest scoring tag
        indicies.
    viterbi_score: A float containing the score for the Viterbi sequence.
  """
  trellis = np.zeros_like(score)
  backpointers = np.zeros_like(score, dtype=np.int32)
  trellis[0] = score[0]

  for t in range(1, score.shape[0]):
    v = np.expand_dims(trellis[t - 1], 1) + transition_params
    trellis[t] = score[t] + np.max(v, 0)
    backpointers[t] = np.argmax(v, 0)

  viterbi = [np.argmax(trellis[-1])]
  for bp in reversed(backpointers[1:]):
    viterbi.append(bp[viterbi[-1]])
  viterbi.reverse()

  viterbi_score = np.max(trellis[-1])
  return viterbi, viterbi_score

if __name__ == '__main__':
    char_dict = read_vocabulary('/home/di/pycharmProjects/segment/kcws-master/kcws/models/basic_vocab.txt')
    seg_x,seg_t, seg_y, seg_graph = getSegWeights()

    word_dict = read_vocabulary('/home/di/pycharmProjects/segment/kcws-master/kcws/models/word_vocab.txt')
    pos_dict_temp = read_vocabulary('/home/di/pycharmProjects/segment/kcws-master/kcws/models/pos_vocab.txt')
    pos_dict=dict(zip(pos_dict_temp.values(), pos_dict_temp.keys()))
    pos_wx,pos_cx,pos_t,pos_y,pos_graph = getPosWeights()

    while True:
        input_string = raw_input('输入语句： > ')
        # 退出
        if input_string == 'quit':
            exit()
        input_string_list = []
        input_string_vec = []
        char_list = []
        word_list = []
        num_string = len(input_string.decode("utf-8"))
        seg_zero_list = [0] * (80 - num_string)

        for word_m in input_string.decode("utf-8"):
            vec=int(char_dict.get(word_m.encode("utf-8"), -1))
            input_string_list.append(word_m)
            input_string_vec.append(vec)
        input_vec= input_string_vec + seg_zero_list
        # print input_string_vec
        with tf.Session(graph=seg_graph) as sess:
            seg_feed_dict = {seg_x: [input_vec]}

            seg_unary_score_val, seg_transMatrix = sess.run([seg_y, seg_t], seg_feed_dict)
            # print 'transMatrix',transMatrix.shape,transMatrix
            # print 'score:',unary_score_val.shape,unary_score_val
            for seg_tf_unary_scores_, pos_y_, pos_sequence_length_ in zip(
                    seg_unary_score_val, input_vec, [num_string]):

                seg_tf_unary_scores_ = seg_tf_unary_scores_[:num_string]
                seg_viterbi_sequence, _ = viterbi_decode(
                    seg_tf_unary_scores_, seg_transMatrix)

            # print "标签为：", seg_viterbi_sequence
        for i in range(num_string):
            if seg_viterbi_sequence[i] == 0:
                word_s = input_string_list[i]
                word_list.append(word_s)
            elif (seg_viterbi_sequence[i] > 0 and seg_viterbi_sequence[i] <3):
                char_list.append(input_string_list[i])
            elif seg_viterbi_sequence[i] == 3:
                char_list.append(input_string_list[i])
                word_m = "".join(char_list)
                word_list.append(word_m)
                char_list = []

#词性标注部分

        num_word = len(word_list)
        input_string_vector = []
        char_encode = []
        char_vector = []
        word_zero_list = [0] * (50 - num_word)

        for w in word_list:
            print w
            word_encode = int(word_dict.get(w.encode("utf8"), 0))
            input_string_vector.append(word_encode)
            num_char = len(w)
            char_zero_list = [0] * (5 - num_char)

            for c in w:

                if num_char <= 5:
                    char_encode.append(int(char_dict.get(c.encode("utf8"), 0)))
                else:
                    continue
            if num_char < 5:
                char_encode += char_zero_list
            else:
                continue
            char_vector += char_encode
            char_encode = []
        char_vector = char_vector + [0] * (250 - len(char_vector))
        input_string_vector += word_zero_list

        print input_string_vector,char_vector
        with tf.Session(graph=pos_graph) as sess:
            pos_feed_dict = {pos_wx: [input_string_vector],
                             pos_cx: [char_vector]}
            pos_unary_score, pos_transMatrix = sess.run([pos_y, pos_t], pos_feed_dict)

            for pos_tf_unary_scores_, pos_y_, pos_sequence_length_ in zip(pos_unary_score, input_string_vector, [num_word]):
                pos_tf_unary_scores_ = pos_tf_unary_scores_[:num_word]
                pos_viterbi_sequence, _ = tf.contrib.crf.viterbi_decode(pos_tf_unary_scores_, pos_transMatrix)

            # print "词性标签为：", pos_viterbi_sequence
        for j in range(num_word):
            pos_word = pos_dict.get(str(int(pos_viterbi_sequence[j])), 0)
            print word_list[j], pos_word



        print "*"*20,"end","*"*20



