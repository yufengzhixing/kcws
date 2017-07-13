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

    vocab_dict=read_vocabulary('/home/di/pycharmProjects/segment/kcws-master/kcws/models/basic_vocab.txt')
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

    while True:
        input_string = raw_input('输入语句： > ')
        # 退出
        if input_string == 'quit':
            exit()
        input_string_list = []
        input_string_vec = []
        word_list = []
        num_string = len(input_string.decode("utf-8"))
        complement_list = [0]*(80-num_string)

        for word in input_string.decode("utf-8"):
            vec=int(vocab_dict.get(word.encode("utf-8"), -1))
            input_string_list.append(word)
            input_string_vec.append(vec)
        input_vec=input_string_vec+complement_list
        print input_string_vec
        with tf.Session(graph=graph) as sess:
            feed_dict = {x: [input_vec]}

            unary_score_val,transMatrix = sess.run([y,t], feed_dict)
            # print 'transMatrix',transMatrix.shape,transMatrix
            # print 'score:',unary_score_val.shape,unary_score_val
            for tf_unary_scores_, y_, sequence_length_ in zip(
                    unary_score_val, input_vec, [num_string]):

                tf_unary_scores_ = tf_unary_scores_[:num_string]
                viterbi_sequence, _ = viterbi_decode(
                    tf_unary_scores_, transMatrix)

            print "标签为：", viterbi_sequence
        for i in range(num_string):
            if viterbi_sequence[i] == 0:
                print input_string_list[i]
            elif (viterbi_sequence[i] > 0 and viterbi_sequence[i] <3):
                word_list.append(input_string_list[i])
            elif viterbi_sequence[i] == 3:
                word_list.append(input_string_list[i])
                print "".join(word_list)
                word_list=[]



        print "*"*20,"end","*"*20



