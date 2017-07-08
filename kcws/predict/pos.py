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
if __name__ == '__main__':

    char_dict=read_vocabulary('/home/di/pycharmProjects/segment/kcws-master/kcws/models/basic_vocab.txt')
    word_dict = read_vocabulary('/home/di/pycharmProjects/segment/kcws-master/kcws/models/word_vocab.txt')
    pos_dict_temp = read_vocabulary('/home/di/pycharmProjects/segment/kcws-master/kcws/models/pos_vocab.txt')
    pos_dict=dict(zip(pos_dict_temp.values(), pos_dict_temp.keys()))
    # print len(vocab_dict)
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen_model_filename", default="/home/di/pycharmProjects/segment/kcws-master/kcws/models/pos_model.pbtxt", type=str, help="Frozen model file to import")
    args = parser.parse_args()
    #加载已经将参数固化后的图
    graph = load_graph(args.frozen_model_filename)

    wx = graph.get_tensor_by_name('prefix/input_words:0')
    cx = graph.get_tensor_by_name('prefix/input_chars:0')
    t = graph.get_tensor_by_name('prefix/transitions:0')
    y = graph.get_tensor_by_name('prefix/Reshape_9:0')

    # for op in graph.get_operations():
    #     print(op.name,op.values())
    strings = ['苏剑林','是','科学','空间','的','博主','。']
    num_word = len(strings)
    input_string_vector = []
    char_encode = []
    char_vector = []
    word_zero_list = [0] * (50 - num_word)

    for s in strings:
        # print s
        word_encode=int(word_dict.get(s, 0))
        input_string_vector.append(word_encode)
        for c in s.decode("utf8"):
            num_char=len(s.decode("utf8"))
            char_zero_list = [0] * (5 - num_char)
            if num_char <=5:
                char_encode.append(int(char_dict.get(c.encode("utf8"), 0)))
            else:continue
        if num_char<5:
            char_encode += char_zero_list
        else:continue
        char_vector += char_encode
    char_vector = char_vector + [0]*(250-len(char_vector))
    input_string_vector += word_zero_list

    # print input_string_vector,char_vector
    with tf.Session(graph=graph) as sess:
        feed_dict = {wx: [input_string_vector],
                     cx: [char_vector]}
        unary_score_val, transMatrix = sess.run([y, t], feed_dict)

        print unary_score_val.shape,transMatrix.shape
        for tf_unary_scores_, y_, sequence_length_ in zip(unary_score_val, input_string_vector, [num_word]):
            tf_unary_scores_ = tf_unary_scores_[:num_word]
            viterbi_sequence, _ = tf.contrib.crf.viterbi_decode(tf_unary_scores_, transMatrix)

        print "标签为：", viterbi_sequence
    for j in range(num_word):
        pos_word=pos_dict.get(str(int(viterbi_sequence[j])),0)
        print strings[j],pos_word


    print "*" * 20, "end", "*" * 20
