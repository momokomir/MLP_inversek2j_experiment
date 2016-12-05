"""MLP_boosting_algorithm"""
# !usr/bin/python
# -*-coding:utf-8-*-
# python3   source activate py3_env

# training.py
# Qiaojing
# 2016.10.19

import numpy as np
from fann2 import libfann


NET_NUM = 5
SAMPLE_NUM = 7000  # the number of samples
DELTA = 1.5*0.0024178536
weight_vector = np.array([1.0 / SAMPLE_NUM for x in range(SAMPLE_NUM)])
score = [0 for i in range(NET_NUM + 1)]
error_vector = []
APP = "_inversek2j_train.data"


def train(i):
    """train the net"""
    # set parameters
    learning_rate = 0.1
    num_input = 2
    num_hidden = 8
    num_output = 2
    layers = [num_input, num_hidden, num_output]

    desired_error = 0.0001
    max_iterations = 10000
    iterations_between_reports = 1000

    # create ann and set it
    ann = libfann.neural_net()
    ann.create_standard_array(layers)

    ann.set_learning_rate(learning_rate)
    ann.set_activation_function_hidden(libfann.SIGMOID)
    ann.set_activation_function_output(libfann.LINEAR)
    ann.set_training_algorithm(libfann.TRAIN_RPROP)

    # handle trainfile
    net = "net_" + str(i)
    trainfile = "./train.data/" + net + APP

    # read the train_data from file
    train_data = libfann.training_data()
    train_data.read_train_from_file(trainfile)

    # train it
    ann.train_on_data(train_data, max_iterations, iterations_between_reports, desired_error)
    print("Train MSE: %s" % ann.get_MSE())

    del error_vector[:]
    for i in range(SAMPLE_NUM):
        error_vector.append(
            np.sum(np.abs(np.array(ann.run(train_data.get_input()[i])) - np.array(train_data.get_output()[i]))))

    # print(error_vector)

    # read test_data from file
    test_data = libfann.training_data()
    test_data.read_train_from_file("inversek2j_test.data")

    # test
    ann.reset_MSE()
    ann.test_data(test_data)
    print("Test MSE: %s" % ann.get_MSE())

    # destroy the stucture
    # ann.save("inversek2j_train.net")
    ann.destroy()


def hs_func():
    """define a ufunc to caculate hs for np array"""
    def hs(x):
        """hs for number"""
        if x > 0:
            return 1
        else:
            return 0

    return np.frompyfunc(hs, 1, 1)


def exp_func(alpha):
    """define a ufunc to caculate exp for np.array"""
    def func(x):
        """exp function for number"""
        return np.exp(alpha * (x - 1))

    return np.frompyfunc(func, 1, 1)


def cac_weight(weight_vector, i):
    """calculate weight"""
    error_vector_tmp = np.array(error_vector)

    # caculate the total error
    tmp = hs_func()(error_vector_tmp - DELTA)
    e = np.dot(weight_vector, tmp)

    print("overall weighted error e:%s" % e)

    # if e is greater than 0.5, exit the loop
    if e > 0.5:
        i -= 1
        print("ERROR:  e > 0.5, i = %d" % i)
        return 11

    # calculate alpha
    alpha = np.log((1 - e) / (e + 0.000001))

    # update weight_vector
    weight_vector = weight_vector * exp_func(alpha)(tmp)
    weight_vector = weight_vector.astype(np.float64)

    # save a parameter
    # score[i] = sum(hs_func()(DELTA - error_vector_tmp)) * alpha

    # sampling
    prob_vector = weight_vector / sum(weight_vector)
    sample_index = np.random.choice(SAMPLE_NUM, SAMPLE_NUM, p=prob_vector)

    # after sampling, the order of weight changed, so:
    weight_vector = weight_vector[sample_index]
    # np.savetxt("f.data",weight_vector)

    # Generate new trainfile
    net = "net_" + str(i)
    trainfile = "./train.data/" + net + APP

    with open(trainfile) as fin:
        data = fin.read().splitlines(True)
    with open('./tmp/b.data', 'w') as fout:
        fout.writelines(data[1:])

    data = np.loadtxt("./tmp/b.data").reshape(-1, 4)
    new = data[sample_index].reshape(-1, 2)

    np.savetxt("./tmp/c.data", new, fmt="%s")

    # save it
    net = "net_" + str(i + 1)
    trainfile = "./train.data/" + net + APP
    with open("./tmp/c.data") as fc:
        data = fc.read().splitlines(True)
    with open(trainfile, "w") as fd:
        fd.writelines("%s 2 2\n" % SAMPLE_NUM)
        fd.writelines(data)


def main():
    """main function"""
    print("### Training the net: ###\n")

    for i in range(1, NET_NUM + 1):
        print("### Training net: %s ###" % i)
        train(i)

        # Update the weight_vector & generate new dataset
        res = cac_weight(weight_vector, i)
        if res == 11:
            break

    # print the score of each net
    # print("score of each net:%s" % score)
    # print("Delta: 15 * %s" % (DELTA/15))


if __name__ == '__main__':
    main()
