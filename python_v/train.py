# !/bin/usr/python

###################################
# main.py
# qiao
# 20161012 verion: v0.1
#
###################################

from fann2 import libfann

connection_rate = 1
learning_rate = 0.7
num_input = 2
num_hidden = 4
num_output = 2

desired_error = 0.001
max_iterations = 10000
iterations_between_reports = 1000

# train the network
ann_train = libfann.neural_net()
ann_train.create_sparse_array(connection_rate, (num_input, num_hidden, num_output))
ann_train.set_learning_rate(learning_rate)
ann_train.set_activation_function_output(libfann.SIGMOID_SYMMETRIC_STEPWISE)
ann_train.train_on_file("inversek2j_train.data", max_iterations, iterations_between_reports, desired_error)

ann_train.save("inversek2j.net")

print "# Train MSE%s" %(ann_train.get_MSE())


ann_train.destroy()


