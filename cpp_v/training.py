# !usr/bin/python

# training.py
# Qiaojing
# 2016.10.19

import subprocess
import numpy as np
NET_NUM = 2
SAMPLE_NUM = 7000 # the number of samples







########### Training the net ###########
print("# Training the net: \n")
for i in range(1, NET_NUM):

    net = "net_" + str(i)
    trainFile = "./train.data/" + net + "_inversek2j_train.data"

    # Invoke "train_1_hidden_layer" to train the net:
    bashCommand = "bash ./run.sh %s %s " %(net, trainFile)
    print(bashCommand + "\n")
    process = subprocess.Popen(bashCommand.split())
    process.communicate()

    # Update the weight_vector


    # Generate new training data set: "net_2_inversek2j_train.data"
    net = "net_" + str(i+1)
    trainfile = "./train.data/" + net + "_inversek2j_train.data"





