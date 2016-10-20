# !usr/bin/python

# train.py
# Qiaojing
# 2016.10.19

import subprocess
import numpy as np







########### Train the net ###########
net = "net_1"
trainfile = "net_1_inversek2j_train.data"
testfile = "net_1_inversek2j_test.data"

bashCommand = "bash ./run.sh %s %s %s" %(net, trainfile, testfile)
print(bashCommand)
process = subprocess.Popen(bashCommand.split())
process.communicate()
