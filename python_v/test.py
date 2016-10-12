# !/bin/usr/python

###################################
# main.py
# qiao
# 20161012 verion: v0.1
#
###################################

from fann2 import libfann

# test
ann_test = libfann.neural_net()
ann_test.create_from_file("inversek2j.net")
testData = ann_test.read_train_from_file("inversek2j_test.data")
ann_test.test("testData")

print "# Test MSE:%s" %(ann_test.get_MSE())

ann_test.destroy()
ann_train.destroy()


