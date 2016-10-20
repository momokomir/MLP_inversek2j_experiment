# !/bin/sh
# net=net_1
# trainfile=inversek2j_train.data
# ./train_1_hidden_layer net_1 inversek2j_train.data
./train_1_hidden_layer $1 $2
 mv *.net net
