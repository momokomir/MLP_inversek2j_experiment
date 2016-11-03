# !/bin/sh
# net=net_1
# trainfile=inversek2j_train.data
# ./train_1_hidden_layer net_1 ./train.data/net_1_inversek2j_train.data 10000 2 8 2
./train_1_hidden_layer $1 $2 $3 $4 $5 $6
 mv *.net net/
