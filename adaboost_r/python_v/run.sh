# !/bin/sh

# clean
rm -rf tmp/*.data
rm -rf train.data/*.data
cp inversek2j_train.data train.data/net_1_inversek2j_train.data

# run the algorithm
source activate py3_env
python training.py
