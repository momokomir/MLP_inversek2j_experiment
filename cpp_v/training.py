# !usr/bin/python
# python3   source acativate py3_env

# training.py
# Qiaojing
# 2016.10.19

import subprocess
import numpy as np

NET_NUM = 5
SAMPLE_NUM = 7000 # the number of samples
DELTA = 15*0.00144656
weight_vector = np.array([1.0/SAMPLE_NUM for x in range(SAMPLE_NUM)])
tt = [0 for x in range(NET_NUM+1)]

# define a ufunc to caculate hs
def hs_func():
    def hs(x):
        if x > 0:
            return 1
        else:
            return 0
    return np.frompyfunc(hs, 1, 1)

# define a ufunc to caculate exp
def exp_func(alpha):
    def func(x):
        return np.exp(alpha * (x - 1))
    return np.frompyfunc(func, 1, 1)


def cac_weight(weight_vector, i):
    net = "net_" + str(i)
    errorFile = "./error/" + net + "_error.txt" # net_1_error.txt

    # read the error_vector from file
    error_vector = np.loadtxt(errorFile)

    # caclulate the total error
    tmp = hs_func()(error_vector - DELTA)
    e = np.dot(weight_vector, tmp)
    #print("tmp:%s"%tmp)

    #print(weight_vector.dtype)
    #print("weight_vector:%s"%weight_vector)
    print("e:%s"%e)

    # if e is greater than 0.5, exit the loop
    if e > 0.5:
        i = i - 1
        print("ERROR:  e > 0.5, i = %d"%i )
        return 11
    # caculate alpha
    alpha = np.log((1-e)/e)
    print("alpha:%s"%alpha)

    # update weight_vecotr
    weight_vector = weight_vector * exp_func(alpha)(tmp)
    weight_vector = weight_vector.astype(np.float64)
    #print(weight_vector.dtype)
    #print("weight_vector:%s"%weight_vector)



    # save a parameter
    tt[i] = sum(hs_func()(DELTA - error_vector)) * alpha

    # Generate new training data set: "net_2_inversek2j_train.data"
    net = "net_" + str(i)
    trainfile = "./train.data/" + net + "_inversek2j_train.data"

    prob_vector = weight_vector / sum(weight_vector)
    sample_index = np.random.choice(SAMPLE_NUM, SAMPLE_NUM, p=prob_vector)
    weight_vector = weight_vector[sample_index]
    # np.savetxt("f.data",weight_vector)

    with open(trainfile, 'r') as fin:
        data = fin.read().splitlines(True)
    with open('b.data', 'w') as fout:
        fout.writelines(data[1:])

    data = np.loadtxt("b.data").reshape(-1, 4)
    new = data[sample_index].reshape(-1, 2)

    np.savetxt("c.data", new, fmt = "%s")

    # save it
    net = "net_" + str(i+1)
    trainfile = "./train.data/" + net + "_inversek2j_train.data"
    with open("c.data", "r") as fc:
        data = fc.read().splitlines(True)
    with open(trainfile, "w") as fd:
        fd.writelines("%s 2 2\n"%SAMPLE_NUM)
        fd.writelines(data)



########### Training the net ###########
print("### Training the net: ###\n")
for i in range(1, NET_NUM+1):

    net = "net_" + str(i)
    trainFile = "./train.data/" + net + "_inversek2j_train.data"

    # Invoke "train_1_hidden_layer" to train the net:
    bashCommand = "bash ./run.sh %s %s " %(net, trainFile)
    print(bashCommand + "\n")
    process = subprocess.Popen(bashCommand.split())
    process.communicate()

    # Update the weight_vector & generate new dataset
    res = cac_weight(weight_vector,i)
    if res == 11:
        break

print("tt:%s"%tt)


