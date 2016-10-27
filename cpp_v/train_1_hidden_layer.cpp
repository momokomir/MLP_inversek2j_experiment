/* train_1_hidden_layer.cpp
 *  Train a NN with only one hidden layer use "FANN" library.
 *  usage:
 *       ./trian_1_hidden_layer net_1 inversek2j_train.data 
 *
 * Qiaojing
 * 2016.10.19
 *
 * */


# include "fann.h"
# include <iostream>
# include <floatfann.h>
# include <fstream>
# include <cmath> /* abs */

int main(int argc, char *argv[])
{
    // ./train_1_hidden_layer net_1 inversek2j_train.data 
    std::string weakerNet = argv[1];
    std::string trainFile = argv[2];

    // some arguments
    unsigned int epochs_between_reports = 1000;
    unsigned int max_epochs = 50000;
    double desired_error = 0.00001;
    double learning_rate = 0.1;
    // Topology of MLP
    const unsigned int num_layers = 3;
    const unsigned int num_input = 2;
    const unsigned int num_neurons_hidden = 2;
    const unsigned int num_output = 2;

    // Create MLP
    struct fann *ann = fann_create_standard(num_layers, num_input, num_neurons_hidden, num_output);
    struct fann_train_data *train_data = fann_read_train_from_file(trainFile.c_str());

    // Set aguments
    fann_set_activation_function_hidden(ann, FANN_SIGMOID);
    fann_set_activation_function_output(ann, FANN_LINEAR);
    fann_set_training_algorithm(ann, FANN_TRAIN_RPROP);

    fann_set_learning_rate(ann, learning_rate);

    // Training
    fann_train_on_data(ann, train_data, max_epochs, epochs_between_reports, desired_error);
    std::string fannSave =  weakerNet + std::string("_inversek2j.net");
    fann_save(ann, fannSave.c_str());
    std::cout << "# Train MSE: " << fann_get_MSE(ann) << std::endl;


    // Caculate the error of each sample and write them to the error file: net_1_error.txt
    std::string errorFile = weakerNet + std::string("_error.txt");
    std::ofstream f1("./error/"+ errorFile);
    if (!f1){
        std::cout << "Failed to write file: " << errorFile << std::endl;
        return 0;
    }
    
    unsigned int sampleNum = fann_length_train_data(train_data);// 用fann的函数读trainFile的样本个数

    std::cout << "# sampleNum: " << sampleNum << std::endl;
    
    for (int i=0; i<sampleNum; i++){
       // test
       // std::cout << *fann_get_train_input(train_data,i) << "<-input   output->" << *fann_get_train_output(train_data,i) << std::endl;
       f1 << std::abs(*fann_run(ann, fann_get_train_input(train_data,i)) - *fann_get_train_output(train_data,i)) << std::endl;
    }

    f1.close();

    // Destroy
    fann_destroy(ann);
    fann_destroy_train(train_data);

} //end main
