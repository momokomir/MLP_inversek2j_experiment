/* train_1_hidden_layer.cpp
 *  Train a NN with only one hidden layer use "FANN" library.
 *  usage:
 *      ./train_1_hidden_layer net_1 ./train.data/net_1_inversek2j_train.data 10000 2 8 2
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
# include <stdlib.h> /* strtol */

int main(int argc, char *argv[])
{
    // ./train_1_hidden_layer net_1 ./train.data/net_1_inversek2j_train.data 10000 2 8 2
    /*
     argv[1]:net_1
     argv[2]:inversek2j_train.data
     argv[3]:10000  max_epochs
     argv[4]:2 num_input
     argv[5]:2 num_neurons_hidden
     argv[6]:2 num_output

    */
    std::string weakerNet = argv[1];
    std::string trainFile = argv[2];

    // some arguments
    unsigned int epochs_between_reports = 10000;
    unsigned int max_epochs = strtol(argv[3], NULL, 10);
    double desired_error = 0.00001;
    double learning_rate = 0.1;
    // Topology of MLP
    const unsigned int num_layers = 3;
    const unsigned int num_input = strtol(argv[4], NULL, 10);
    const unsigned int num_neurons_hidden = strtol(argv[5], NULL, 10);
    const unsigned int num_output = strtol(argv[6], NULL, 10);

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
    // f2 to save the train_out
    std::ofstream f2("./error/train_out");
    if (!f2){
        std::cout << "Failed to write file: " << errorFile << std::endl;
        return 0;
    }
    // f3 to save the orig_out
    std::ofstream f3("./error/orig_out");
    if (!f3){
        std::cout << "Failed to write file: " << errorFile << std::endl;
        return 0;
    }
    
    unsigned int sampleNum = fann_length_train_data(train_data);// 用fann的函数读trainFile的样本个数

    //std::cout << "# sampleNum: " << sampleNum << std::endl;
    
    for (int i=0; i<sampleNum; i++){
       // write the error file
       //

       //TODO: why f1, f2, f3 just has 7000 samples?
       //      the correct answer should be 7000*2
       f1 << std::abs(*fann_run(ann, fann_get_train_input(train_data,i)) - *fann_get_train_output(train_data,i)) << std::endl;

       f2 << *fann_run(ann, fann_get_train_input(train_data,i)) << std::endl;
//       if(i == 1){
//            std::cout << *fann_run(ann, fann_get_train_input(train_data,i)) << std::endl;
//       }
       f3 << *fann_get_train_output(train_data,i) << std::endl;
//       if ( i == 1){
//            std::cout << *fann_get_train_output(train_data,i) << std::endl;
//       }
    }

    f1.close();
    f2.close();
    f3.close();

    // Destroy
    fann_destroy(ann);
    fann_destroy_train(train_data);

} //end main
