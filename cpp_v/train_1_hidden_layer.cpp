# include "fann.h"
# include <iostream>
# include <floatfann.h>

int main(int argc, char *argv[])
{
    // ./trian_1_hidden_layer net_1 net_1_inversek2j_train.data net_1_inversek2j_test.data  
    std::string weakerNet = argv[1];
    std::string trainFile = argv[2];
    std::string testFile = argv[3];


    // some arguments
    unsigned int epochs_between_reports = 1000;
    unsigned int max_epochs = 10000;
    double desired_error = 0.00001;
    double learning_rate = 0.1;
    // Topology of MLP
    const unsigned int num_layers = 3;
    const unsigned int num_input = 2;
    const unsigned int num_neurons_hidden = 4;
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
    std::string fannSave = weakerNet + std::string("_inversek2j.net");
    fann_save(ann, fannSave.c_str());
    std::cout << "# Train MSE: " << fann_get_MSE(ann) << std::endl;

    // Testing
    struct fann_train_data *test_data = fann_read_train_from_file(testFile.c_str());
    fann_reset_MSE(ann);
    fann_test_data(ann, test_data);
    std::cout <<  "# Test MSE: " << fann_get_MSE(ann) << std::endl;

    // Destroy
    fann_destroy_train(test_data);
    fann_destroy(ann);







}//end main
