/* test_nn.cpp
 *  Test the NN trained before.
 *  usage:
 *       ./test_nn net_1_inversek2j.net inversek2j_test.data  
 *
 * Qiaojing
 * 2016.10.19
 *
 * */


# include "fann.h"
# include <iostream>
# include <floatfann.h>

int main(int argc, char* argv[]){
    // ./test_nn net_1_inversek2j.net inversek2j_test.data
    const char* netFile = argv[1];
    std::string testFile = argv[2];

    struct fann *ann = fann_create_from_file(netFile);

    // Testing
    struct fann_train_data *test_data = fann_read_train_from_file(testFile.c_str());
    fann_reset_MSE(ann);
    fann_test_data(ann, test_data);
    std::cout <<  "# Test MSE: " << fann_get_MSE(ann) << std::endl;




    fann_destroy_train(test_data);
    fann_destroy(ann);

    return 0;
}
