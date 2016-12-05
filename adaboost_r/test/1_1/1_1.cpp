#include<iostream>
using namespace std;
#include <fann.h>
#include <fann_cpp.h>
#include <floatfann.h>
int main()
{
  FANN::neural_net nn;
  const float desired_error = 0.00001;
  const unsigned int max_epochs = 5000;
  const unsigned int epochs_between_reports = 1000;
  const unsigned int layers_count = 3;
  const unsigned int layers[layers_count] = {1, 5, 1};
  nn.create_standard_array(layers_count, layers);
  nn.train_on_file("test.train", max_epochs, epochs_between_reports, desired_error);

  fann_type i[3] = {1,2,4};
  fann_type o[3] = {1,2,4};
 
  fann_type *out_vec;
  out_vec = nn.test(i, o);
  std::cout << "out_vec: " << out_vec[0] << std::endl;
  std::cout << "out_vec: " << out_vec[1] << std::endl;
  std::cout << "out_vec length: " << out_vec[3] << std::endl;
  
}
