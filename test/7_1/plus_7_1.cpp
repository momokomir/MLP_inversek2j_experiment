#include<iostream>
using namespace std;
#include <fann.h>
#include <fann_cpp.h>
#include <floatfann.h>
int main()
{
  FANN::neural_net nn;
  const float desired_error = 0.00001;
  const unsigned int max_epochs = 500000;
  const unsigned int epochs_between_reports = 1000;
  const unsigned int layers_count = 3;
  const unsigned int layers[layers_count] = {7, 5, 1};
  nn.create_standard_array(layers_count, layers);
  nn.train_on_file("test.train", max_epochs, epochs_between_reports, desired_error);

  fann_type i[7];
  i[0] = 0.429961; i[1] = 0.0509753; i[2] = 0.381578; i[3] = 0.0266957; i[4] = 0.000117862; i[5] = 0.00707172; i[6] = 0.0221581;
  fann_type *o = nn.run(i);
  std::cout << "output (run) is " << o[0] << std::endl;

  return 0;
}
