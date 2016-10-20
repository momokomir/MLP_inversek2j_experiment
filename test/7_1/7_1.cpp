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
  std::cout << "output (run) is " << *o << std::endl;

/*   FANN::training_data dd; */
  // fann_type **input_vector = dd.get_input();
  // fann_type *output_vector = nn.run(*input_vector);

  // std::cout << "input_vector:" << *input_vector <<std::endl;
  /* std::cout << "output_vector:" << *output_vector << std::endl; */

  FANN::training_data dd;
  dd.read_train_from_file("test.train");
  fann_type **i1 = dd.get_input();
  std::cout << "i1 is " << **i1 << std::endl;//i1 is 0.0812069
  std::cout << "***********" << std::endl;//i1 is 0.0812069
  
  //i1 is 0.0812069
  //i1 is 0.0812069
  for (int j = 0; j<4; j++)
  {
    for(int i = 0; i<7; i++)
    {
    std::cout << "i1 is " <<(i1)[j][i] << std::endl;
    }
/*     *********** */
 // i1 is 0.0812069
// i1 is 0.429961
// i1 is 0.0983558
/* i1 is 0.0983558 */

/* i1 is 0.0812069 */
// i1 is 0.0812069
// i1 is 0.429961
// i1 is 0.0983558
/* i1 is 0.0983558 */
/* i1 is 0.0812069 */
// i1 is 0.0812069
// i1 is 0.429961
// i1 is 0.0983558
/* i1 is 0.0983558 */
  /*   i1 is 0.0812069 */
// i1 is 0.0812069
// i1 is 0.429961
// i1 is 0.0983558
/* i1 is 0.0983558 */
  }

  fann_type **k1 = dd.get_output();
  std::cout << "k1 is " << **k1 << std::endl; 


  fann_type *o2 = nn.run(*i1); // 只能得到一个output哎
  std::cout << "o2 is " << *o2 << std::endl;
  std::cout << "o2 is " << o2[1] << std::endl;
  std::cout << "################# " << o2[1] << std::endl;
  for (int t = 0; t < 4; t++)
  {
    std::cout << "o2 is " << *o2++ << std::endl;
    
  
  }
  std::cout << "################# " << o2[1] << std::endl;


  

  fann_type *o3[4];
  o3[0] = nn.test(i1[0], *k1);
  o3[1] = nn.test(i1[1], *k1);
  o3[2] = nn.test(i1[2], *k1);
  o3[3] = nn.test(i1[3], *k1);

  std::cout << "o3 is " << *o3[0] << std::endl;
  std::cout << "o3 is " << *o3[1] << std::endl;
  std::cout << "o3 is " << *o3[2] << std::endl;
  std::cout << "o3 is " << *o3[3] << std::endl;


  std::cout << "***********" << std::endl;//i1 is 0.0812069
  std::cout << "i1 is " <<*(i1[0]) << std::endl;
  std::cout << "i1 is " <<*(i1[1]) << std::endl;
  
  




  return 0;
}
