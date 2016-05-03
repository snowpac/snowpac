#include <iostream>
#include <Eigen/Dense>
#include <vector>
//#include "MinimumFrobeniusNormModel.hpp"
#include "BasisForMinimumFrobeniusNormModel.hpp"
#include "ImprovePoisedness.hpp"
#include "MyBlackBoxFunction.cpp"

int main (int argc, const char* argv[]) {

  int dimension = 2;
  int nb_nodes  = 5;

  Eigen::VectorXd f_vals( nb_nodes );
  f_vals << 0e0, 0e0, 0e0, 1e0, -1e0;

  std::vector<Eigen::VectorXd> nodes;
  for (int i = 0; i < nb_nodes; i++)
    nodes.push_back(Eigen::VectorXd::Zero( dimension));
  nodes[0] << 0e0, 0e0;
  nodes[1] << 0.6e0, 0.6e0;
  nodes[2] << 0.6e0, 0.55e0;
  nodes[3] << -.6e0, -0.6e0;
  nodes[4] << -0.55e0, -0.6e0;

  Eigen::VectorXd eval_point ( dimension );
  eval_point << 5e-1, 5e-1;
  
  BasisForMinimumFrobeniusNormModel basis ( dimension, nb_nodes );

  basis.compute_basis_coefficients ( nodes );

/*
  std::cout << "x1" << std::endl;
  MinimumFrobeniusNormModel mfn_model ( basis );
  std::cout << "x2" << std::endl;
  mfn_model.set_function_values ( f_vals );
  std::cout << "x3" << std::endl;
  double result = mfn_model.evaluate_model ( eval_point );
  std::cout << "the value is " << result << std::endl ;

  MyBlackBoxFunction mybb;
  Eigen::VectorXd bb_vals(2);
  mybb.evaluate_blackbox( eval_point, bb_vals );
  std::cout << bb_vals << std::endl;
 */
 
  std::cout << "x1 " <<std::endl;

  ImprovePoisedness ip ( 5e0, basis );

  ip.improve_poisedness( 0, nodes );

  for (int i = 0; i < nodes.size(); i++) {
    std::cout << nodes[i] << std::endl; 
    std::cout << "----------" << std::endl;
  }
   

  return 1;


}
