#include <iostream>
#include <fstream>
#include <Eigen/Dense>
#include <vector>
#include "SubproblemOptimization.hpp"
#include "MinimumFrobeniusNormModel.hpp"
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
  nodes[0] << 0.50, 0.50;
  nodes[1] << 0.75, 0.50;
  nodes[2] << 0.25, 0.50;
  nodes[3] << 0.50, 0.75;
  nodes[4] << 0.50, 0.25;

  Eigen::VectorXd eval_point ( dimension );
  eval_point << 0.5, 0.5;
  
  BasisForMinimumFrobeniusNormModel basis ( dimension, nb_nodes );
  std::vector<MinimumFrobeniusNormModel> mfn_models;

  MinimumFrobeniusNormModel mfn_model ( basis );

  mfn_models.push_back( mfn_model );
  mfn_models.push_back( mfn_model );
  mfn_models.push_back( mfn_model );

  basis.compute_basis_coefficients ( nodes );

  MyBlackBoxFunction my_blackbox;
  std::vector<Eigen::VectorXd> bb_evals;

  bb_evals.resize ( 3 );


  bb_evals[0].resize ( nb_nodes );
  bb_evals[1].resize ( nb_nodes );
  bb_evals[2].resize ( nb_nodes );

  Eigen::VectorXd vals(3);

  for ( int i = 0; i < nb_nodes; i++) {
    my_blackbox.evaluate( nodes[ i ], vals );
    bb_evals[0](i) = vals(0);
    bb_evals[1](i) = vals(1);
    bb_evals[2](i) = vals(2);
  }

  mfn_models[0].set_function_values( bb_evals[0] );
  mfn_models[1].set_function_values( bb_evals[1] );
  mfn_models[2].set_function_values( bb_evals[2] );


  //std::cout << mfn_models[0].evaluate( eval_point ) << std::endl;
  //std::cout << mfn_models[1].evaluate( eval_point ) << std::endl;
  //std::cout << mfn_models[2].evaluate( eval_point ) << std::endl;


  //std::cout << bb_evals[0] << std::endl;

  double delta = 1e0;
  Eigen::VectorXd inner_boundary_path(2);
  inner_boundary_path << 1e-2, 1e-2;
  Eigen::VectorXd feasibility_thresholds(2);
  feasibility_thresholds << 1e0, 1.0;
  Eigen::VectorXd lb(2);
  Eigen::VectorXd ub(2);
  lb << 0e0, 0e0;
  ub << 1e0, 1e0;
  SubproblemOptimization<MinimumFrobeniusNormModel> so(mfn_models, 
   delta, inner_boundary_path, feasibility_thresholds);
  so.set_lower_bounds ( lb );
  so.set_upper_bounds ( ub );

  double result = so.compute_criticality_measure ( eval_point );

  std::cout << ".result = " << result << std::endl;
  std::cout << eval_point << std::endl;
 
  eval_point << .5, 0.5;
  
  delta = 0.1;

  result = so.compute_trial_point ( eval_point );

  std::cout << ".results = " << result << std::endl;
  std::cout << eval_point << std::endl;

  eval_point << 0.5, 0.5;

  result = so.restore_feasibility ( eval_point );

  std::cout << ".results = " << result << std::endl;
  std::cout << eval_point << std::endl;

  return 1;

  std::cout << mfn_models[0].evaluate( eval_point ) << std::endl;
  std::cout << mfn_models[1].evaluate( eval_point ) << std::endl;
  std::cout << mfn_models[2].evaluate( eval_point ) << std::endl;

  Eigen::VectorXd x_loc(2);
  Eigen::VectorXd fvals(3);
  std::ofstream outputfile ( "surrogate_data_o.dat" );
  if ( outputfile.is_open( ) ) {
    for (double i = -1.0; i <= 2.0; i+=0.01) {
      x_loc(0) = i;
      for (double j = -1.0; j < 2.0; j+=0.01) {
        x_loc(1) = j;
        fvals(0) = mfn_models[0].evaluate( x_loc );
        fvals(1) = mfn_models[1].evaluate( x_loc );
        fvals(2) = mfn_models[2].evaluate( x_loc );
        outputfile << x_loc(0) << "; " << x_loc(1) << "; " << fvals(0)<< "; " << 
                     fvals(1)<< "; " << fvals(2) << std::endl;
      }
    }
    outputfile.close( );
  } else std::cout << "Unable to open file." << std::endl;

  


  return 1;


}
