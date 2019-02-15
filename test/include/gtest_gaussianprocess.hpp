#include "GaussianProcess.hpp"
#include "BlackBoxBaseClass.hpp"
#include "gtest/gtest.h"
#include <Eigen/Core>

//--------------------------------------------------------------------------------
class BlackboxMock : public BlackBoxBaseClass{
    double evaluate_samples ( std::vector<double> const &samples, const unsigned int index, void *param ){
      double mean = 0.;
      for(const double& sample : samples ){
        mean += sample;
      }
      return mean/samples.size();
    }
};


class Wrapper_GaussianProcess : public GaussianProcess 
{
  public:
    Wrapper_GaussianProcess ( int n, double &delta_input, BlackBoxBaseClass* blackbox ) :
       GaussianProcess ( n, delta_input, blackbox) { };

    int test_buildInverse () 
    { 
      double diff;
      std::vector<std::vector<double>> inverse_matlab(nb_gp_nodes);
      inverse_matlab[0] = {0.017009364391436e+3,  -0.067181301923826e+3,   0.058432084209320e+3,   0.004370711030606e+3,  -0.011230418690370e+3,};
      inverse_matlab[1] = {-0.067181301923828e+3,   0.431827516730635e+3,  -0.661351771916356e+3,   0.277639396103871e+3,   0.017558838858369e+3,};
      inverse_matlab[2] = {0.058432084209324e+3,  -0.661351771916369e+3,   1.488787185037279e+3,  -0.975819014100332e+3,   0.092554285455364e+3,};
      inverse_matlab[3] = {0.004370711030603e+3,   0.277639396103886e+3,  -0.975819014100347e+3,   0.827406368424901e+3,  -0.136303808492066e+3,};
      inverse_matlab[4] = {-0.011230418690369e+3,   0.017558838858366e+3,   0.092554285455370e+3,  -0.136303808492069e+3,   0.039277563223752e+3,};

      build_inverse();
      for(int i = 0; i < nb_gp_nodes; ++i){
        for(int j = 0; j < nb_gp_nodes; ++j){
          diff = fabs(L_inverse[i][j] - inverse_matlab[i][j]);
          if( diff > 1e-8){
            std::cout << "FAILED: "<< L_inverse[i][j]  << " - " << inverse_matlab[i][j] << " = " << diff << std::endl;
            return 0;
          }
        }
      }

      return 1; 
    }

    int test_compute_var_meanGP () 
    {
      std::vector<double> xstar = {0.};
      double result_matlab = 9.844240598591604e-04;
      std::vector<double> noise = {0.04, 0.04, 0.04, 0.04, 0.04};

      double var_meanGP = compute_var_meanGP(xstar, noise);
      return (fabs(result_matlab - var_meanGP) < 1e-8); 
    }

    int test_compute_cov_meanGPMC () 
    { 
      std::vector<double> xstar = {0.};
      double result_matlab = 0.001070406968878;

      double cov_meanGPMC = compute_cov_meanGPMC(xstar, 3, 0.04);
      return (fabs(result_matlab - cov_meanGPMC) < 1e-8); 
    }

    /*
    int test_bootstrap_diffGPMC () 
    { 
      std::vector<double> xstar = {0.};
      double result_matlab = 0.001070406968878;
      std::vector<double> samples = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
      double cov_meanGPMC = bootstrap_diffGPMC(xstar, samples, -1);
      std::cout << result_matlab << " vs. " << cov_meanGPMC << std::endl;
      return 1; 
    }
    */
};
//--------------------------------------------------------------------------------


//--------------------------------------------------------------------------------
TEST ( GaussianProcessTest, test_buildInverse ) 
{
  double delta_input = 1.0;
  int dim = 1;

  int nb_of_nodes = 5; 
  std::vector<std::vector<double>> nodes;
  std::vector<double> values = {-2.9730, -0.5505, 0.1033, 0, 0.6533};
  std::vector<double> noise = {0.04, 0.04, 0.04, 0.04, 0.04};
  nodes.resize(nb_of_nodes);
  for(int i = 0; i < nb_of_nodes; ++i){
    nodes[i].resize(dim);
  }
  nodes[0][0] = -0.9998;
  nodes[1][0] = -0.3953;
  nodes[2][0] = -0.1660;
  nodes[3][0] = 0;
  nodes[4][0] = 0.4406;

  BlackBoxBaseClass* blackbox_mock = new BlackboxMock();
  Wrapper_GaussianProcess W(dim, delta_input, blackbox_mock);
  W.build(nodes, values, noise);
  EXPECT_EQ( 1, W.test_buildInverse() );
  delete blackbox_mock;
}
//--------------------------------------------------------------------------------

//--------------------------------------------------------------------------------
TEST ( GaussianProcessTest, test_compute_var_meanGP ) 
{
  double delta_input = 1.0;
  int dim = 1;

  int nb_of_nodes = 5; 
  std::vector<std::vector<double>> nodes;
  std::vector<double> values = {-2.9730, -0.5505, 0.1033, 0, 0.6533};
  std::vector<double> noise = {0.04, 0.04, 0.04, 0.04, 0.04};
  nodes.resize(nb_of_nodes);
  for(int i = 0; i < nb_of_nodes; ++i){
    nodes[i].resize(dim);
  }
  nodes[0][0] = -0.9998;
  nodes[1][0] = -0.3953;
  nodes[2][0] = -0.1660;
  nodes[3][0] = 0;
  nodes[4][0] = 0.4406;

  BlackBoxBaseClass* blackbox_mock = new BlackboxMock();
  Wrapper_GaussianProcess W(dim, delta_input, blackbox_mock);
  W.build(nodes, values, noise);
  W.build_inverse();
  EXPECT_EQ( 1, W.test_compute_var_meanGP() );
  delete blackbox_mock;
}
//--------------------------------------------------------------------------------

//--------------------------------------------------------------------------------
TEST ( GaussianProcessTest, test_compute_cov_meanGPMC ) 
{
  double delta_input = 1.0;
  int dim = 1;

  int nb_of_nodes = 5; 
  std::vector<std::vector<double>> nodes;
  std::vector<double> values = {-2.9730, -0.5505, 0.1033, 0, 0.6533};
  std::vector<double> noise = {0.04, 0.04, 0.04, 0.04, 0.04};
  nodes.resize(nb_of_nodes);
  for(int i = 0; i < nb_of_nodes; ++i){
    nodes[i].resize(dim);
  }
  nodes[0][0] = -0.9998;
  nodes[1][0] = -0.3953;
  nodes[2][0] = -0.1660;
  nodes[3][0] = 0;
  nodes[4][0] = 0.4406;

  BlackBoxBaseClass* blackbox_mock = new BlackboxMock();
  Wrapper_GaussianProcess W(dim, delta_input, blackbox_mock);
  W.build(nodes, values, noise);
  W.build_inverse();
  EXPECT_EQ( 1, W.test_compute_cov_meanGPMC() );
  delete blackbox_mock;
}
//--------------------------------------------------------------------------------

//--------------------------------------------------------------------------------
/*
TEST ( GaussianProcessTest, test_bootstrap_diffGPMC ) 
{
  double delta_input = 1.0;
  int dim = 1;

  int nb_of_nodes = 5; 
  std::vector<std::vector<double>> nodes;
  std::vector<double> values = {-2.9730, -0.5505, 0.1033, 0, 0.6533};
  std::vector<double> noise = {0.04, 0.04, 0.04, 0.04, 0.04};
  nodes.resize(nb_of_nodes);
  for(int i = 0; i < nb_of_nodes; ++i){
    nodes[i].resize(dim);
  }
  nodes[0][0] = -0.9998;
  nodes[1][0] = -0.3953;
  nodes[2][0] = -0.1660;
  nodes[3][0] = 0;
  nodes[4][0] = 0.4406;

  BlackBoxBaseClass* blackbox_mock = new BlackboxMock();
  Wrapper_GaussianProcess W(dim, delta_input, blackbox_mock);
  W.build(nodes, values, noise);
  W.build_inverse();
  EXPECT_EQ( 1, W.test_bootstrap_diffGPMC() );
  delete blackbox_mock;
}
 */
//--------------------------------------------------------------------------------
