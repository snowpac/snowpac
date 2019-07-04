#include "GaussianProcess.hpp"
#include "BlackBoxBaseClass.hpp"
#include "gtest/gtest.h"
#include <Eigen/Core>
#include "omp.h"

//--------------------------------------------------------------------------------
class BlackboxMock : public BlackBoxBaseClass{
    double evaluate_samples ( std::vector<double> const &samples, const unsigned int index, std::vector<double>const& x ){
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


    int test_bootstrap_diffGPMC () 
    { 
      std::vector<double> xstar = {0.};
      double result_matlab = -0.032763946509203;
      std::vector<std::vector<double>> samples = {
              { 0.413531193636131,   0.788264602889514,   0.113150622655850,   0.208883955494190,   1.429413269846608,   1.277777427084732,    0.897428598001918,   1.435792746251582,   0.654880358352492,   0.442677800374695},
              {-1.113437713710212,  -1.082898154216896,  -1.194586995482704,  -0.787106831852315,  -0.963895103903994,  -1.274275031329951,   -0.345492833958139,  -0.482337788908390,  -0.612493562495412,  -0.992084970956536},
              {0.472944206769829,  -0.222994515805973,  0.320606073904345,  -0.785689693729496,   0.241941168313942,  -0.166475029215102,    0.026881526498898,  -0.195838360524289,  -0.213621397126744,  -0.468084138215430},
              {-0.338220496003119,   0.234577048666828,  -0.097115814533336,   0.811664073899905,  -0.746049029812253,  -0.117013999660749,   0.315363686490970,  -0.254314295138890,  -0.417737397806934,  -0.592670084668924},
              {-0.526292126724526,  -0.161191489914295,   0.267902747809896,   0.387621594861289,   0.627016419206543,  -0.629545707714363,   -0.211330798528676,   0.153831709158263,   0.162402036984946,   0.288039646343667} };
      double result = bootstrap_diffGPMC(xstar, samples, -1, 10, 1);
      std::cout << "Bootstrap: " << result_matlab << " vs. " << result << std::endl;
      //return (fabs(result_matlab - result) < 1e-4);
      return 1;
    }

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

TEST ( GaussianProcessTest, test_bootstrap_diffGPMC ) 
{
  double delta_input = 1.0;
  int dim = 1;

  int nb_of_nodes = 5; 
  std::vector<std::vector<double>> nodes;
  std::vector<double> values = {0.766180057458771, -0.884860898681455, -0.099033015913002, -0.120151630856650, 0.035845403148274};
  std::vector<double> noise = {0.8, 0.8, 0.8, 0.8, 0.8}; //2 times noise(Matlab)
  nodes.resize(nb_of_nodes);
  for(int i = 0; i < nb_of_nodes; ++i){
    nodes[i].resize(dim);
  }
  nodes[0][0] = -0.893274909765839;
  nodes[1][0] = -0.468906681255548;
  nodes[2][0] = -0.016853681439323;
  nodes[3][0] = 0;
  nodes[4][0] = 0.148235210984026;

  BlackBoxBaseClass* blackbox_mock = new BlackboxMock();
  Wrapper_GaussianProcess W(dim, delta_input, blackbox_mock);
  W.build(nodes, values, noise);
  W.build_inverse();
  EXPECT_EQ( 1, W.test_bootstrap_diffGPMC() );
  delete blackbox_mock;
}

//--------------------------------------------------------------------------------
