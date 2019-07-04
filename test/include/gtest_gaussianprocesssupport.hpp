#include "GaussianProcessSupport.hpp"
#include "BlackBoxBaseClass.hpp"
#include "BlackBoxData.hpp"
#include "gtest/gtest.h"

//--------------------------------------------------------------------------------
class BlackboxMock_GPSupport : public BlackBoxBaseClass{
    double evaluate_samples ( std::vector<double> const &samples, const unsigned int index, std::vector<double>const& x ){
      double mean = 0.;
      for(const double& sample : samples ){
        mean += sample;
      }
      return mean/samples.size();
    }
};

class Wrapper_GaussianProcessSupport : public GaussianProcessSupport
{
    BlackBoxData* evaluations;
    BlackBoxBaseClass* blackbox;

  public:

    Wrapper_GaussianProcessSupport ( ) :
       GaussianProcessSupport ( ) {

      double delta_input = 1.0;
      int dim = 1;

      int nb_of_nodes = 5;
      std::vector<double> values = {-2.9730, -0.5505, 0.1033, 0, 0.6533};
      std::vector<double> noise = {0.04, 0.04, 0.04, 0.04, 0.04};
      std::vector<std::vector<double>> nodes;
      nodes.resize(nb_of_nodes);
      for(int i = 0; i < nb_of_nodes; ++i){
        nodes[i].resize(dim);
      }
      nodes[0][0] = -1;
      nodes[1][0] = -0.45;
      nodes[2][0] = 0;
      nodes[3][0] = 0.5;
      nodes[4][0] = 1;

      evaluations = new BlackBoxData(1, 1);
      evaluations->nodes = nodes;
      evaluations->values.resize(1);
      evaluations->values[0] = values;
      evaluations->noise.resize(1);
      evaluations->noise[0] = noise;
      evaluations->values_MC.resize(1);
      evaluations->values_MC[0] = values;
      evaluations->noise_MC.resize(1);
      evaluations->noise_MC[0] = noise;
      evaluations->active_index = {0, 1, 2, 3, 4};
      evaluations->best_index = 2;

      int number_processes_input = 1;
      delta_input = 1;
      blackbox = new BlackboxMock_GPSupport();
      std::vector<int> update_at_evaluations_input = {6, 10, 20};
      int update_interval_length_input = 10;
      std::string gaussian_process_type = "GP";
      int exitconst = 0;
      bool use_analytic_smoothing = true;

      initialize (dim, number_processes_input, delta_input, blackbox, update_at_evaluations_input,
                    update_interval_length_input, gaussian_process_type, exitconst, use_analytic_smoothing);
      update_gaussian_processes(*evaluations);
      set_gaussian_process_delta(1.);
    };

    int test_compute_fill_width ()
    {
      double result = 0.275;

      double fill_width = compute_fill_width(*evaluations);
      return (fabs(result - fill_width) < 1e-8);
    }

};
//--------------------------------------------------------------------------------


//--------------------------------------------------------------------------------
TEST ( GaussianProcessSupportTest, test_compute_fill_width )
{

  Wrapper_GaussianProcessSupport W;
  EXPECT_EQ( 1, W.test_compute_fill_width() );
}
//--------------------------------------------------------------------------------
