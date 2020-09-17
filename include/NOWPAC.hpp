/**************************************************************************************/
/* Copyright (c) 2014, Massachusetts Institute of Technology                          */
/* All rights reserved.                                                               */
/*                                                                                    */
/* Redistribution and use in source and binary forms, with or without modification,   */
/* are permitted provided that the following conditions are met:                      */
/*                                                                                    */
/*  1. Redistributions of source code must retain the above copyright notice,         */
/*     this list of conditions and the following disclaimer.                          */
/*                                                                                    */
/*  2. Redistributions in binary form must reproduce the above copyright notice,      */
/*     this list of conditions and the following disclaimer in the documentation      */
/*     and/or other materials provided with the distribution.                         */
/*                                                                                    */
/* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND    */
/* ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED      */
/* WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. */
/* IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,   */
/* INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT */
/* NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR */
/* PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,  */
/* WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) */
/* ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE         */
/* POSSIBILITY OF SUCH DAMAGE.                                                        */
/**************************************************************************************/


//#ifndef HNOWPAC
//#define HNOWPAC

#include "BlackBoxData.hpp"
//#include "BasisForMinimumFrobeniusNormModel.hpp"
#include "LegendreBasisForMinimumFrobeniusNormModel.hpp"
#include "SubproblemOptimization.hpp"
#include "MinimumFrobeniusNormModel.hpp"
#include "ImprovePoisedness.hpp"
#include "BlackBoxBaseClass.hpp"
#include "GaussianProcessSupport.hpp"
#include "VectorOperations.hpp"
#include "NoiseDetection.hpp"
#include <Eigen/Core>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <random>
#include <iomanip>
#include <fstream>
#include <cassert>
#include <limits>
#include <float.h>
#include <sys/stat.h>
#include <algorithm>    // std::sort

//! NOWPAC
template<class TSurrogateModel = MinimumFrobeniusNormModel, 
         class TBasisForSurrogateModel = LegendreBasisForMinimumFrobeniusNormModel>
class NOWPAC : protected NoiseDetection<TSurrogateModel> {
  private:
    int dim;
    int nb_constraints;
    GaussianProcessSupport gaussian_processes;
    BlackBoxBaseClass *blackbox = NULL;
    TBasisForSurrogateModel surrogate_basis;
    TSurrogateModel surrogate_model_prototype;
    std::vector<TSurrogateModel> surrogate_models;
    std::vector<double> blackbox_values;
    std::vector<double> blackbox_noise;
    std::vector<std::vector<double>> blackbox_samples;
    bool stochastic_optimization;
    void blackbox_evaluator ( std::vector<double> const&, bool );
    void blackbox_evaluator ( );
    void update_surrogate_models ( );
    void update_trustregion ( double );
    bool best_point_is_feasible ( );
    bool last_point_is_feasible ( );
    void add_trial_node( );
    double compute_acceptance_ratio ( );
    void write_to_file( );
    void output_for_plotting( const int&, const int&, std::vector< double > const&);
    void check_parameter_consistency( std::vector<double> const&);
    std::unique_ptr<ImprovePoisedness> surrogate_nodes;
    std::unique_ptr< SubproblemOptimization<TSurrogateModel> > surrogate_optimization;
    const char *double_format = "  %.16e  ";
    const char *int_format    = "  %d  ";
    const char* output_filename = NULL;
    std::FILE *output_file = NULL;
    void *user_data_pointer = NULL;
    BlackBoxData evaluations;
    std::vector<double> x_trial;
    std::vector<double> x_sample;
    bool delta_max_is_set;
    double delta, delta_min, delta_max;
    double omega, theta, gamma, gamma_inc, mu;
    double eta_0, eta_1, eta_2, eps_c;
    double threshold_for_poisedness_constant;
    std::vector<double> inner_boundary_path_constants;
    std::vector<double> max_inner_boundary_path_constants;
    std::vector<double> lower_bound_constraints;
    std::vector<double> upper_bound_constraints;
    double criticality_value = -1;
    double trial_model_value = -1;
    double tmp_dbl = -1;
    double stepsize[2];
    double max_noise;
    int noise_observation_span;
    int nb_allowed_noisy_iterations;
    bool noise_detection;
    bool noise_termination;
    int verbose;
    double acceptance_ratio = -1;
    int replace_node_index = -1;
    int max_number_blackbox_evaluations;
    bool max_number_blackbox_evaluations_is_set;
    int max_number_accepted_steps;
    bool max_number_accepted_steps_is_set;
    int number_accepted_steps;
    std::vector<int> update_at_evaluations;
    int update_interval_length;
    int EXIT_FLAG;
    int NOEXIT;
    double tmp_dbl1 = -1;
    bool use_approx_gaussian_process = false;
    std::string gaussian_process_type = "GP";
    double nonlinear_radius_factor = 1.5; //3.0 matches with GP active index points
    int fixed_seed = -1;
    bool out_of_bounds;
    int output_steps = 1;
    bool use_hard_box_constraints = false;
    bool use_asynchronous_evaluations = false;
    bool use_analytic_smoothing = false;
public:
    //! Constructor
    /*!
     Constructor to set number of design parameters
     \param n number of design parameters
    */
    NOWPAC ( int ); 
    //! Constructor
    /*!
     Constructor to set number of design parameters and output file name
     \param n number of design parameters
     \param fn_output name of output file
    */
    NOWPAC ( int, const char* ); 
    //! Destructor
    ~NOWPAC ( ); 
    //! Set black box evaluator (with constraints)
    /*!
     Function to set black box
     \param bb user implemenation of black box \see BlackBoxBaseClass
     \param m number of constraints
    */
    void set_blackbox ( BlackBoxBaseClass&, int );
    //! Set black box evaluator (without constraints)
    /*!
     Function to set black box
     \param bb user implemenation of black box \see BlackBoxBaseClass
    */
    void set_blackbox ( BlackBoxBaseClass& );
    //! Function to start optimization (without restart option)
    /*!
     Function to start optimization
     \param x initial starting point (input) and optimal solution (output)
     \param val optimal value of objective function on output
    */
    int optimize ( std::vector<double>&, std::vector<double>& );
    //! Function to start optimization (with restart option)
    /*!
     Function to start optimization. If this function is called with bb_data with
     a new instance of BlackBoxData, the optimization is started from the initial point.
     Otherwise, if BlackBoxData stems from a previous optimization run, then the optimization
     will be continued where the previous optimization has been terminated.
     \see BlackBoxData
     \param x initial starting point (input) and optimal solution (output)
     \param val optimal value of objective function on output
     \param bb_data container to store data for restart 
    */
    int optimize ( std::vector<double>&, std::vector<double>&, BlackBoxData& );
    //! Function to set option for optimizer
    /*!
     Function to set option for optimizer:
      - "verbose" [no output (0), 
                   output of final result (1), 
                   output of intermediate steps (2), 
                   full verbosity (3)]
      - "GP_adaption_factor" [ no default | > 0 ]
      - "observation span" [ default : 5 |  >= 2 ]
      - "allowed_noisy_iterations" [ default : 3 | >= 0 ]
      - "max_nb_accepted_steps" [ no default | >= 1]
     \param option_name name of the option to be set (see documentation)
     \param option_value value of the option to be set
    */
    void set_option ( std::string const&, int const& );
    //! Function to set option for optimizer
    /*!
     Function to set option for optimizer:
      - "gamma"
      - "gamma_inc"
      - "omega"
      - "theta"
      - "eta_0"
      - "eta_1"
      - "eps_c"
      - "mu"
      - "geometry_threshold"
      - "eps_b"
     \param option_name name of the option to be set (see documentation)
     \param option_value value of the option to be set
    */
    void set_option ( std::string const&, double const& );
    //! Function to set option for optimizer
    /*!
     Function to set option for optimizer:
      - "stochastic_optimization" [ default : false ]
      - "noise_detection" [ default : false ]
      - "noise_termination" [ default : false ]
     \param option_name name of the option to be set (see documentation)
     \param option_value value of the option to be set
    */
    void set_option ( std::string const&, bool const& );
    //! Function to set option for optimizer
    /*!
     Function to set option for optimizer:
      - "eps_b"
     \param option_name name of the option to be set (see documentation)
     \param option_value value of the option to be set
    */
    void set_option ( std::string const&, std::vector<double> const& );
    //! Function to set option for optimizer
    /*!
     Function to set option for optimizer:
      - "GP_adaption_steps"
     \param option_name name of the option to be set (see documentation)
     \param option_value value of the option to be set
    */
    void set_option ( std::string const&, std::vector<int> const& );

    //! Function to set option for optimizer
    /*!
     Function to set option for optimizer:
      - "Approximate GP"
     \param option_name name of the option to be set (see documentation)
     \param option_value value of the option to be set
    */
    void set_option ( std::string const&, std::string const& );

    //! Function to set user data for black box evaluator 
    /*!
     Function to set user data for black box evaluator. The optimizer does not have 
     access to this data, it is passed to the black box.
        \see BlackBoxBaseClase 
     \param data data to be passed to the black box function provided by the user
    */
    void user_data ( void* );
    //! Function to set lower bound constraints on designs
    /*!
     Function to set lower bound constraints on designs
     \param bounds lower bounds on the design variables 
    */ 
    void set_lower_bounds ( std::vector<double> const& );
    //! Function to set upper bound constraints on designs
    /*!
     Function to set upper bound constraints on designs
     \param bounds upper bounds on the design variables 
    */ 
    void set_upper_bounds ( std::vector<double> const& );
    //! Function to set initial trust region radius
    /*!
     Function to set initial trust region radius
     \param init_delta initial trust region radius
    */
    void set_trustregion ( double const& );
    //! Function to set initial and final trust region radius
    /*!
     Function to set initial and final trust region radius. The optimization is stopped
     whenever the trust region radius falls below the final trust region radius
     \param init_delta initial trust region radius
     \param min_delta final trust region radius
    */
    void set_trustregion ( double const&, double const& );
    //! Function to set maximal trust region radius
    /*! 
     Function to set maximal trust region radius
     \param max_delta maximal trust region radius
    */
    void set_max_trustregion ( double const& );
    //! Function to set maximal number of black box evaluations
    /*!
     Function to set maximal number of black box evaluations
     \param max_number_evaluations maximal number of black box evaluations
    */
    void set_max_number_evaluations ( int const& );

};
//#endif


//--------------------------------------------------------------------------------    
template<class TSurrogateModel, class TBasisForSurrogateModel>
NOWPAC<TSurrogateModel, TBasisForSurrogateModel>::NOWPAC ( 
  int n, const char *fn_output ) : 
    NOWPAC ( n )
{
  struct stat buffer;   
  // check if file exists
  if ( stat (fn_output, &buffer) == 0 && false) {
    // do not overwrite file if it exists
    std::cout << "Error   : Output file already exists." << std::endl;
    EXIT_FLAG = -4;
  } else {
    output_filename = fn_output;
    output_file = fopen(output_filename, "w");
  }
}
//--------------------------------------------------------------------------------

//--------------------------------------------------------------------------------    
template<class TSurrogateModel, class TBasisForSurrogateModel>
NOWPAC<TSurrogateModel, TBasisForSurrogateModel>::~NOWPAC ( ) {
  if ( output_filename != NULL ) fclose( output_file );
}
//--------------------------------------------------------------------------------

//--------------------------------------------------------------------------------    
template<class TSurrogateModel, class TBasisForSurrogateModel>
NOWPAC<TSurrogateModel, TBasisForSurrogateModel>::NOWPAC ( int n ) :
  dim ( n ),
  surrogate_basis ( n ),
  surrogate_model_prototype( surrogate_basis ),
  NoiseDetection<TSurrogateModel>( surrogate_models, delta )
{ 
  stochastic_optimization = false;
  x_trial.resize( dim );
  x_sample.resize( dim );
  delta = 1e0;
  delta_max_is_set = false;
  delta_max = 1e0;
  delta_min = 1e-3;
  threshold_for_poisedness_constant = 5e1;
  omega = 0.4; // 0.6
  theta = 0.5;
  gamma = 0.8;
  gamma_inc = 1.4;
  eta_0 = 0.000001; 
  eta_1 = 0.2;
  eta_2 = 0.7;
  eps_c = 1e-3;
  mu = 1e0;
  nb_constraints = -1;
  max_noise = -1e0;
  noise_observation_span = 5;
  nb_allowed_noisy_iterations = 3;
  noise_termination = false;
  noise_detection   = false;
  stepsize[0] = 1e0; stepsize[1] = 0e0;
  evaluations.max_nb_nodes = (dim*dim + 3*dim + 2)/2; //fully quadratic
//  evaluations.max_nb_nodes = dim +1; //linear
//  evaluations.max_nb_nodes = 2*dim+1; //fully linear
  max_number_blackbox_evaluations_is_set = false;
  max_number_blackbox_evaluations = INT_MAX;
  max_number_accepted_steps_is_set = false;
  max_number_accepted_steps = INT_MAX;
  number_accepted_steps = 0;
  verbose = 3;
  NOEXIT = 100;
  EXIT_FLAG = NOEXIT;
  update_interval_length = 0;//5 * dim;

}
//--------------------------------------------------------------------------------

//--------------------------------------------------------------------------------
template<class TSurrogateModel, class TBasisForSurrogateModel>
void NOWPAC<TSurrogateModel, TBasisForSurrogateModel>::set_option ( 
  std::string const &option_name, double const &option_value )
{
  if ( option_name.compare( "gamma" ) == 0 ) 
    { gamma = option_value; return; }
  if ( option_name.compare( "gamma_inc" ) == 0 ) 
    { gamma_inc = option_value; return; }
  if ( option_name.compare( "omega" ) == 0 ) 
    { omega = option_value; return; }
  if ( option_name.compare( "theta" ) == 0 ) 
    { theta = option_value; return; }
  if ( option_name.compare( "eta_0" ) == 0 ) 
    { eta_0 = option_value; return; }
  if ( option_name.compare( "eta_1" ) == 0 ) 
    { eta_1 = option_value; return; }
  if ( option_name.compare( "eta_2" ) == 0 ) 
    { eta_2 = option_value; return; }
  if ( option_name.compare( "eps_c" ) == 0 ) 
    { eps_c = option_value; return; }
  if ( option_name.compare( "mu" ) == 0 ) 
    { mu = option_value; return; }
  if ( option_name.compare( "geometry_threshold" ) == 0 ) 
    { threshold_for_poisedness_constant = option_value; return; }
  if ( option_name.compare( "eps_b" ) == 0 ) { 
    if ( nb_constraints == -1 ) {
      std::cout << "Warning : Unable to set inner boundary path constants." << std::endl;
      std::cout << "          Please call set_blackbox first." << std::endl;
      return;
    } else {
      for ( int i = 0; i < nb_constraints; i++ ) {
        inner_boundary_path_constants.at( i ) = option_value; 
        max_inner_boundary_path_constants.at( i ) = option_value; 
      }
      return;
    } 
  }
  std::cout << "Warning : Unknown parameter double(" << option_name << ")"<< std::endl;
  return;
}
//--------------------------------------------------------------------------------

//--------------------------------------------------------------------------------
template<class TSurrogateModel, class TBasisForSurrogateModel>
void NOWPAC<TSurrogateModel, TBasisForSurrogateModel>::set_option ( 
  std::string const &option_name, bool const &option_value )
{
  if ( option_name.compare( "stochastic_optimization" ) == 0 ) { 
    stochastic_optimization  = option_value; 
    if ( stochastic_optimization && evaluations.values.size() > 0 ) {
      blackbox_noise.resize( nb_constraints + 1);
    }
    return; 
  }
  if ( option_name.compare( "noise_detection" ) == 0 ) {
    noise_detection = option_value;
    return;
  } 
  if ( option_name.compare( "noise_termination" ) == 0 ) {
    noise_termination = option_value;
    return;
  } 
  if ( option_name.compare( "use_hard_box_constraints" ) == 0){
    use_hard_box_constraints = option_value;
    return;
  }
  if ( option_name.compare( "use_asynchronous_evaluations" ) == 0){
    use_asynchronous_evaluations = option_value;
    return;
  }
  if ( option_name.compare( "use_analytic_smoothing") == 0 ){
    if (option_value)
      assert(stochastic_optimization);
    blackbox_samples.resize( nb_constraints + 1);
    use_analytic_smoothing = option_value;
    return;
  }

  std::cout << "Warning : Unknown parameter bool(" << option_name << ")"<< std::endl;
  return;
}
//--------------------------------------------------------------------------------

//--------------------------------------------------------------------------------
template<class TSurrogateModel, class TBasisForSurrogateModel>
void NOWPAC<TSurrogateModel, TBasisForSurrogateModel>::set_option ( 
  std::string const &option_name, std::vector<double> const &option_value )
{
  if ( option_name.compare( "eps_b" ) == 0 ) { 
    if ( nb_constraints == -1 ) {
      std::cout << "Warning : Unable to set inner boundary path constants." << std::endl;
      std::cout << "          Please call set_blackbox first." << std::endl;
      return;
    } else {
      for ( int i = 0; i < option_value.size(); ++i ) {
        inner_boundary_path_constants.push_back( option_value.at(i) ); 
        max_inner_boundary_path_constants.push_back( option_value.at(i) ); 
      }
      return; 
    }
  }
  std::cout << "Warning : Unknown parameter std::vector<double>(" << option_name << ")"<< std::endl;
  return;
}
//--------------------------------------------------------------------------------

//--------------------------------------------------------------------------------
template<class TSurrogateModel, class TBasisForSurrogateModel>
void NOWPAC<TSurrogateModel, TBasisForSurrogateModel>::set_option ( 
  std::string const &option_name, std::vector<int> const &option_value )
{
  if ( option_name.compare( "GP_adaption_steps" ) == 0 ) { 
    for ( int i = 0; i < option_value.size(); ++i )
      update_at_evaluations.push_back( option_value.at( i ) );
    return;
  }
  std::cout << "Warning : Unknown parameter std::vector<int>(" << option_name << ")"<< std::endl;
  return;
}
//--------------------------------------------------------------------------------
//--------------------------------------------------------------------------------
template<class TSurrogateModel, class TBasisForSurrogateModel>
void NOWPAC<TSurrogateModel, TBasisForSurrogateModel>::set_option ( 
  std::string const &option_name, std::string const &option_value )
{
  if ( option_name.compare( "GP_type" ) == 0 ) {
    use_approx_gaussian_process = false;
    if( option_value.compare( "GP" ) == 0){
      use_approx_gaussian_process = false;
    }else if( option_value.compare( "SOR" ) == 0){
      use_approx_gaussian_process = true;
    }else if( option_value.compare( "DTC" ) == 0){
      use_approx_gaussian_process = true;
    }else if( option_value.compare( "FITC" ) == 0){
      use_approx_gaussian_process = true;
    }else{
      std::cout << "NOWPAC: No value set for GP type. Set to default Full Gaussian Process." << std::endl;
      use_approx_gaussian_process = false;
    }
    gaussian_process_type = option_value;
    return;
  }
  std::cout << "Warning : Unknown parameter std::string(" << option_name << ")"<< std::endl;
  return;
}
//--------------------------------------------------------------------------------

//--------------------------------------------------------------------------------
// sets options for integer values
//--------------------------------------------------------------------------------
template<class TSurrogateModel, class TBasisForSurrogateModel>
void NOWPAC<TSurrogateModel, TBasisForSurrogateModel>::set_option ( 
  std::string const &option_name, int const &option_value )
{
  if ( option_name.compare( "verbose" ) == 0 ) { 
    verbose = option_value; 
    return;
  }
  if ( option_name.compare( "GP_adaption_factor" ) == 0 ) { 
    update_interval_length = option_value; 
    return;
  }
  if ( option_name.compare( "observation_span" ) == 0 ) {
    noise_observation_span = option_value;
    return;
  } 
  if ( option_name.compare( "allowed_noisy_iterations" ) == 0 ) {
    nb_allowed_noisy_iterations = option_value;
    return;
  } 
  if ( option_name.compare( "max_nb_accepted_steps" ) == 0 ) {
    max_number_accepted_steps_is_set = true;
    max_number_accepted_steps = option_value;
    return;
  }
  if ( option_name.compare( "seed" ) == 0 ){
    if(option_value < 0){
      std::cout << "Error: Given seed value has to be equal or larger than 0! Exit!" << std::endl;
      exit(-1);
    }
    fixed_seed = option_value;
    return;
  }
  std::cout << "Warning : Unknown parameter (" << option_name << ")"<< std::endl;
  return;
}
//--------------------------------------------------------------------------------

//--------------------------------------------------------------------------------
template<class TSurrogateModel, class TBasisForSurrogateModel>
void NOWPAC<TSurrogateModel, TBasisForSurrogateModel>::set_lower_bounds ( 
  std::vector<double> const& bounds )
{
  lower_bound_constraints = bounds;
  return;
}
//--------------------------------------------------------------------------------

//--------------------------------------------------------------------------------
// sets lower bound constraints for optimization
//--------------------------------------------------------------------------------
template<class TSurrogateModel, class TBasisForSurrogateModel>
void NOWPAC<TSurrogateModel, TBasisForSurrogateModel>::set_upper_bounds ( 
  std::vector<double> const& bounds )
{
  upper_bound_constraints = bounds;
  return;
}
//--------------------------------------------------------------------------------


//--------------------------------------------------------------------------------
// sets initial and minimal trust region radius
//--------------------------------------------------------------------------------
template<class TSurrogateModel, class TBasisForSurrogateModel>
void NOWPAC<TSurrogateModel, TBasisForSurrogateModel>::set_trustregion ( 
  double const &init_delta, double const &min_delta )
{
  delta = init_delta;
  delta_min = min_delta;
  return;
}
//--------------------------------------------------------------------------------

//--------------------------------------------------------------------------------
// sets initial trust region radius,
// minimal trust region radius is not used as stopping criterion
//--------------------------------------------------------------------------------
template<class TSurrogateModel, class TBasisForSurrogateModel>
void NOWPAC<TSurrogateModel, TBasisForSurrogateModel>::set_trustregion ( 
  double const &init_delta )
{
  set_trustregion ( init_delta, delta_min );
  return;
}
//--------------------------------------------------------------------------------

//--------------------------------------------------------------------------------
// sets maximal trust region radius
//--------------------------------------------------------------------------------
template<class TSurrogateModel, class TBasisForSurrogateModel>
void NOWPAC<TSurrogateModel, TBasisForSurrogateModel>::set_max_trustregion ( 
  double const &max_delta )
{
  delta_max_is_set = true;
  delta_max = max_delta;
  return;
}
//--------------------------------------------------------------------------------

//--------------------------------------------------------------------------------
// sets maximal number of black box evaluations before termination
//--------------------------------------------------------------------------------
template<class TSurrogateModel, class TBasisForSurrogateModel>
void NOWPAC<TSurrogateModel, TBasisForSurrogateModel>::set_max_number_evaluations ( 
  int const &max_number_evaluations )
{
  max_number_blackbox_evaluations = max_number_evaluations;
  max_number_blackbox_evaluations_is_set = true;
  return;
}
//--------------------------------------------------------------------------------

//--------------------------------------------------------------------------------
// sets the black box with constraints
//--------------------------------------------------------------------------------
template<class TSurrogateModel, class TBasisForSurrogateModel>
void NOWPAC<TSurrogateModel, TBasisForSurrogateModel>::set_blackbox ( 
  BlackBoxBaseClass &bb, int m )
{
  blackbox = &bb;
  nb_constraints = m;
  blackbox_values.resize( (nb_constraints+1) );
  evaluations.initialize ( nb_constraints+1, dim );
  if ( stochastic_optimization ) {
    blackbox_noise.resize( nb_constraints + 1);
  }
  inner_boundary_path_constants.resize( nb_constraints );
  max_inner_boundary_path_constants.resize( nb_constraints );
  for (int i = 0; i < nb_constraints; ++i) {
    inner_boundary_path_constants.at( i ) = 1e1; 
    max_inner_boundary_path_constants.at( i ) = 1e1; 
  }
  return;
}
//--------------------------------------------------------------------------------

//--------------------------------------------------------------------------------
// sets the black box without constraints
//--------------------------------------------------------------------------------
template<class TSurrogateModel, class TBasisForSurrogateModel>
void NOWPAC<TSurrogateModel, TBasisForSurrogateModel>::set_blackbox ( 
  BlackBoxBaseClass &bb )
{
  set_blackbox ( bb, 0 );
  return;
}
//--------------------------------------------------------------------------------

//--------------------------------------------------------------------------------
template<class TSurrogateModel, class TBasisForSurrogateModel>
double NOWPAC<TSurrogateModel, TBasisForSurrogateModel>::compute_acceptance_ratio ( )
{

  //acceptance_ratio = ( evaluations.values[0].at( evaluations.best_index ) - 
  //                     evaluations.values[0].back() )  /
  //                   ( evaluations.values[0].at( evaluations.best_index ) - 
  //                     surrogate_models[0].evaluate( evaluations.transform(x_trial) ) );

  bool cur_point_is_feasible = true;

  for (int i = 0; i < nb_constraints; ++i ){
    if(surrogate_models[i+1].evaluate( evaluations.transform(x_trial) ) > 0.){
        cur_point_is_feasible = false;
        break;
    }
  }
  
  double numerator, denominator;
  double R_best, R_last;
  double m_best, m_last;
  if (cur_point_is_feasible){ 
    R_best = evaluations.values[0].at( evaluations.best_index );
    R_last = evaluations.values[0].back();
    m_best = surrogate_models[0].evaluate( evaluations.transform(evaluations.nodes[evaluations.best_index]) );
    m_last = surrogate_models[0].evaluate( evaluations.transform(x_trial) );
  }else{ //Feasibility restoration mode, different objective
    R_best = 0.;
    R_last = 0.;
    m_best = 0.;
    m_last = 0.;
    for (int i = 0; i < nb_constraints; ++i ) {
      if (surrogate_models[i+1].evaluate( evaluations.transform(x_trial) ) > 0.){
        R_best += evaluations.values[i+1].at( evaluations.best_index )*evaluations.values[i+1].at( evaluations.best_index );
        R_last += evaluations.values[i+1].back() * evaluations.values[i+1].back();
        m_best += surrogate_models[i+1].evaluate( evaluations.transform(evaluations.nodes[evaluations.best_index]) ) * 
                  surrogate_models[i+1].evaluate( evaluations.transform(evaluations.nodes[evaluations.best_index]) );
        m_last += surrogate_models[i+1].evaluate( evaluations.transform(x_trial) ) * 
                  surrogate_models[i+1].evaluate( evaluations.transform(x_trial) );
      }
    }
  }

  numerator = R_best - R_last;
  denominator = m_best - m_last;

  acceptance_ratio = (std::fabs(denominator) > DBL_MIN) ? numerator / denominator : numerator;

  if(verbose >= 3){
    std::cout << "*******************************" << std::endl; 
    if (!cur_point_is_feasible) std::cout << "#AcceptRatio# #FEASRES#: Feasibility restoration acceptance ratio" << std::endl;
    std::cout << "#AcceptRatio# x_trial: [ "; 
    for(int i = 0; i < x_trial.size(); ++i){
     std::cout << x_trial[i] << " ";
    }
    std::cout << "]" << std::endl;
    std::cout << "#AcceptRatio# R_best: " <<  R_best << " R_last: " << R_last  << " numerator: " << numerator << std::endl;
    std::cout << "#AcceptRatio# m_best: " <<  m_best << " m_last: " << m_last << " denominator: " << denominator << std::endl;
    std::cout << "#AcceptRatio# Acceptance ratio: " << acceptance_ratio << " eta_1: " << eta_1 << " eta_0: " << eta_0 << std::endl; 
    std::cout << "*******************************" << std::endl; 
    std::cout << std::flush; 
  }
  return acceptance_ratio;
}
//--------------------------------------------------------------------------------

//--------------------------------------------------------------------------------
// sets user data that is passed through (S)NOWPAC to the black box function
//--------------------------------------------------------------------------------
template<class TSurrogateModel, class TBasisForSurrogateModel>
void NOWPAC<TSurrogateModel, TBasisForSurrogateModel>::user_data ( void *data ) 
{ 
  user_data_pointer = data; 
  return; 
}
//--------------------------------------------------------------------------------

//--------------------------------------------------------------------------------
template<class TSurrogateModel, class TBasisForSurrogateModel>
void NOWPAC<TSurrogateModel, TBasisForSurrogateModel>::blackbox_evaluator ( 
  std::vector<double> const &x, bool set_node_active ) 
{
  if ( evaluations.nodes.size() >= max_number_blackbox_evaluations) {
    EXIT_FLAG = 1;
    return;
  }
  // evaluate blackbox and check results for consistency
  if ( stochastic_optimization ) {
    if(use_analytic_smoothing){
      blackbox->evaluate(x, blackbox_values, blackbox_noise, blackbox_samples, user_data_pointer);
    }else {
      blackbox->evaluate(x, blackbox_values, blackbox_noise, user_data_pointer);
    }
    if ( blackbox_noise.size() != (unsigned) (nb_constraints+1) ) EXIT_FLAG = -5;
    for ( int i = 0; i < nb_constraints+1; ++i ) {
      if ( blackbox_noise[i] < 0.0 || blackbox_noise[i] != blackbox_noise[i] ) {
        EXIT_FLAG = -5;
        return;
      }
    }
  } else {
    blackbox->evaluate( x, blackbox_values, user_data_pointer ); 
  }
  if ( blackbox_values.size() != (unsigned) (nb_constraints+1) ) EXIT_FLAG = -5;
  for ( int i = 0; i < nb_constraints+1; ++i ) {
    if ( blackbox_values[i] != blackbox_values[i] ) {
      EXIT_FLAG = -5;
      return;
    }
  }


  // add evaluations to blackbox data
  evaluations.nodes.push_back( x );
  for (int i = 0; i < nb_constraints+1; ++i) {
    evaluations.values[i].push_back( blackbox_values.at(i) );
    if ( stochastic_optimization ) {
      evaluations.values_MC[i].push_back( blackbox_values.at(i) );
      evaluations.noise[i].push_back( blackbox_noise.at(i) );
      evaluations.noise_MC[i].push_back( blackbox_noise.at(i) );
      if(use_analytic_smoothing){
        evaluations.samples[i].push_back(blackbox_samples[i]);
      }
    }
  }

  if ( set_node_active )
    evaluations.active_index.push_back( evaluations.nodes.size()-1 );    

  if ( stochastic_optimization && evaluations.nodes.size() > dim ) {
    if(use_approx_gaussian_process){
          gaussian_processes.set_constraint_ball_center(evaluations.nodes[evaluations.best_index]);
          gaussian_processes.set_constraint_ball_radius(nonlinear_radius_factor*delta);
    }
    EXIT_FLAG = gaussian_processes.smooth_data ( evaluations );
  }

  write_to_file();

  //TODO Remove this later
  if(false){
    output_for_plotting(output_steps, 0, x);
    output_steps++;
  }
  return;
}  
//--------------------------------------------------------------------------------

//--------------------------------------------------------------------------------
template<class TSurrogateModel, class TBasisForSurrogateModel>
void NOWPAC<TSurrogateModel, TBasisForSurrogateModel>::blackbox_evaluator ( ) 
{
  int nb_nodes_tmp = evaluations.nodes.size();
  int nb_vals_tmp = evaluations.values[0].size( );
  if ( nb_nodes_tmp == nb_vals_tmp ) return;

  if ( nb_nodes_tmp >= max_number_blackbox_evaluations) {
    EXIT_FLAG = 1;
    return;
  }

  if(!use_asynchronous_evaluations){
    for ( int i = nb_vals_tmp; i < nb_nodes_tmp; ++i ) {
      // evaluate blackbox and check results for consistency
      if ( stochastic_optimization ) { //TODO: PARALLEL EVAL HERE
        if(use_analytic_smoothing){
          blackbox->evaluate(evaluations.nodes[ i ], blackbox_values, blackbox_noise, blackbox_samples, user_data_pointer);
        }else {
          blackbox->evaluate(evaluations.nodes[ i ], blackbox_values, blackbox_noise, user_data_pointer);
        }
        if ( blackbox_noise.size() != (unsigned) (nb_constraints+1) ) EXIT_FLAG = -5;
        for ( int i = 0; i < nb_constraints+1; ++i ) {
          if ( blackbox_noise[i] < 0.0 || blackbox_noise[i] != blackbox_noise[i] ) {
            EXIT_FLAG = -5;
            return;
          }
        }
      } else {
        blackbox->evaluate( evaluations.nodes[ i ], blackbox_values, 
                            user_data_pointer ); 
      }

      if ( blackbox_values.size() != (unsigned) (nb_constraints+1) ) EXIT_FLAG = -5;
      for ( int i = 0; i < nb_constraints+1; ++i ) {
        if ( blackbox_values[i] != blackbox_values[i] ) {
          EXIT_FLAG = -5;
          return;
        }
      }

      // add evaluations to blackbox data
      for (int j = 0; j < nb_constraints+1; ++j) {
        evaluations.values[j].push_back( blackbox_values.at(j) );
        if ( stochastic_optimization ) {
          evaluations.values_MC[j].push_back( blackbox_values.at(j) );
          evaluations.noise[j].push_back( blackbox_noise.at(j) );
          evaluations.noise_MC[j].push_back( blackbox_noise.at(j) );
          if(use_analytic_smoothing){
            evaluations.samples[j].push_back(blackbox_samples[j]);
          }
        }
      }  
    }
  }else{
    int tmp_nb_evals = nb_nodes_tmp-nb_vals_tmp;
    std::vector<std::vector<double>> blackbox_values_tmp;
    blackbox_values_tmp.resize(tmp_nb_evals);
    for(int i = 0; i < tmp_nb_evals; ++i){
      blackbox_values_tmp[i].resize( nb_constraints + 1 );
    }
    std::vector<std::vector<double>> blackbox_noise_tmp;
    if( stochastic_optimization ){
      blackbox_noise_tmp.resize(tmp_nb_evals);
      for(int i = 0; i < tmp_nb_evals; ++i){
        blackbox_noise_tmp.resize( nb_constraints + 1 );
      }
    }
    for ( int i = nb_vals_tmp; i < nb_nodes_tmp; ++i ) {
      if ( stochastic_optimization ) { //TODO: PARALLEL EVAL HERE
        blackbox->evaluate_nowait( evaluations.nodes[ i ], blackbox_values_tmp[ i - nb_vals_tmp ], 
                            blackbox_noise_tmp[ i - nb_vals_tmp], user_data_pointer );
      } else {
        blackbox->evaluate_nowait( evaluations.nodes[ i ], blackbox_values_tmp[ i - nb_vals_tmp ], 
                            user_data_pointer ); 
      }
    }
    blackbox->synchronize();
    if( stochastic_optimization ){
      for ( int i = 0; i < tmp_nb_evals; ++i ) {
        if ( blackbox_noise_tmp[i].size() != (unsigned) (nb_constraints+1) ) EXIT_FLAG = -5;

        for ( int j = 0; j < nb_constraints+1; ++j ) {
          if ( blackbox_noise_tmp[i][j] < 0.0 ) {
            EXIT_FLAG = -5;
            return;
          }
        }

      }
    }
    for ( int i = 0; i < tmp_nb_evals; ++i ) {
      if ( blackbox_values_tmp[i].size() != (unsigned) (nb_constraints+1) ) EXIT_FLAG = -5;
    }
    // add evaluations to blackbox data
    for ( int i = 0; i < tmp_nb_evals; ++i ) {
      for (int j = 0; j < nb_constraints+1; ++j) {
        evaluations.values[j].push_back( blackbox_values_tmp.at(i).at(j) );
        if ( stochastic_optimization ) {
          evaluations.values_MC[j].push_back( blackbox_values_tmp.at(i).at(j) );
          evaluations.noise[j].push_back( blackbox_noise_tmp.at(i).at(j) );
          evaluations.noise_MC[j].push_back( blackbox_noise_tmp.at(i).at(j) );
          if(use_analytic_smoothing){
              evaluations.samples[j].push_back(blackbox_samples[j]);
          }
        }
      }
    }
  }

  assert ( evaluations.nodes.size() == evaluations.values[0].size() );

  if ( stochastic_optimization ){
    if(use_approx_gaussian_process){
          gaussian_processes.set_constraint_ball_center(evaluations.nodes[evaluations.best_index]);
          gaussian_processes.set_constraint_ball_radius(nonlinear_radius_factor*delta);
    } 
    EXIT_FLAG = gaussian_processes.smooth_data ( evaluations );
  }


  write_to_file();
  //TODO: Remove this later
  if(false){
    for(int i = nb_vals_tmp; i < nb_nodes_tmp; ++i){
      output_for_plotting(output_steps, i-nb_vals_tmp, evaluations.nodes[ i ]);
    }
    output_steps++;
  }
  return;
}  
//--------------------------------------------------------------------------------


//--------------------------------------------------------------------------------
template<class TSurrogateModel, class TBasisForSurrogateModel>
bool NOWPAC<TSurrogateModel, TBasisForSurrogateModel>::best_point_is_feasible ( ) 
{

  bool best_point_is_feasible = true;

  // Check if current best point is feasible
  for (int i = 0; i < nb_constraints; ++i){
    if(evaluations.values[i+1][evaluations.best_index] > 0.){
      best_point_is_feasible = false;
      break;
    }
  }
  
  return best_point_is_feasible;
}
//--------------------------------------------------------------------------------

//--------------------------------------------------------------------------------
template<class TSurrogateModel, class TBasisForSurrogateModel>
bool NOWPAC<TSurrogateModel, TBasisForSurrogateModel>::last_point_is_feasible ( ) 
{
  bool point_is_feasible = true;

  bool best_point_is_feasible = true;
  bool back_point_is_feasible = true;

  // Check if new point is feasible
  for (int i = 0; i < nb_constraints; ++i){
    if(evaluations.values[i+1].back() > 0.){
      back_point_is_feasible = false;
    }
  } 

  // Check if current best point is feasible
  for (int i = 0; i < nb_constraints; ++i){
    if(evaluations.values[i+1][evaluations.best_index] > 0.){
      best_point_is_feasible = false;
    }
  }
  
  if(!best_point_is_feasible && back_point_is_feasible){ 
    //From infeasible to feasible: Take this point
    point_is_feasible = true;
  }else if(!best_point_is_feasible && !back_point_is_feasible){ 
    //From infeasible to infeasible: Check if Feas Restore Condition improved based on robust measures  
    point_is_feasible = true;
    double feasiblity_obj_best = 0.;
    double feasiblity_obj_last = 0.;
    for (int i = 0; i < nb_constraints; ++i){
      if(evaluations.values[i+1].back() > 0.){
        feasiblity_obj_best += evaluations.values[i+1][evaluations.best_index]
                               * evaluations.values[i+1][evaluations.best_index] 
                             + delta * evaluations.values[i+1][evaluations.best_index];
        feasiblity_obj_last += evaluations.values[i+1].back()
                               * evaluations.values[i+1].back() 
                             + delta * evaluations.values[i+1].back();
      }
    } 
    if(feasiblity_obj_last >= feasiblity_obj_best){
      point_is_feasible = false;
    }
  }else if(best_point_is_feasible && !back_point_is_feasible){ 
    //From feasible to infeasible: Reject this point, update inner boundary path
    for (int i = 0; i < nb_constraints; ++i){
      if(evaluations.values[i+1].back() > 0.){
        inner_boundary_path_constants.at(i) *= 2e0;
      }
    } 
    point_is_feasible = false;
  }else{
    //Both are feasible  
    point_is_feasible = true;
  }
  
  return point_is_feasible;
}
//--------------------------------------------------------------------------------

//--------------------------------------------------------------------------------
template<class TSurrogateModel, class TBasisForSurrogateModel>
void NOWPAC<TSurrogateModel, TBasisForSurrogateModel>::update_surrogate_models ( )
{
  bool best_index_in_active_set = false;
  for (int i = 0; i < evaluations.active_index.size(); ++i) {
    if ( evaluations.active_index[i] == evaluations.best_index) {
      best_index_in_active_set = true;
      break;
    }    
  }

  //if(!best_index_in_active_set){
  //  evaluations.active_index.push_back(evaluations.best_index);
  //}

  assert( best_index_in_active_set );

  surrogate_basis.compute_basis_coefficients ( evaluations.get_scaled_active_nodes( delta ) ) ;

  for (int i = 0; i < nb_constraints+1; ++i ) {
    surrogate_models[ i ].set_function_values ( evaluations.get_active_values(i) );
  }

  return;
}
//--------------------------------------------------------------------------------

//--------------------------------------------------------------------------------
template<class TSurrogateModel, class TBasisForSurrogateModel>
void NOWPAC<TSurrogateModel, TBasisForSurrogateModel>::update_trustregion ( 
  double scaling_factor )
{
  tmp_dbl = max_noise;
  max_noise = 0e0;
    std::vector<double> noise_per_function;
    noise_per_function.resize(1+nb_constraints);
    for ( int j = 0; j < nb_constraints+1; ++j ) {
        noise_per_function[j] = 0e0;
    }
  if ( stochastic_optimization ) {
    for ( int i = 0; i < evaluations.active_index.size(); ++i ) {
      if ( evaluations.active_index[ i ] != evaluations.best_index ) continue;
      for ( int j = 0; j < nb_constraints+1; ++j ) {
        if ( evaluations.noise[ j ].at( evaluations.active_index[ i ] ) > max_noise ) {
          if ( this->diff_norm( 
                 evaluations.nodes[ evaluations.active_index[ i ] ], 
                 evaluations.nodes[ evaluations.best_index ] ) <= delta )
            max_noise = evaluations.noise[ j ].at( evaluations.active_index[ i ] );
        }
        if ( evaluations.noise[ j ].at( evaluations.active_index[ i ] ) > noise_per_function[j] ) {
          if ( this->diff_norm(
                  evaluations.nodes[ evaluations.active_index[ i ] ],
                  evaluations.nodes[ evaluations.best_index ] ) <= delta )
            noise_per_function[j] = evaluations.noise[ j ].at( evaluations.active_index[ i ] );
        }
      }
    }
  }


 if ( tmp_dbl >= 0e0 ) {
   if ( max_noise > 2e0*tmp_dbl ) max_noise = 2e0*tmp_dbl;
 } 

 if ( noise_detection && scaling_factor >= 1e0) this->reset_noise_detection();

  delta *= scaling_factor;
  if(verbose >= 3){
    std::cout << std::endl << "------------------------- " << std::endl;
    std::cout << "#Noise#Cur delta " << delta << " Scaling factor: " << scaling_factor << std::endl;
    std::cout << "#Noise#MAXNOISE " << max_noise << " sqrt = " << sqrt( max_noise ) << std::endl;
    std::cout << "#Noise#   ObjF " << noise_per_function[0] << " sqrt = " << sqrt( noise_per_function[0] ) << std::endl;
    for ( int j = 1; j < nb_constraints+1; ++j ) {
      std::cout << "#Noise#   C" << j << " " << noise_per_function[j] << " sqrt = " << sqrt( noise_per_function[j] ) << std::endl;
    }
    std::cout <<  "------------------------- " << std::endl;
  }
  if ( stochastic_optimization ) {
    double ar_tmp = acceptance_ratio;
    if ( ar_tmp < 0e0 ) ar_tmp = -ar_tmp;
    if (ar_tmp < 1e0) ar_tmp = 1e0;
    if (ar_tmp > 2e0) ar_tmp = 2e0;
      ar_tmp = 1e0;//sqrt(2.0);
    if ( delta < sqrt(1e0*max_noise)*1e0*ar_tmp  ){ 
      //std::cout << "#Noise#   Apply lower bound to: " << delta << " to " << sqrt(1e0*max_noise) * 1e0 * ar_tmp << std::endl;
      delta = sqrt(1e0*max_noise) * 1e0 * ar_tmp; 
    }
  }
  if ( delta > delta_max ){
    //std::cout << "#Noise#   Setting delta to delta_max: " << delta << " to " << delta_max << std::endl;
   delta = delta_max;
  }
 if ( delta < delta_min ) EXIT_FLAG = 0;
    //std::cout << "#Noise#   delta: " << delta << std::endl;
  /*if (use_approx_gaussian_process){
    gaussian_processes.set_constraint_ball_radius(1.5*delta);
  }*/

  return;
}
//--------------------------------------------------------------------------------

//--------------------------------------------------------------------------------
template<class TSurrogateModel, class TBasisForSurrogateModel>
void NOWPAC<TSurrogateModel, TBasisForSurrogateModel>::add_trial_node ( ) 
{
  if ( evaluations.active_index.size( ) < evaluations.max_nb_nodes  ) {
    evaluations.active_index.push_back ( evaluations.nodes.size()-1 );
  } else if(replace_node_index != -1){
    if(evaluations.active_index[ replace_node_index] != evaluations.best_index) {
    evaluations.active_index[ replace_node_index ] = evaluations.nodes.size()-1;
    }
  } else if(replace_node_index == -1 && verbose >= 3){
    std::cout << "New node does not improve poisedness and is not added surrogate construction." << std::endl;
  } else{
    assert(false);
  }

  if ( stochastic_optimization ){
    if(use_approx_gaussian_process){
          gaussian_processes.set_constraint_ball_center(evaluations.nodes[evaluations.best_index]);
          gaussian_processes.set_constraint_ball_radius(nonlinear_radius_factor*delta);
    } 
    EXIT_FLAG = gaussian_processes.smooth_data ( evaluations );
  }
  return;
}
//--------------------------------------------------------------------------------



//--------------------------------------------------------------------------------
template<class TSurrogateModel, class TBasisForSurrogateModel>
void NOWPAC<TSurrogateModel, TBasisForSurrogateModel>::write_to_file ( ) 
{

    // ouptut current number of black box evaluations
    fprintf(output_file, int_format, evaluations.nodes.size());
    // output the current design
    for(int i = 0; i < dim; ++i) {
        fprintf(output_file, double_format, evaluations.nodes.at(evaluations.best_index).at(i));
    }
    // output the current best value of the objective
    fprintf(output_file, double_format, evaluations.values.at(0).at(evaluations.best_index));
    // output the current trust region radius
    fprintf(output_file, double_format, delta);
    // output the current values of the constraints
    for(int i = 0; i < nb_constraints; ++i) {
        fprintf(output_file, double_format, evaluations.values.at(i+1).at(evaluations.best_index));
    }
    
    // output the last added design
    for(int i = 0; i < dim; ++i) {
        fprintf(output_file, double_format, evaluations.nodes.back().at(i));
    }
    // output the last value of the objective
    fprintf(output_file, double_format, evaluations.values.at(0).back() );

    // output the last values of the constraints
    for(int i = 0; i < nb_constraints; ++i) {
        fprintf(output_file, double_format, evaluations.values.at(i+1).back() );
    }
  
    if(stochastic_optimization){
      // output mean and variance of current value of objective gp
      //double mean = -1;
      //double var = -1;
      //gaussian_processes.evaluate_gaussian_process_at(0, evaluations.nodes.at(evaluations.best_index), mean, var);
      //fprintf(output_file, double_format, mean);
      //fprintf(output_file, double_format, var);

      // output the noise for best value of the objective
      fprintf(output_file, double_format, evaluations.noise.at(0).at(evaluations.best_index));

      // output the noise for best values of the constraints
      for(int i = 0; i < nb_constraints; ++i) {
          fprintf(output_file, double_format, evaluations.noise.at(i+1).at(evaluations.best_index));
      }

      // output the noise for last added value of the objective
      fprintf(output_file, double_format, evaluations.noise.at(0).back() );

      // output the noise for last added values of the constraints
      for(int i = 0; i < nb_constraints; ++i) {
          fprintf(output_file, double_format, evaluations.noise.at(i+1).back() );
      }
      if(use_analytic_smoothing) {
        std::vector<std::vector<double>> analytic_smoothing_quantities;
        analytic_smoothing_quantities = gaussian_processes.getBest_index_analytic_information();
        for(int i = 0; i < 1 + nb_constraints; ++i){
          for(int j = 0; j < analytic_smoothing_quantities[i].size(); ++j){
            fprintf(output_file, double_format, analytic_smoothing_quantities[i][j] );
          }
        }
      }
    }

    fprintf(output_file, "\n");
    fflush(output_file);

}
//--------------------------------------------------------------------------------

//--------------------------------------------------------------------------------
template<class TSurrogateModel, class TBasisForSurrogateModel>
void NOWPAC<TSurrogateModel, TBasisForSurrogateModel>::check_parameter_consistency ( 
  std::vector<double> const &x ) 
{
  if ( nb_constraints == -1 ) {
    std::cout << "Error   : No objective function has been specified." << std::endl;
    EXIT_FLAG = -4;
  }
  if ( max_number_blackbox_evaluations < dim+1 ) {
    std::cout << "Error   : Maximal number of blackbox evaluations to small." << std::endl;
    EXIT_FLAG = -4;
  }
  if ( !max_number_blackbox_evaluations_is_set && delta_min < 0e0 ) {

  }
  if ( !noise_detection && noise_termination && verbose > 0 ) {
    std::cout << "Warning : Noise termination requested, but noise detection is turned off." << std::endl;
    std::cout << "          Turning on noise detection." << std::endl;
    noise_detection = true;
    //EXIT_FLAG = -4;
  } 
  if ( max_number_accepted_steps <= 0 ) {
    std::cout << "Error   : Invalid parameter value for max_nb_accepted_steps (>= 1)." << std::endl;
    EXIT_FLAG = -4;
  }
  if ( delta_min < 0e0 && !max_number_blackbox_evaluations_is_set && 
       !max_number_accepted_steps_is_set ) {
    std::cout << "Error   : No termination criterion is specified." << std::endl;
    EXIT_FLAG = -4;
  }
  if ( delta <= delta_min ) {
    std::cout << "Error   : Initial trust region radius has to be larger than the minimal trust region radius." << std::endl;
    EXIT_FLAG = -4;
  }
  if ( delta > delta_max && delta_max_is_set ) {
    std::cout << "Error   : Initial trust region radius has to be smaller or equal than the maximal trust region radius." << std::endl;
    EXIT_FLAG = -4;
  } else if ( delta > delta_max && !delta_max_is_set ) {
    std::cout << "Warning : Setting maximal trust region radius to " << delta << "." << std::endl;
    delta_max = delta;
  }
  if ( !lower_bound_constraints.empty() ) {
    for ( int i = 0; i < dim; ++i ) {
      if ( x[i] < lower_bound_constraints[i] ) {
        std::cout << "Error   : Initial design violates lower bound constraints." << std::endl;
        EXIT_FLAG = -4;       
        break;
      }
    }
  }
  if ( !upper_bound_constraints.empty() ) {
    for ( int i = 0; i < dim; ++i ) {
      if ( x[i] > upper_bound_constraints[i] ) {
        std::cout << "Error   : Initial design violates upper bound constraints." << std::endl;
        EXIT_FLAG = -4;       
        break;
      }
    }
  }
  if ( gamma <= 0.0 || gamma > 1.0 || gamma > gamma_inc ) {
    std::cout << "Error   : Invalid parameter value for gamma: ]0, gamma_inc]" << std::endl;
    EXIT_FLAG = -4;
  }
  if ( gamma <= 0.0 || gamma >= 1.0 ) {
    std::cout << "Error   : Invalid parameter value for gamma (]0, 1[)" << std::endl;
    EXIT_FLAG = -4;
  }
  if ( gamma_inc <= 1.0 ) {
    std::cout << "Error   : Invalid parameter value for gamma_inc (> 1)." << std::endl;
    EXIT_FLAG = -4;
  }
  if ( omega <= 0.0 || omega >= 1.0 ) {
    std::cout << "Error   : Invalid parameter value for omega (]0, 1[)." << std::endl;
    EXIT_FLAG = -4;
  }
  if ( theta <= 0.0 || theta >= 1.0 ) {
    std::cout << "Error   : Invalid parameter value for theta (]0, 1[)." << std::endl;
    EXIT_FLAG = -4;
  }
  if ( eta_0 <= 0.0 || eta_0 > eta_1 ) {
    std::cout << "Error   : Invalid parameter value for eta_0 (]0, eta_1])." << std::endl;
    EXIT_FLAG = -4;
  }
  if ( eta_1 <= 0.0 || eta_0 > eta_1 || eta_1 >= 1.0) {
    std::cout << "Error   : Invalid parameter value for eta_1 ([eta_0, 1[)." << std::endl;
    EXIT_FLAG = -4;
  }
  if ( eta_2 <= 0.0 || eta_1 > eta_2 || eta_2 >= 1.0) {
    std::cout << "Error   : Invalid parameter value for eta_2 ([eta_1, 1[)." << std::endl;
    EXIT_FLAG = -4;
  }
  if ( mu <= 0.0 ) {
    std::cout << "Error   : Invalid parameter value for mu (> 0)." << std::endl;
    EXIT_FLAG = -4;
  }
  if ( eps_c <= 0.0 ) {
    std::cout << "Error   : Invalid parameter value for eps_c (> 0)." << std::endl;
    EXIT_FLAG = -4;
  }
  if ( nb_constraints > 0 ) {
    for ( int i = 0; i < nb_constraints; ++i ) {
      if ( inner_boundary_path_constants[i] <= 0.0 ) {
        std::cout << "Error   : Invalid parameter value for inner boundary path constant (> 0)." << std::endl;
        EXIT_FLAG = -4;
        break;
      }
    }
  }
  if ( threshold_for_poisedness_constant <= 1.0 ) {
    std::cout << "Error   : Invalid parameter value for geometry_threshold (> 1)." << std::endl;
    EXIT_FLAG = -4;
  } 
  if ( verbose < 0 || verbose > 3 ) {
    std::cout << "Error   : Invalid parameter value for statistics (0, 1, 2, 3)." << std::endl;
    EXIT_FLAG = -4;
  } 
  if ( nb_allowed_noisy_iterations < 0 ) {
    std::cout << "Error   : Invalid parameter value for allowed_noisy_iterations (>= 0)." << std::endl;
    EXIT_FLAG = -4;
  }
  if ( noise_observation_span <= 1 ) {
    std::cout << "Error   : Invalid parameter value for observation_span (>= 2)." << std::endl;
    EXIT_FLAG = -4;
  }
  if ( update_interval_length < 0 ) {
    std::cout << "Error   : Invalid parameter value for GP_update_interval_length (>= 0)." << std::endl;
    EXIT_FLAG = -4;
  }
  if ( !update_at_evaluations.empty() ) {
    for ( int i = 0; i < (int) update_at_evaluations.size(); ++i ) { 
      if ( update_at_evaluations[i] < 0 ) {
        std::cout << "Error   : Invalid parameter value for GP_apaption_steps (>= 0)." << std::endl;
        EXIT_FLAG = -4;
        break;
      }
    }
  }

  return;
}
//--------------------------------------------------------------------------------

//--------------------------------------------------------------------------------
//XXX--------------------------------------------------------------------------------
template<class TSurrogateModel, class TBasisForSurrogateModel>
void NOWPAC<TSurrogateModel, TBasisForSurrogateModel>::output_for_plotting ( const int& evaluation_step, const int& sub_index, std::vector< double > const& best_node )
{

  std::cout << "Writing Output..." << std::endl;
  std::vector<double> x_loc(dim);
  std::vector<double> fvals(nb_constraints+1);
  std::vector<double> fvar(nb_constraints+1);
  int xcoord = 0;
  int ycoord = 1;
  double var = 0;
  double upper_bound = 1;
  double lower_bound = 0;
  std::ofstream outputfile ( "blackbox_data_" + std::to_string(evaluation_step) + "_" + std::to_string(sub_index) + ".dat");
  if (outputfile.is_open() ) {
  	for (int i = 0; i < evaluations.nodes.size(); ++i){
  		for(int j = 0; j < dim; ++j){
  			outputfile <<  evaluations.nodes.at(i).at(j) << ';';
  		}
  		for(int j = 0; j < nb_constraints + 1; ++j){
  			outputfile << evaluations.values.at(j).at(i) << ';';
  		}
  		outputfile << std::endl;
  	}
    outputfile.close();
  }else{
   std::cout << "Unable to open file" << std::endl;
  }

  outputfile.open( "gp_best_point_" + std::to_string(evaluation_step) + "_" + std::to_string(sub_index) + ".dat" );
  if ( outputfile.is_open( ) ) {
        //fvals.at(0) = surrogate_models[0].evaluate( x_loc );
        gaussian_processes.evaluate_gaussian_process_at(0, best_node, fvals.at(0), var);
        outputfile << best_node.at(xcoord) << "; " << best_node.at(ycoord) << "; " << fvals.at(0)<<"; " << var << std::endl;
        for ( int k = 0; k < nb_constraints; ++k) {
            //fvals.at(k+1) = surrogate_models[k+1].evaluate( x_loc );
            gaussian_processes.evaluate_gaussian_process_at(k + 1, best_node, fvals.at(k + 1), var);
            outputfile << best_node.at(xcoord) << "; " << best_node.at(ycoord) << "; " << fvals.at(k + 1) << "; " << var << std::endl;
        }
        outputfile.close();
  } else {std::cout << "Unable to open file." << std::endl;}

		outputfile.open( "gp_points_" + std::to_string(evaluation_step) + "_" + std::to_string(sub_index) + ".dat" );
		std::vector<std::vector<double>> gp_nodes;
		if ( outputfile.is_open( ) ) {
			gp_nodes = gaussian_processes.get_nodes_at(0);
			for ( int i = 0; i < gp_nodes.size(); ++i) {
					for (int j = 0; j < gp_nodes[i].size(); ++j){
							outputfile << gp_nodes[i][j] << " ";
					}
					outputfile << std::endl;
			}
			outputfile.close();
		} else {std::cout << "Unable to open file." << std::endl;}

  outputfile.open ( "gp_data_" + std::to_string(evaluation_step) + "_" + std::to_string(sub_index) + ".dat" );
  if ( outputfile.is_open( ) ) {
    for ( int i = 0; i < dim; ++i)
      x_loc.at(i) = 0e0;
    for (double i = 0.0; i <= 1.0; i+=0.05) {
      x_loc.at(xcoord) = i;
      for (double j = 0.0; j <= 1.0; j+=0.05) {
        x_loc.at(ycoord) = j;
        //fvals.at(0) = surrogate_models[0].evaluate( x_loc );
        gaussian_processes.evaluate_gaussian_process_at(0, x_loc, fvals.at(0), fvar.at(0));
        outputfile << x_loc.at(xcoord) << "; " << x_loc.at(ycoord) << "; " << fvals.at(0)<<"; " << fvar.at(0) << ";";
        for ( int k = 0; k < nb_constraints; ++k) {
          //fvals.at(k+1) = surrogate_models[k+1].evaluate( x_loc );
          gaussian_processes.evaluate_gaussian_process_at(k+1, x_loc, fvals.at(k+1), fvar.at(0));
          outputfile << fvals.at(k+1) << "; ";
        }
        outputfile << std::endl;
      }
    }
    outputfile.close( );
  } else {std::cout << "Unable to open file." << std::endl;}

    outputfile.open( "active_nodes_" + std::to_string(evaluation_step) + "_" + std::to_string(sub_index) + ".dat" );
		if ( outputfile.is_open( ) ) {
					//fvals.at(0) = surrogate_models[0].evaluate( x_loc );
					for(int i = 0; i < evaluations.active_index.size(); ++i){
						for(int j = 0; j < evaluations.nodes[evaluations.active_index[i]].size(); ++j){
							outputfile << evaluations.nodes[evaluations.active_index[i]][j] << ";";
						}
						outputfile << std::endl;
					}
					outputfile.close();
		} else {std::cout << "Unable to open file." << std::endl;}

    if(use_approx_gaussian_process) {
        outputfile.open(
                "gp_induced_points_" + std::to_string(evaluation_step) + "_" + std::to_string(sub_index) + ".dat");
        if (outputfile.is_open()) {
            //fvals.at(0) = surrogate_models[0].evaluate( x_loc );
            gaussian_processes.evaluate_gaussian_process_at(0, best_node, fvals.at(0), var); //reset u_indices and augmented_u
            std::vector< std::vector<double> > u_nodes;
            u_nodes.clear();
            gaussian_processes.get_induced_nodes_at(0, u_nodes);
            std::cout << "U_matrix: " << std::endl;
            VectorOperations::print_matrix(u_nodes);
            for (int i = 0; i < u_nodes.size(); ++i) {
                for (int j = 0; j < u_nodes[i].size(); ++j){
                    outputfile << u_nodes[i][j] << " ";
                }
                outputfile << std::endl;
            }
            outputfile.close();
        } else {std::cout << "Unable to open file." << std::endl;}
    }
  
  /*
  outputfile.open ( "surrogate_data_" + std::to_string(evaluation_step) + "_" + std::to_string(sub_index) + ".dat" );
  if ( outputfile.is_open( ) ) {
      for (int i = 0; i < dim; ++i)
          x_loc.at(i) = 0e0;
      for (double i = -1.0; i <= 1.0; i += 0.01) {
          //x_loc.at(xcoord) =  i;
          x_loc.at(xcoord) = ((i + 1) / (2) * (upper_bound - lower_bound) + lower_bound);
          for (double j = -1.0; j < 1.0; j += 0.01) {
              //x_loc.at(ycoord) = j;
              x_loc.at(ycoord) = ((j + 1) / (2) * (upper_bound - lower_bound) + lower_bound);
              fvals.at(0) = surrogate_models[0].evaluate( x_loc );
              outputfile << x_loc.at(xcoord) << "; " << x_loc.at(ycoord) << "; " << fvals.at(0) << "; ";
              for (int k = 0; k < nb_constraints; ++k) {
                  fvals.at(k+1) = surrogate_models[k+1].evaluate( x_loc );
                  outputfile << fvals.at(k + 1) << "; ";
              }
              outputfile << std::endl;
          }
      }
      outputfile.close();
  }else std::cout << "Unable to open file." << std::endl;
   */


  /*outputfile.open ( "data_" + std::to_string(evaluation_step) + ".dat" );
  if ( outputfile.is_open( ) ) {
    std::vector< std::vector<double> > outputnodes;
    outputfile << delta << "; " << evaluations.active_index.size() << "; ";
    for ( int i = 0; i < dim-2; ++i)     
      outputfile << "0 ;";
    outputfile << std::endl;
    outputnodes = evaluations.get_scaled_active_nodes( delta );
    for ( int i = 0; i < evaluations.active_index.size(); ++i) {
      for ( int j = 0; j < dim; ++j )
        outputfile << outputnodes[i][j] << "; ";
      outputfile << std::endl;
    }
    for ( int j = 0; j < dim; ++j )
      outputfile << evaluations.nodes[evaluations.best_index][j] << "; ";
    outputfile << std::endl;    
    for ( int j = 0; j < dim; ++j )
    outputfile << x_trial[j] << "; "; 
    outputfile << std::endl;    
    for ( int j = 0; j < nb_constraints; ++j ) {
      if (acceptance_ratio >= eta_0)
        outputfile << "1; " << inner_boundary_path_constants[j] << "; " ;
      else
        outputfile << "0; " << inner_boundary_path_constants[j] << "; ";
      for ( int i = 0; i < dim-2; ++i)     
        outputfile << "0 ;";
      outputfile << std::endl;
    }
    outputfile << xcoord << "; " << ycoord << "; ";
    for ( int i = 0; i < dim-2; ++i)     
      outputfile << "0 ;";
    outputfile << std::endl;
    outputfile.close( );
  } else std::cout << "Unable to open file." << std::endl;*/

  //std::cout << "Press Enter to Continue";
  //std::cin.ignore(std::numeric_limits<std::streamsize>::max(),'\n');
  //system("read -n1 -r -p \"Press any key to continue...\"");
    std::cout << "...Done!" << std::endl;
  return;
}
//XXX--------------------------------------------------------------------------------
//--------------------------------------------------------------------------------

//--------------------------------------------------------------------------------
template<class TSurrogateModel, class TBasisForSurrogateModel>
int NOWPAC<TSurrogateModel, TBasisForSurrogateModel>::optimize ( 
  std::vector<double> &x, std::vector<double> &val, BlackBoxData &bb_data ) 
{

  if ( evaluations.nodes.size() > 0 ) { 
    evaluations = bb_data;
    EXIT_FLAG = NOEXIT;
    delta = evaluations.get_scaling( );
  }
//  assert ( evaluations.values[0].size() == evaluations.nodes.size() );
//  if ( stochastic_optimization ) 
//    assert ( evaluations.noise[0].size() == evaluations.nodes.size() );
  EXIT_FLAG = optimize(x, val);  
  bb_data = evaluations;
  return EXIT_FLAG;
}
//--------------------------------------------------------------------------------

//--------------------------------------------------------------------------------
template<class TSurrogateModel, class TBasisForSurrogateModel>
int NOWPAC<TSurrogateModel, TBasisForSurrogateModel>::optimize ( 
  std::vector<double> &x, std::vector<double> &val ) 
{

  check_parameter_consistency( x );  
  if ( EXIT_FLAG != NOEXIT ) { 
   std::cout << std::endl;
   return EXIT_FLAG;
  }

  std::cout.setf( std::ios::fixed, std:: ios::floatfield );
  std::cout << std::setprecision(8);

  std::random_device rand_dev;
  int random_seed = (fixed_seed != -1) ? fixed_seed : rand_dev(); //25041981;//
  std::mt19937 rand_generator(random_seed);
  std::normal_distribution<double> norm_dis(0e0,2e-1);

  if ( evaluations.nodes.size() == 0 ) {
    for (int i = 0; i < nb_constraints+1; ++i )
      surrogate_models.push_back ( surrogate_model_prototype );

    if ( noise_detection )
      this->initialize_noise_detection( noise_observation_span, nb_allowed_noisy_iterations);

    surrogate_nodes.reset(new ImprovePoisedness ( surrogate_basis, 
                                                  threshold_for_poisedness_constant,
                                                  evaluations.max_nb_nodes, 
                                                  delta, 
                                                  verbose, upper_bound_constraints, lower_bound_constraints, 
                                                  use_hard_box_constraints) );
    surrogate_optimization.reset( new SubproblemOptimization<TSurrogateModel> (
      surrogate_models, delta, inner_boundary_path_constants) );
    if ( !upper_bound_constraints.empty() ) 
      surrogate_optimization->set_upper_bounds ( upper_bound_constraints );
    if ( !lower_bound_constraints.empty() ) 
      surrogate_optimization->set_lower_bounds ( lower_bound_constraints );

    if ( stochastic_optimization )
      gaussian_processes.initialize ( dim, nb_constraints + 1, delta, blackbox,
                                      update_at_evaluations, update_interval_length, gaussian_process_type, NOEXIT, use_analytic_smoothing );

    if ( verbose >= 2 ) { std::cout << "Initial evaluation of black box functions" << std::endl; }
    // initial evaluations 
    evaluations.best_index = 0;
    blackbox_evaluator( x, true );  
    /*if(use_approx_gaussian_process){
          gaussian_processes.set_constraint_ball_center(evaluations.nodes[evaluations.best_index]);
    } */
    if ( EXIT_FLAG != NOEXIT ){
      std::cout << "ERROR   : Black box returned invalid value" << std::endl << std::fflush; 
      return EXIT_FLAG;
    }
    if ( !last_point_is_feasible ( ) ) {
      if ( verbose >= 1 )
        std::cout << "Initial point is not feasibile" << std::endl << std::flush;
//      EXIT_FLAG = -3;
//      return EXIT_FLAG;
    }
    for (int i = 0; i < dim; ++i ) {
      x_sample = x;
      if(!upper_bound_constraints.empty()){ //Is there an upper bound
    	  if(x_sample.at(i)+delta < upper_bound_constraints.at(i) ){ //Is new surrogate point in domain
    		  x_sample.at(i) += delta; //if yes, create that point
        }else if(!lower_bound_constraints.empty()){//if no, check if there are lower bounds
      	  if(x_sample.at(i) - delta > lower_bound_constraints.at(i) ){//if lower bounds, check if -delta is in bounds
      		  x_sample.at(i) -= delta; //if yes, create that point
          }else{//neither in upper nor lower bounds
            x_sample.at(i) = (upper_bound_constraints[i] - x_sample[i]) > (x_sample[i] - lower_bound_constraints[i]) ?
                            upper_bound_constraints[i] : lower_bound_constraints[i]; //take point with maximal distance to x_trial
          }
        }else{//if no lower bounds, but upper, take -delta as new point
          x_sample.at(i) -= delta;
        }
      }else{//no upper bounds, just take new point
    	  x_sample.at(i) += delta;
      }
      blackbox_evaluator( x_sample, true );
      if ( EXIT_FLAG != NOEXIT ){
        std::cout << "ERROR   : Black box returned invalid value" << std::endl << std::flush; 
        return EXIT_FLAG;
      }
    }

  } else {
    if ( evaluations.values[0].size() == 0 )
    if ( verbose >= 2 ) { std::cout << "Evaluating initial nodes .. " << std::flush; }
    blackbox_evaluator ( ); 
    if ( EXIT_FLAG != NOEXIT ){
      std::cout << "ERROR   : Black box returned invalid value" << std::endl << std::flush; 
      return EXIT_FLAG;
    }
    if ( verbose >= 2 ) { std::cout << "done" << std::endl << std::flush; }
  }

   
/*
  std::vector<double> x_loc(2);
  std::vector<double> fvals(3);
  std::ofstream outputfile ( "surrogate_data_o.dat" );
  if ( outputfile.is_open( ) ) {
    for (double i = -1.0; i <= 2.0; i+=0.1) {
      x_loc.at(0) = i;
      for (double j = -1.0; j < 2.0; j+=0.1) {
        x_loc.at(1) = j;
        fvals.at(0) = surrogate_models[0].evaluate( evaluations.transform(x_loc) );
        fvals.at(1) = surrogate_models[1].evaluate( evaluations.transform(x_loc) );
        fvals.at(2) = surrogate_models[2].evaluate( evaluations.transform(x_loc) );
        outputfile << x_loc.at(0) << "; " << x_loc.at(1) << "; " << fvals.at(0)<< "; " << 
                     fvals.at(1)<< "; " << fvals.at(2) << std::endl;
      }
    }
    outputfile.close( );
  } else std::cout << "Unable to open file." << std::endl;

  return 0;
*/

  //Check if initial point is infeasible
  bool initial_point_is_infeasible = false; 
  for (int i = 0; i < nb_constraints; ++i){
	if(evaluations.values[i+1].at(evaluations.best_index) > 0.){
		initial_point_is_infeasible = true;
		if (verbose == 3) {std::cout << "Initial point is not feasible ..." << std::endl;};
		break;
	}
  }
  double max_constraint_violation, cur_constraint_violation, tmp_constraint_violation;
  max_constraint_violation = std::numeric_limits<double>::infinity();
  if( initial_point_is_infeasible ){ //If infeasible we look if one of the sample points is feasible
	bool cur_point_is_feasible = true;
	bool found_feasible_point = true;
	for(int h = 0; h < evaluations.nodes.size(); ++h){
		for (int i = 0; i < nb_constraints; ++i){
			if(evaluations.values[i+1].at(h) > 0.){
				cur_point_is_feasible = false;
				break;
			}
		}
		if(cur_point_is_feasible){
			found_feasible_point = true;
			if(evaluations.values[0].at(h) < evaluations.values[0].at(evaluations.best_index)){
				evaluations.best_index = h;
			}
		}
		cur_point_is_feasible = true;
 	}	
	if(!found_feasible_point){ //If we did not find a feasible point we look for the point with the lowest constraint violation
		cur_constraint_violation = std::numeric_limits<double>::infinity();
		for(int h = 0; h < evaluations.nodes.size(); ++h){
			tmp_constraint_violation = 0.0;
			for(int i = 0; i < nb_constraints; ++i){
				tmp_constraint_violation += evaluations.values[i+1].at(h)*evaluations.values[i+1].at(h);
			}
			if(tmp_constraint_violation < cur_constraint_violation){
				cur_constraint_violation = tmp_constraint_violation;
				evaluations.best_index = h;
			} 
		}
	}
	if( evaluations.best_index != 0 ){ //We found a better point, resample now
		for (int i = 0; i < dim; ++i ) {
      			x_sample = evaluations.nodes[ evaluations.best_index ];
      			if(!upper_bound_constraints.empty()){ //Is there an upper bound
    	  			if(x_sample.at(i)+delta < upper_bound_constraints.at(i) ){ //Is new surrogate point in domain
    		  			x_sample.at(i) += delta; //if yes, create that point
        			}else if(!lower_bound_constraints.empty()){//if no, check if there are lower bounds
      	  				if(x_sample.at(i) - delta > lower_bound_constraints.at(i) ){//if lower bounds, check if -delta is in bounds
      		  				x_sample.at(i) -= delta; //if yes, create that point
          				}else{//neither in upper nor lower bounds
            					x_sample.at(i) = (upper_bound_constraints[i] - x_sample[i]) > (x_sample[i] - lower_bound_constraints[i]) ?
                	            			upper_bound_constraints[i] : lower_bound_constraints[i]; //take point with maximal distance to x_trial
        	  			}
	        		}else{//if no lower bounds, but upper, take -delta as new point
          				x_sample.at(i) -= delta;
       			 	}
	      		}else{//no upper bounds, just take new point
    	  			x_sample.at(i) += delta;
      			}
	      		blackbox_evaluator( x_sample, true );
      			if ( EXIT_FLAG != NOEXIT ){
        		std::cout << "ERROR   : Black box returned invalid value" << std::endl << std::fflush; 
	        	return EXIT_FLAG;
      			}
    		}
	}
  }

  if ( verbose == 3 ) { std::cout << "Building initial models ... "; }
  update_surrogate_models( );
  if ( verbose == 3 ) { std::cout << "done" << std::endl; }
	

  x_trial = evaluations.nodes[ evaluations.best_index ];
  if ( verbose == 3 ) { std::cout << "Value of initial criticality measure : "; }
  criticality_value = surrogate_optimization->compute_criticality_measure( x_trial );

  if ( verbose == 3 ) { std::cout << criticality_value << std::endl;; }


  do {

    /*----------------------------------------------------------------------------------*/
    /*- STEP 1 - STEP 1 - STEP 1 - STEP 1 - STEP 1 - STEP 1 - STEP 1 - STEP 1 - STEP 1 -*/
    /*----------------------------------------------------------------------------------*/
    //if criticality measure is small, check if model improvement is necessary
    if ( best_point_is_feasible( ) ) {
      if ( criticality_value <= eps_c ) {
        if ( verbose == 3 && delta > mu * criticality_value ) {
          std::cout << " Criticality measure below threshold" << std::endl;
          std::cout << " -----------------------------------" << std::endl;
        }
        while ( delta > mu * criticality_value ) {
          //output_for_plotting( number_accepted_steps) ;
          update_trustregion( omega );
          update_surrogate_models( ); //xxx
          if ( EXIT_FLAG != NOEXIT ) break;
          //if (verbose == 3) {
          //  std::cout << "#Noise# Adjusting trust-region radius to " << delta << std::endl;
          //}
          surrogate_nodes->improve_poisedness ( evaluations.best_index, evaluations );
          if ( stochastic_optimization ) {
            for (int i = 0; i < dim; ++i ){
              if(use_hard_box_constraints){
                out_of_bounds = true;
                do{
                  x_sample.at(i) = evaluations.nodes[ evaluations.best_index ].at( i ) +
                           delta * norm_dis( rand_generator ) * 1e0;
                  if(x_sample[i] >= lower_bound_constraints[i] && x_sample[i] <= upper_bound_constraints[i]){
                    out_of_bounds = false;
                  }
                }while(out_of_bounds);
              }else{
                x_sample.at(i) = evaluations.nodes[ evaluations.best_index ].at( i ) +
                           delta * norm_dis( rand_generator ) * 1e0;
              }
            }
            evaluations.nodes.push_back( x_sample );
          }
          blackbox_evaluator( );
          if ( EXIT_FLAG != NOEXIT ) break;
          update_surrogate_models( );
          if ( noise_detection ) {
            if ( this->detect_noise( ) && noise_termination ) EXIT_FLAG = -2; 
          }
          x_trial = evaluations.nodes[ evaluations.best_index ];

          criticality_value = surrogate_optimization->compute_criticality_measure( x_trial );

        // TODO: check if while should be exited on infeasible points...
        // -> it works fine without exiting.
        // -> could be exited without breaking stochastic algorithm

          //output_for_plotting( number_accepted_steps) ;
          if (verbose == 3) {
            std::cout << " Value of criticality measure is " << criticality_value << std::endl;
            std::cout << " -----------------------------------" << std::endl << std::flush;
          }
        }
      }

    }
    /*----------------------------------------------------------------------------------*/
    /*- STEP 2 - STEP 2 - STEP 2 - STEP 2 - STEP 2 - STEP 2 - STEP 2 - STEP 2 - STEP 2 -*/
    /*----------------------------------------------------------------------------------*/
    while (EXIT_FLAG == NOEXIT)  {
      if ( verbose == 3 ) { 
        std::cout << "Computing trial point ... ";
        std::cout << std::flush;
      }

      x_trial = evaluations.nodes[ evaluations.best_index ];

      trial_model_value = surrogate_optimization->compute_trial_point ( x_trial );
 
      if ( verbose == 3 ) { std::cout << "done" << std::endl << std::flush; }

      /*-------------------------------------------------------------------------*/
      /*- STEP 3 - STEP 3 - STEP 3 - STEP 3 - STEP 3 - STEP 3 - STEP 3 - STEP 3 -*/
      /*-------------------------------------------------------------------------*/
      if ( verbose == 3 ) { std::cout << "Checking feasibility .... " << std::flush; }
      blackbox_evaluator( x_trial, false );   
      if ( EXIT_FLAG == NOEXIT ) {
        if ( verbose == 3 ) { std::cout << "done" << std::endl; }
      } else {
        if ( verbose == 3 && EXIT_FLAG == 1) { 
          std::cout << "canceled" << std::endl; 
          std::cout << std::endl;
          std::cout << "Maximum number of black box evaluations reached" << std::endl;
        } else if ( EXIT_FLAG == -5 ) {
          if ( verbose == 3 ) std::cout << "canceled" << std::endl; 
          std::cout << std::endl;
          std::cout << "ERROR   : Black box returned invalid value" << std::endl << std::flush; 
        }
        break;
      }
      if ( !last_point_is_feasible( ) ) {
        if ( verbose == 3 ) { 
          std::cout << "#FEASVIOLATION Step rejected since feasibility violated" << std::endl << std::flush;
          for (int i = 0; i < nb_constraints; ++i){
            std::cout << " Constraint " << i+1 
             << ": [Best: " << evaluations.values[i+1][evaluations.best_index]
             << ", Back: " << evaluations.values[i+1].back() << "]"<< std::endl << std::flush;
          }
        }

        tmp_dbl = 1e0;
        for ( int i = 0; i < evaluations.active_index.size(); ++i ) {
           if ( this->diff_norm ( evaluations.nodes.at( evaluations.active_index.at(i) ),
                                  x_trial ) < 1e-3 * delta ) {
             tmp_dbl = -1e0;
             break;
           }
        }
        if ( tmp_dbl > 0e0 ) {
          replace_node_index = surrogate_nodes->replace_node( evaluations.best_index, 
                                                             evaluations, x_trial );
          add_trial_node( );
          if ( EXIT_FLAG != NOEXIT ) break;
        }
        //output_for_plotting( number_accepted_steps ) ;
        update_trustregion( theta );
        if ( verbose == 3 ) {
          std::cout << " #FEASVIOLATION: Updating trust region: " << delta << std::endl << std::flush;
        }
        update_surrogate_models( );

        
        if ( EXIT_FLAG != NOEXIT ) break;
        surrogate_nodes->improve_poisedness (evaluations.best_index, evaluations );
        if ( stochastic_optimization ) {
          for (int i = 0; i < dim; ++i ){
            if(use_hard_box_constraints){
              out_of_bounds = true;
              do{
                x_sample.at(i) = evaluations.nodes[ evaluations.best_index ].at( i ) +
                         delta * norm_dis( rand_generator ) * 1e0;
                if(x_sample[i] >= lower_bound_constraints[i] && x_sample[i] <= upper_bound_constraints[i]){
                  out_of_bounds = false;
                }
              }while(out_of_bounds);
            }else{
              x_sample.at(i) = evaluations.nodes[ evaluations.best_index ].at( i ) +
                         delta * norm_dis( rand_generator ) * 1e0;
            }
          }

          evaluations.nodes.push_back( x_sample );
        }
        blackbox_evaluator( );
        if ( EXIT_FLAG != NOEXIT ) break;
        update_surrogate_models( );

        if ( noise_detection ) {
          if ( this->detect_noise( ) && noise_termination ) EXIT_FLAG = -2; 
        }

//        //output_for_plotting ( number_accepted_steps );
      } else {
        if ( verbose == 3 ) { std::cout << "Found feasible trial step" << std::endl << std::flush; }
        break;
      }
    }
 
    /*----------------------------------------------------------------------------------*/
    /*- STEP 4 - STEP 4 - STEP 4 - STEP 4 - STEP 4 - STEP 4 - STEP 4 - STEP 4 - STEP 4 -*/
    /*----------------------------------------------------------------------------------*/
    if ( EXIT_FLAG == NOEXIT ) { 
      acceptance_ratio = compute_acceptance_ratio ( );

      //output_for_plotting( number_accepted_steps );

      tmp_dbl = this->diff_norm( x_trial, evaluations.nodes[ evaluations.best_index ] ) / delta;
      stepsize[0] = ( tmp_dbl + stepsize[1])/ 2e0;
      stepsize[1] = tmp_dbl;
      for (int i = 0; i < nb_constraints; ++i) 
        inner_boundary_path_constants.at(i) = pow(stepsize[0], 2e0) * 
                                              max_inner_boundary_path_constants.at(i);

      if ( verbose >= 2 ) std::cout << "*****************************************" << std::endl; 

      if ( acceptance_ratio >= eta_2 && acceptance_ratio < 2e0 ) {
        if ( verbose >= 2 ) { std::cout << "Step successful and increase." << std::endl << std::flush; }
        update_trustregion( gamma_inc );
      } 
      fflush(stdout);
      if ( (acceptance_ratio >= eta_1 && acceptance_ratio < eta_2) || acceptance_ratio >= 2e0 ) {
        if ( verbose >= 2 ) { std::cout << "Step acceptable and keep." << std::endl << std::flush; }
      }
      if ( acceptance_ratio >= eta_0 && acceptance_ratio < eta_1 ) {
        if ( verbose >= 2 ) { std::cout << "Step acceptable but shrink." << std::endl << std::flush; }
        update_trustregion( gamma );
      }
      if ( acceptance_ratio < eta_0) {
        if ( verbose >= 2 ) { std::cout << "Step rejected and shrink." << std::endl << std::flush; }
        //Shrinking step done in if statement below.
      }

      if ( acceptance_ratio >= eta_0 ) {
        replace_node_index = surrogate_nodes->replace_node( -1, 
                                                           evaluations, x_trial );
        if ( EXIT_FLAG != NOEXIT ) break;
        evaluations.best_index = evaluations.nodes.size()-1;
        add_trial_node( );
        /*if(use_approx_gaussian_process){
          gaussian_processes.set_constraint_ball_center(evaluations.nodes[evaluations.best_index]);
        }*/
        update_surrogate_models( );
        number_accepted_steps++;
        if ( number_accepted_steps >= max_number_accepted_steps ) EXIT_FLAG = -6;
      }
      if ( acceptance_ratio < eta_0 ) {
        tmp_dbl = 1e0;
        for ( int i = 0; i < evaluations.active_index.size(); ++i ) {
           if ( this->diff_norm ( evaluations.nodes.at( evaluations.active_index.at(i) ),
                                  evaluations.nodes.at( evaluations.nodes.size()-1 ) ) < 1e-2 * delta ) {
             tmp_dbl = -1e0;
             break;
          }
        } 
        if ( tmp_dbl > 0e0 ) { // XXX --- XXX
          replace_node_index = surrogate_nodes->replace_node( evaluations.best_index, 
                                                             evaluations, x_trial );
          //assert ( evaluations.active_index[replace_node_index] != evaluations.best_index);
          add_trial_node( );
          if ( EXIT_FLAG != NOEXIT ) break;
          update_surrogate_models( );
        }
        if ( stochastic_optimization ) {
          for (int i = 0; i < dim; ++i ){
            if(use_hard_box_constraints){
              out_of_bounds = true;
              do{
                x_sample.at(i) = evaluations.nodes[ evaluations.best_index ].at( i ) +
                         delta * norm_dis( rand_generator ) * 1e0;
                if(x_sample[i] >= lower_bound_constraints[i] && x_sample[i] <= upper_bound_constraints[i]){
                  out_of_bounds = false;
                }
              }while(out_of_bounds);
            }else{
              x_sample.at(i) = evaluations.nodes[ evaluations.best_index ].at( i ) +
                         delta * norm_dis( rand_generator ) * 1e0;
            }
          }
          evaluations.nodes.push_back( x_sample );
        }
        blackbox_evaluator( );
        if ( EXIT_FLAG == NOEXIT ) {
          update_trustregion( gamma );
          update_surrogate_models( );
          surrogate_nodes->improve_poisedness( evaluations.best_index, evaluations );
          blackbox_evaluator( );
          if ( EXIT_FLAG != NOEXIT ) {
            std::cout << "*****************************************" << std::endl << std::flush;
            break;
          }
          update_surrogate_models( );
          if (noise_detection ) {
            if ( this->detect_noise( ) && noise_termination ) EXIT_FLAG = -2; 
          }
        } else {
          std::cout << "*****************************************" << std::endl << std::flush;
          if ( EXIT_FLAG == -5 ) {
            std::cout << std::endl;
            std::cout << "ERROR   : Black box returned invalid value" << std::endl << std::fflush; 
          }
          break;
        }
      }

      /**Check if all active indices are actually in the trust region**/
      /*
      int initial_best_index = evaluations.best_index;
      int cur_active_index;
      std::vector<double> cur_node;
      bool feasibility_check = false;
      int diff_nodes = evaluations.active_index.size() - (dim + 1); //How many more nodes than (dim + 1). This many nodes we can remove
      std::vector<double> active_node_distance(evaluations.active_index.size());
      std::cout << "Active Indices: " << std::endl;
      for(auto val: evaluations.active_index){
        std::cout << val << ", ";
      }
      std::cout << std::endl;
      for(int i=0; i < evaluations.active_index.size(); ++i){
        cur_active_index = evaluations.active_index[i];
        cur_node = evaluations.nodes[cur_active_index];
        active_node_distance[i] = VectorOperations::diff_norm(cur_node, evaluations.nodes[evaluations.best_index]);

        if(active_node_distance[i] > delta){
          std::cout << "Index  " << cur_active_index << " will be deleted with distance " << active_node_distance[i] << " > " << delta << std::endl;
        }
      }
      std::vector<int> idx(active_node_distance.size());
      std::iota(idx.begin(), idx.end(), 0);
      std::sort(idx.begin(), idx.end(),
           [&active_node_distance](int i1, int i2) {return active_node_distance[i1] > active_node_distance[i2];});
      for(auto i: idx){
        if(active_node_distance[i] <= delta){
          break; //All leftover points are in the trust region, we are fine
        }
        if(diff_nodes > 0){//We can still remove some points
          evaluations.active_index[i] = -1; //Mark for deletion
          diff_nodes--;
        }
      }
      std::cout << "Marked Indices: " << std::endl;
      for(auto val: evaluations.active_index){
        std::cout << val << ", ";
      }
      std::cout << std::endl;
      evaluations.active_index.erase(std::remove(evaluations.active_index.begin(), evaluations.active_index.end(), -1), evaluations.active_index.end()); //Delete all elements equal -1
      std::cout << "Updated Indices: " << std::endl;
      for(auto val: evaluations.active_index){
        std::cout << val << ", ";
      }
      std::cout << std::endl;
      update_surrogate_models( );
      //surrogate_nodes->improve_poisedness( evaluations.best_index, evaluations );
      */
      /***/

      x_trial = evaluations.nodes[ evaluations.best_index ];

      criticality_value = surrogate_optimization->compute_criticality_measure( x_trial );
    
      if ( verbose >= 2 ) {
        std::cout << "*****************************************" << std::endl;
        if ( verbose == 3 ) {
          std::cout << "  Acceptance ratio      :  " << std::setprecision(8) << acceptance_ratio << std::endl;
          std::cout << "  Criticality measure   :  " << std::setprecision(8) << criticality_value << std::endl;
        }
        std::cout << "  Trust-region radius   :  " << std::setprecision(8) << delta << std::endl;
        std::cout << "  Number of evaluations :  " << std::setprecision(8) << evaluations.nodes.size() << std::endl;
        std::cout << "-----------------------------------------" << std::endl;
        std::cout << "  Current value         :  " << std::setprecision(8) << evaluations.values[0].at( evaluations.best_index ) << std::endl;
        if ( dim > 1) {
          std::cout << "  Current point         : [" << std::setprecision(8) << evaluations.nodes[ evaluations.best_index ].at(0) << std::endl;
          for (int i = 1; i < dim-1; i++)
            std::cout << "                           " << std::setprecision(8) << evaluations.nodes[ evaluations.best_index ].at(i) << std::endl;
          std::cout << "                           " << std::setprecision(8) << evaluations.nodes[ evaluations.best_index ].at(dim-1) << "]"<< std::endl;
        } else
            std::cout << "  Current point         : [" << std::setprecision(8) << evaluations.nodes[ evaluations.best_index ].at(0) << "]"<< std::endl;
          if ( nb_constraints > 0) {
            switch(nb_constraints){
                case 1:
                  std::cout << "  Current constraints   : ["<< std::setprecision(8) << evaluations.values[1].at( evaluations.best_index ) << "]" << std::endl;
                  break;
                case 2:
                  std::cout << "  Current constraints   : ["<< std::setprecision(8) << evaluations.values[1].at( evaluations.best_index ) << std::endl;
                  std::cout << "                           "<< std::setprecision(8) << evaluations.values[2].at( evaluations.best_index ) << "]"<< std::endl;
                  break;
                default:
                  std::cout << "  Current constraints   : ["<< std::setprecision(8) << evaluations.values[1].at( evaluations.best_index ) << std::endl;
                  for (int i = 2; i < nb_constraints - 1; ++i )
                    std::cout << "                           " << std::setprecision(8) << evaluations.values[i].at( evaluations.best_index ) << std::endl;
                  std::cout << "                           " << std::setprecision(8) << evaluations.values[nb_constraints].at( evaluations.best_index ) << "]"<< std::endl;

            }
        }if(stochastic_optimization) {
          std::cout << "  Current Noise        : [" << std::setprecision(8) << evaluations.noise[0].at(evaluations.best_index) << std::endl;
          for(int i = 1; i < evaluations.noise.size() - 1; ++i){
            std::cout << "                           " << std::setprecision(8) << evaluations.noise[i].at(evaluations.best_index) << std::endl;
          }
          std::cout << "                           " << std::setprecision(8) << evaluations.noise[evaluations.noise.size() - 1].at(evaluations.best_index) << "]" << std::endl;
        }
        std::cout << "*****************************************" << std::endl << std::endl << std::flush;
      }

    }

  } while ( EXIT_FLAG == NOEXIT );

  val.resize(1 + nb_constraints);
  for(int i = 0; i < 1 + nb_constraints; ++i){
    val[i] = evaluations.values[i].at(evaluations.best_index);
  }
  x   = evaluations.nodes[ evaluations.best_index ];

  if ( verbose >= 1 ) {
    std::cout << std::endl << "RESULTS OF OPTIMIZATION:" << std::endl;
    std::cout << "------------------------" << std::endl;
    std::cout << "*********************************************" << std::endl;
    if ( dim > 1) {
      std::cout << "  Best point            : [" << std::setprecision(12) << evaluations.nodes[ evaluations.best_index ].at(0) << std::endl;
      for (int i = 1; i < dim-1; i++)
        std::cout << "                           " << std::setprecision(12) << evaluations.nodes[ evaluations.best_index ].at(i) << std::endl;
      std::cout << "                           " << std::setprecision(12) << evaluations.nodes[ evaluations.best_index ].at(dim-1) << "]"<< std::endl;
    } else
      std::cout << "  Best point            : [" << std::setprecision(12) << evaluations.nodes[ evaluations.best_index ].at(0) << "]"<< std::endl;
    std::cout << "---------------------------------------------" << std::endl;
    std::cout << "  Best value            :  " << std::setprecision(12) << evaluations.values[0].at( evaluations.best_index ) << std::endl;
    std::cout << "---------------------------------------------" << std::endl;
    if ( nb_constraints > 0) {
      switch(nb_constraints){
        case 1:
          std::cout << "  Current constraints   : ["<< std::setprecision(8) << evaluations.values[1].at( evaluations.best_index ) << "]" << std::endl;
          break;
        case 2:
          std::cout << "  Current constraints   : ["<< std::setprecision(8) << evaluations.values[1].at( evaluations.best_index ) << std::endl;
          std::cout << "                           "<< std::setprecision(8) << evaluations.values[2].at( evaluations.best_index ) << "]"<< std::endl;
          break;
        default:
          std::cout << "  Current constraints   : ["<< std::setprecision(8) << evaluations.values[1].at( evaluations.best_index ) << std::endl;
          for (int i = 2; i < nb_constraints - 1; ++i )
            std::cout << "                           " << std::setprecision(8) << evaluations.values[i].at( evaluations.best_index ) << std::endl;
          std::cout << "                           " << std::setprecision(8) << evaluations.values[nb_constraints].at( evaluations.best_index ) << "]"<< std::endl;
      }
    }
    if(stochastic_optimization) {
      std::cout << "---------------------------------------------" << std::endl;
      std::cout << "  Noise                 : [" << std::setprecision(12) << evaluations.noise[0].at(evaluations.best_index) << std::endl;
      for(int i = 1; i < evaluations.noise.size() - 1; ++i){
        std::cout << "                           " << std::setprecision(12) << evaluations.noise[i].at(evaluations.best_index) << std::endl;
      }
      std::cout << "                           " << std::setprecision(12) << evaluations.noise[evaluations.noise.size() - 1].at(evaluations.best_index) << "]" << std::endl;
    }
    std::cout << "---------------------------------------------" << std::endl;
    std::cout << "  Trust-region radius   :  " << delta << std::endl;
    if ( EXIT_FLAG != -5 ) {
      std::cout << "  Number of evaluations :  " << evaluations.values[0].size() << std::endl;
    } else {
      std::cout << "  Number of evaluations :  " << evaluations.values[0].size()+1 << std::endl;
    }
    int nb_eval_out_of_bounds = 0;
    for (int j = 0; j < evaluations.nodes.size(); ++j)
    {
      for (int i = 0; i < dim; ++i)
      {
        if (!lower_bound_constraints.empty()) {
          if (evaluations.nodes[j][i] < lower_bound_constraints[i]){
            nb_eval_out_of_bounds++;
          }
        }
        if (!upper_bound_constraints.empty()){
          if (evaluations.nodes[j][i] > upper_bound_constraints[i]){
            nb_eval_out_of_bounds++;
          }
        }
      }
    }
    std::cout << "  Number of evaluations out of bounds : " << nb_eval_out_of_bounds << std::endl;
    std::cout << "---------------------------------------------" << std::endl;
    if ( this->noise_has_been_detected )
      std::cout << " Noise detected" << std::endl;
    if ( EXIT_FLAG == 0)
      std::cout << " Minimal trust region radius reached" << std::endl;
    if ( EXIT_FLAG == 1 )
      std::cout << " Maximal number of evaluations reached" << std::endl;
    if ( EXIT_FLAG == -2 )
      std::cout << " Termination due to Noise" << std::endl;
    if ( EXIT_FLAG == -4 )
      std::cout << " Inconsistent parameter" << std::endl;
    if ( EXIT_FLAG == -5 )
      std::cout << " Black box returned invalid value" << std::endl;
    if ( EXIT_FLAG == -6 )
      std::cout << " Other termination reason" << std::endl;
    if ( EXIT_FLAG == -7 )
      std::cout << " Negative variance in GP estimation due to ill-conditioning" << std::endl;
    std::cout << "*********************************************" << std::endl << std::endl << std::flush;
  }


  
  return 1;
}
//--------------------------------------------------------------------------------
