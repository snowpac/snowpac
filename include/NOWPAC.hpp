//#ifndef HNOWPAC
//#define HNOWPAC

#include "BlackboxData.hpp"
#include "BasisForMinimumFrobeniusNormModel.hpp"
#include "SubproblemOptimization.hpp"
#include "MinimumFrobeniusNormModel.hpp"
#include "ImprovePoisedness.hpp"
#include "BlackBoxBaseClass.hpp"
#include "GaussianProcessSupport.hpp"
#include "VectorOperations.hpp"
#include <Eigen/Core>
#include <vector>
#include <string>
#include <iostream>
#include <random>
#include <iomanip>
#include <fstream>
#include <cassert>

template<class TSurrogateModel = MinimumFrobeniusNormModel, 
         class TBasisForSurrogateModel = BasisForMinimumFrobeniusNormModel>
class NOWPAC : protected VectorOperations {
  private:
    int dim;
    int nb_constraints;
    GaussianProcessSupport gaussian_processes;
    BlackBoxBaseClass *blackbox;
    TBasisForSurrogateModel surrogate_basis;
    std::vector<TSurrogateModel> surrogate_models;
    std::vector<double> blackbox_values;
    std::vector<double> blackbox_noise;
    bool stochastic_optimization;
    void blackbox_evaluator ( std::vector<double> const&, bool );
    void blackbox_evaluator ( );
    void update_surrogate_models ( );
    void update_trustregion ( double );
    bool last_point_is_feasible ( );
    void add_trial_node( );
    double compute_acceptance_ratio ( );
    void *user_data = NULL;
    BlackboxData evaluations;
    std::vector<double> x_trial;
    double delta, delta_min, delta_max;
    double omega, theta, gamma, gamma_inc, mu;
    double eta_0, eta_1, eps_c;
    double threshold_for_poisedness_constant;
    std::vector<double> inner_boundary_path_constants;
    std::vector<double> lower_bound_constraints;
    std::vector<double> upper_bound_constraints;
    double criticality_value;
    double trial_model_value;
    double tmp_dbl;
    int verbose;
    double acceptance_ratio;
    int replace_node_index;
    int max_number_blackbox_evaluations;
    std::vector<double> update_at_evaluations;
    int update_interval_length;
    int EXIT_FLAG;
    int NOEXIT;
  public:
    NOWPAC ( int ); 
    void set_blackbox ( BlackBoxBaseClass&, int );
    void set_blackbox ( BlackBoxBaseClass& );
    int optimize ( std::vector<double>&, double& );
    void set_option ( std::string const&, int const& );
    void set_option ( std::string const&, double const& );
    void set_option ( std::string const&, bool const& );
    void set_option ( std::string const&, std::vector<double> const& );
    void set_option ( std::string const&, std::vector<int> const& );
    void void_user_data ( void* );
    void set_lower_bounds ( std::vector<double> const& );
    void set_upper_bounds ( std::vector<double> const& );
    void set_trustregion ( double const& );
    void set_trustregion ( double const&, double const& );
    void set_max_trustregion ( double const& );
    void set_max_number_evaluations ( int const& );
};

//#endif


//--------------------------------------------------------------------------------    
template<class TSurrogateModel, class TBasisForSurrogateModel>
NOWPAC<TSurrogateModel, TBasisForSurrogateModel>::NOWPAC ( int dim_input ) :
  dim ( dim_input ),
  surrogate_basis ( dim_input, delta )
{ 
  stochastic_optimization = false;
  x_trial.resize( dim );
  delta = 1e0;
  delta_max = 1e0;
  delta_min = 1e-3;
  threshold_for_poisedness_constant = 5e1;
  omega = 0.4; // 0.6
  theta = 0.5;
  gamma = 0.8;
  gamma_inc = 1.4;
  eta_0 = 0.1; 
  eta_1 = 0.7;
  eps_c = 1e-3;
  mu = 1e0;
  nb_constraints = -1;
  evaluations.max_nb_nodes = (dim*dim + 3*dim + 2)/2;
//  evaluations.max_nb_nodes = 2*dim+1;
  max_number_blackbox_evaluations = (int) HUGE_VAL;
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
  if ( option_name.compare( "eps_c" ) == 0 ) 
    { eps_c = option_value; return; }
  if ( option_name.compare( "mu" ) == 0 ) 
    { mu = option_value; return; }
  if ( option_name.compare( "poisedness_threshold" ) == 0 ) 
    { threshold_for_poisedness_constant = option_value; return; }
  if ( option_name.compare( "inner_boundary_path_constants" ) == 0 ) { 
    if ( nb_constraints == 0 ) {
      std::cout << "Unable to set inner boundary path constants" << std::endl;
      std::cout << "Reason: no constraints specified" << std::endl;
      return;
    } else {
      if ( option_value < 0e0 )
        std::cout << "Inner boundary path constants have to be positive" << std::endl;
      for ( int i = 0; i < nb_constraints; i++ )
        inner_boundary_path_constants.at( i ) = option_value; 
      return;
    } 
  }
  std::cout << "Unknown parameter: " << option_name << std::endl;
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
      evaluations.noise.resize( nb_constraints + 1);
      //for (int i = evaluations.noise.size(); 
       //    i < evaluations.values.size(); i++)
       // evaluations.noise[i].resize(0);
    }
    return; 
  }
  std::cout << "Unknown parameter: " << option_name << std::endl;
  return;
}
//--------------------------------------------------------------------------------

//--------------------------------------------------------------------------------
template<class TSurrogateModel, class TBasisForSurrogateModel>
void NOWPAC<TSurrogateModel, TBasisForSurrogateModel>::set_option ( 
  std::string const &option_name, std::vector<double> const &option_value )
{
  if ( option_name.compare( "inner_boundary_path_constants" ) == 0 ) { 
    if ( nb_constraints == 0 ) {
      std::cout << "Unable to set inner boundary path constants" << std::endl;
      std::cout << "Reason: no constraints specified" << std::endl;
      return;
    } else {
      for ( int i = 0; i < option_value.size(); ++i )
        inner_boundary_path_constants.push_back( option_value.at(i) ); 
      return; 
    }
  }
  std::cout << "Unknown parameter: " << option_name << std::endl;
  return;
}
//--------------------------------------------------------------------------------

//--------------------------------------------------------------------------------
template<class TSurrogateModel, class TBasisForSurrogateModel>
void NOWPAC<TSurrogateModel, TBasisForSurrogateModel>::set_option ( 
  std::string const &option_name, std::vector<int> const &option_value )
{
  if ( option_name.compare( "update_at_evaluations" ) == 0 ) { 
    for ( int i = 0; i < option_value.size(); ++i )
      update_at_evaluations.push_back( option_value.at( i ) );
    return;
  }
  std::cout << "Unknown parameter: " << option_name << std::endl;
  return;
}
//--------------------------------------------------------------------------------

//--------------------------------------------------------------------------------
template<class TSurrogateModel, class TBasisForSurrogateModel>
void NOWPAC<TSurrogateModel, TBasisForSurrogateModel>::set_option ( 
  std::string const &option_name, int const &option_value )
{
  if ( option_name.compare( "verbose" ) == 0 ) { 
    verbose = option_value; 
    return;
  }
  if ( option_name.compare( "update_interval_length" ) == 0 ) { 
    update_interval_length = option_value; 
    return;
  }
  std::cout << "Unknown parameter: " << option_name << std::endl;
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
template<class TSurrogateModel, class TBasisForSurrogateModel>
void NOWPAC<TSurrogateModel, TBasisForSurrogateModel>::set_upper_bounds ( 
  std::vector<double> const& bounds )
{
  upper_bound_constraints = bounds;
  return;
}
//--------------------------------------------------------------------------------


//--------------------------------------------------------------------------------
template<class TSurrogateModel, class TBasisForSurrogateModel>
void NOWPAC<TSurrogateModel, TBasisForSurrogateModel>::set_trustregion ( 
  double const &delta_init_input, double const &delta_min_input )
{
  delta = delta_init_input;
  delta_min = delta_min_input;
  return;
}
//--------------------------------------------------------------------------------

//--------------------------------------------------------------------------------
template<class TSurrogateModel, class TBasisForSurrogateModel>
void NOWPAC<TSurrogateModel, TBasisForSurrogateModel>::set_trustregion ( 
  double const &delta_init_input )
{
  set_trustregion ( delta_init_input, 0 );
  return;
}
//--------------------------------------------------------------------------------

//--------------------------------------------------------------------------------
template<class TSurrogateModel, class TBasisForSurrogateModel>
void NOWPAC<TSurrogateModel, TBasisForSurrogateModel>::set_max_trustregion ( 
  double const &delta_max_input )
{
  delta_max = delta_max_input;
  return;
}
//--------------------------------------------------------------------------------

//--------------------------------------------------------------------------------
template<class TSurrogateModel, class TBasisForSurrogateModel>
void NOWPAC<TSurrogateModel, TBasisForSurrogateModel>::set_max_number_evaluations ( 
  int const &max_number_blackbox_evaluations_input )
{
  max_number_blackbox_evaluations = max_number_blackbox_evaluations_input;
  return;
}
//--------------------------------------------------------------------------------

//--------------------------------------------------------------------------------
template<class TSurrogateModel, class TBasisForSurrogateModel>
void NOWPAC<TSurrogateModel, TBasisForSurrogateModel>::set_blackbox ( 
  BlackBoxBaseClass &blackbox_input, int nb_constraints_input )
{
  blackbox = &blackbox_input;
  nb_constraints = nb_constraints_input;
  blackbox_values.resize( (nb_constraints+1) );
  evaluations.values.resize( nb_constraints + 1);
  if ( stochastic_optimization ) {
    blackbox_noise.resize( nb_constraints + 1);
    evaluations.noise.resize( nb_constraints + 1);
  }
  inner_boundary_path_constants.resize( nb_constraints );
  for (int i = 0; i < nb_constraints; i++)
    inner_boundary_path_constants.at( i ) = 0.1; 
  return;
}
//--------------------------------------------------------------------------------

//--------------------------------------------------------------------------------
template<class TSurrogateModel, class TBasisForSurrogateModel>
void NOWPAC<TSurrogateModel, TBasisForSurrogateModel>::set_blackbox ( 
  BlackBoxBaseClass &blackbox_input )
{
  set_blackbox ( blackbox_input, 0 );
  return;
}
//--------------------------------------------------------------------------------

//--------------------------------------------------------------------------------
template<class TSurrogateModel, class TBasisForSurrogateModel>
double NOWPAC<TSurrogateModel, TBasisForSurrogateModel>::compute_acceptance_ratio ( )
{
  if ( stochastic_optimization ) {
//    update_surrogate_models( );
//    trial_model_value = surrogate_models[0].evaluate( x_trial );    
  }
  return ( evaluations.values[0].at( evaluations.best_index ) - 
           evaluations.values[0].back() ) /
         ( evaluations.values[0].at( evaluations.best_index ) - 
           trial_model_value );
}
//--------------------------------------------------------------------------------


//--------------------------------------------------------------------------------
template<class TSurrogateModel, class TBasisForSurrogateModel>
void NOWPAC<TSurrogateModel, TBasisForSurrogateModel>::void_user_data ( void *user_data_input ) 
{ 
  user_data = user_data_input; 
  return; 
}
//--------------------------------------------------------------------------------

//--------------------------------------------------------------------------------
template<class TSurrogateModel, class TBasisForSurrogateModel>
void NOWPAC<TSurrogateModel, TBasisForSurrogateModel>::blackbox_evaluator ( 
  std::vector<double> const &x, bool set_node_active ) 
{
  if ( evaluations.nodes.size() == max_number_blackbox_evaluations) {
    EXIT_FLAG = 1;
    return;
  }
  // evaluate blackbox  
  if ( stochastic_optimization )
    blackbox->evaluate( x, blackbox_values, blackbox_noise, user_data );
  else
    blackbox->evaluate( x, blackbox_values, user_data ); 


//    std::cout << std::setprecision(26) << (blackbox_values.at(0) ) << std::endl;
//    std::cout << std::setprecision(26) << (blackbox_values.at(1) ) << std::endl;
//    std::cout << std::setprecision(26) << (blackbox_values.at(2) ) << std::endl;
//    std::cout << std::setprecision(26) << (blackbox_values.at(3) ) << std::endl;
//    std::cout << std::setprecision(26) << (blackbox_values.at(4) ) << std::endl;

  
  // add evaluations to blackbox data
  evaluations.nodes.push_back( x );
  for ( int i = 0; i < nb_constraints+1; i++) {  
    if ( stochastic_optimization ) 
      evaluations.noise[i].push_back( blackbox_noise.at(i) );
    evaluations.values[i].push_back( blackbox_values.at(i) );
  }  









  if ( set_node_active )
    evaluations.surrogate_nodes_index.push_back( (evaluations.nodes).size()-1 );    

  if ( stochastic_optimization && evaluations.nodes.size() > dim ) 
    gaussian_processes.smooth_data ( evaluations );

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

  for ( int i = nb_vals_tmp; i < nb_nodes_tmp; i++ ) {
    // evaluate blackbox
    if ( stochastic_optimization )
      blackbox->evaluate( evaluations.nodes[ i ], blackbox_values, 
                          blackbox_noise, user_data );
    else
      blackbox->evaluate( evaluations.nodes[ i ], blackbox_values, 
                          user_data ); 

  //  std::cout << std::setprecision(26) << (blackbox_values.at(0) ) << std::endl;
  //  std::cout << std::setprecision(26) << (blackbox_values.at(1) ) << std::endl;
  //  std::cout << std::setprecision(26) << (blackbox_values.at(2) ) << std::endl;
  //  std::cout << std::setprecision(26) << (blackbox_values.at(3) ) << std::endl;
  //  std::cout << std::setprecision(26) << (blackbox_values.at(4) ) << std::endl;


    // add evaluations to blackbox data
    for ( int j = 0; j < nb_constraints+1; j++) {
      if ( stochastic_optimization )
        evaluations.noise[j].push_back( blackbox_noise.at(j) );
      evaluations.values[j].push_back( blackbox_values.at(j) );
    }  
  }
 
  assert ( evaluations.nodes.size() == evaluations.values[0].size() );

  if ( stochastic_optimization ) 
    gaussian_processes.smooth_data ( evaluations );

  return;
}  
//--------------------------------------------------------------------------------

//--------------------------------------------------------------------------------
template<class TSurrogateModel, class TBasisForSurrogateModel>
bool NOWPAC<TSurrogateModel, TBasisForSurrogateModel>::last_point_is_feasible ( ) 
{
  for ( int i = 0; i < nb_constraints; i++ ) {
    tmp_dbl = evaluations.values[i+1].at( evaluations.best_index );
    if ( tmp_dbl < 0e0 ) tmp_dbl = 0e0;
    if ( evaluations.values[i+1].back() > tmp_dbl ) return false;
  }
  return true;
}
//--------------------------------------------------------------------------------

//--------------------------------------------------------------------------------
template<class TSurrogateModel, class TBasisForSurrogateModel>
void NOWPAC<TSurrogateModel, TBasisForSurrogateModel>::update_surrogate_models ( )
{
  surrogate_basis.compute_basis_coefficients ( evaluations );
  for ( int i = 0; i < nb_constraints+1; i++ )
    surrogate_models[ i ].set_function_values ( evaluations.values[i],
                                                evaluations.noise[i],
                                                evaluations.surrogate_nodes_index );
  return;
}
//--------------------------------------------------------------------------------

//--------------------------------------------------------------------------------
template<class TSurrogateModel, class TBasisForSurrogateModel>
void NOWPAC<TSurrogateModel, TBasisForSurrogateModel>::update_trustregion ( 
  double scaling_factor )
{
  tmp_dbl = 0e0;
  if ( stochastic_optimization ) {
    for ( int i = 0; i < evaluations.surrogate_nodes_index.size(); i++ ) {
      for ( int j = 0; j < nb_constraints+1; j++) {
        if ( evaluations.noise[ j ].at( evaluations.surrogate_nodes_index[i] ) > tmp_dbl ) {
          if ( diff_norm( 
                 evaluations.nodes[ evaluations.surrogate_nodes_index[i] ], 
                 evaluations.nodes[ evaluations.best_index ] ) <= delta )
            tmp_dbl = evaluations.noise[ j ].at( evaluations.surrogate_nodes_index[i] );
        }
      }
    }
  }
  delta *= scaling_factor;
//  std::cout << std::endl << "------------------------- " << std::endl;
//  std::cout << "MAXNOISE " << tmp_dbl << " sqrt = " << sqrt( tmp_dbl ) << std::endl;
//  std::cout <<  "------------------------- " << std::endl;
  if ( stochastic_optimization ) {
    int ar_tmp = fabs(acceptance_ratio);
    if (ar_tmp < 1e0) ar_tmp = 1e0;
    if (ar_tmp > 2e1) ar_tmp = 2e1;
    ar_tmp = 1e0;
    if ( delta < sqrt(1e0*tmp_dbl)*1e0*ar_tmp  ) 
      delta = sqrt(1e0*tmp_dbl) * 1e0 * ar_tmp; 
  }
  if ( delta > delta_max ) delta = delta_max;
  if ( delta < delta_min ) EXIT_FLAG = 0;
  return;
}
//--------------------------------------------------------------------------------

//--------------------------------------------------------------------------------
template<class TSurrogateModel, class TBasisForSurrogateModel>
void NOWPAC<TSurrogateModel, TBasisForSurrogateModel>::add_trial_node ( ) 
{
  if ( diff_norm ( evaluations.nodes.at( evaluations.best_index), 
                   evaluations.nodes.at( evaluations.nodes.size()-1 ) ) > 1e-2 * delta ) {
    if ( evaluations.surrogate_nodes_index.size( ) < evaluations.max_nb_nodes )
      evaluations.surrogate_nodes_index.push_back ( evaluations.nodes.size()-1 );
    else
      evaluations.surrogate_nodes_index[ replace_node_index ] = evaluations.nodes.size()-1;
  }
  if ( stochastic_optimization )
    gaussian_processes.smooth_data ( evaluations );
  return;
}
//--------------------------------------------------------------------------------

//--------------------------------------------------------------------------------
template<class TSurrogateModel, class TBasisForSurrogateModel>
int NOWPAC<TSurrogateModel, TBasisForSurrogateModel>::optimize ( 
  std::vector<double> &x, double &val ) 
{
  
  if ( nb_constraints == -1 ) {
    EXIT_FLAG = -4;
    std::cout << "No objective function has been specified" << std::endl;
  }
  if ( max_number_blackbox_evaluations < dim+1 ) {
    std::cout << "Maximal number of blackbox evaluations to small" << std::endl;
    EXIT_FLAG = -4;
  }
  if ( EXIT_FLAG != NOEXIT ) return EXIT_FLAG;
  std::cout.setf( std::ios::fixed, std:: ios::floatfield );
  std::cout << std::setprecision(8);


  TSurrogateModel surrogate_model_prototype( surrogate_basis );
  for ( int i = 0; i < nb_constraints+1; i++ )
    surrogate_models.push_back ( surrogate_model_prototype );

  ImprovePoisedness surrogate_nodes ( surrogate_basis, threshold_for_poisedness_constant,
                                      evaluations.max_nb_nodes, delta, verbose );
  SubproblemOptimization<TSurrogateModel> surrogate_optimization (
    surrogate_models, delta, inner_boundary_path_constants);
  if ( !upper_bound_constraints.empty() ) 
    surrogate_optimization.set_upper_bounds ( upper_bound_constraints );
  if ( !lower_bound_constraints.empty() ) 
    surrogate_optimization.set_lower_bounds ( lower_bound_constraints );

  if ( stochastic_optimization)
    gaussian_processes.initialize ( dim, nb_constraints + 1, delta,
                                    update_at_evaluations, update_interval_length );

//  std::random_device rand_dev;
  int random_seed = 25041981;//rand_dev();
  std::mt19937 rand_generator(random_seed);
  std::normal_distribution<double> norm_dis(0e0,3e-1);

  if ( verbose >= 2 ) { std::cout << "Initial evaluation of black box functions" << std::endl; }
  // initial evaluations 
  evaluations.best_index = 0;
  blackbox_evaluator( x, true );   
  if ( !last_point_is_feasible ( ) ) {
    if ( verbose >= 1 ) 
      std::cout << "Initial point is not feasibile" << std::endl;
//    EXIT_FLAG = -3;
//    return EXIT_FLAG;
  }
  for ( int i = 0; i < dim; i++ ) {
    x_trial = x;
    x_trial.at(i) += delta;
    blackbox_evaluator( x_trial, true );
  }
  if ( verbose == 3 ) { std::cout << "Building initial models ... "; }
  update_surrogate_models();
  if ( verbose == 3 ) { std::cout << "done" << std::endl; }



/*
  std::vector<double> x_loc(2);
  std::vector<double> fvals(3);
  std::ofstream outputfile ( "surrogate_data_o.dat" );
  if ( outputfile.is_open( ) ) {
    for (double i = -1.0; i <= 2.0; i+=0.1) {
      x_loc.at(0) = i;
      for (double j = -1.0; j < 2.0; j+=0.1) {
        x_loc.at(1) = j;
        fvals.at(0) = surrogate_models[0].evaluate( x_loc );
        fvals.at(1) = surrogate_models[1].evaluate( x_loc );
        fvals.at(2) = surrogate_models[2].evaluate( x_loc );
        outputfile << x_loc.at(0) << "; " << x_loc.at(1) << "; " << fvals.at(0)<< "; " << 
                     fvals.at(1)<< "; " << fvals.at(2) << std::endl;
      }
    }
    outputfile.close( );
  } else std::cout << "Unable to open file." << std::endl;

  return 0;
*/

  x_trial = evaluations.nodes[ evaluations.best_index ];
  if ( verbose == 3 ) { std::cout << "Value of criticality measure : "; }
  criticality_value = surrogate_optimization.compute_criticality_measure( x_trial );
  if ( verbose == 3 ) { std::cout << criticality_value << std::endl;; }

  do {
    /*----------------------------------------------------------------------------------*/
    /*- STEP 1 - STEP 1 - STEP 1 - STEP 1 - STEP 1 - STEP 1 - STEP 1 - STEP 1 - STEP 1 -*/
    /*----------------------------------------------------------------------------------*/
    //if criticality measure is small, check if model improvement is necessary
    if ( criticality_value <= eps_c ) {
      if ( verbose == 3 && delta > mu * criticality_value ) {
        std::cout << " Criticality measure below threshold" << std::endl;
        std::cout << " -----------------------------------" << std::endl;
      }
      while ( delta > mu * criticality_value ) {
        update_trustregion( omega );
        if ( EXIT_FLAG != NOEXIT ) break;
        if (verbose == 3) {
          std::cout << " Adjusting trust-region radius to " << delta << std::endl;
        }
        surrogate_nodes.improve_poisedness ( evaluations.best_index, evaluations );
        if ( stochastic_optimization ) {
          for ( int i = 0; i < dim; i++ )
            x_trial.at(i) = evaluations.nodes[ evaluations.best_index].at( i ) +
                         delta * norm_dis( rand_generator ) * 1e0;
          evaluations.nodes.push_back( x_trial );
        }
        blackbox_evaluator( );
        if ( EXIT_FLAG != NOEXIT ) break;
        update_surrogate_models( );
        x_trial = evaluations.nodes[ evaluations.best_index ];
        criticality_value = surrogate_optimization.compute_criticality_measure( x_trial );
        if (verbose == 3) {
          std::cout << " Value of criticality measure is " << criticality_value << std::endl;
          std::cout << " -----------------------------------" << std::endl;
        }
      }
    }
    /*----------------------------------------------------------------------------------*/
    /*- STEP 2 - STEP 2 - STEP 2 - STEP 2 - STEP 2 - STEP 2 - STEP 2 - STEP 2 - STEP 2 -*/
    /*----------------------------------------------------------------------------------*/
    while (EXIT_FLAG == NOEXIT)  {
      if ( verbose == 3 ) { 
        std::cout << "Computing trial point ... ";
      }
      x_trial = evaluations.nodes[ evaluations.best_index ];
      trial_model_value = surrogate_optimization.compute_trial_point ( x_trial );
      if ( verbose == 3 ) { std::cout << "done" << std::endl; }
      trial_model_value = surrogate_models[0].evaluate( x_trial );

      /*-------------------------------------------------------------------------*/
      /*- STEP 3 - STEP 3 - STEP 3 - STEP 3 - STEP 3 - STEP 3 - STEP 3 - STEP 3 -*/
      /*-------------------------------------------------------------------------*/
      if ( verbose == 3 ) { std::cout << "Checking feasibility .... "; }
      blackbox_evaluator( x_trial, false );   
      if ( verbose == 3 ) { std::cout << "done" << std::endl; }
      if ( !last_point_is_feasible( ) ) {
        if ( verbose == 3 ) { std::cout << " Feasibility violated" << std::endl; }
        replace_node_index = surrogate_nodes.replace_node( evaluations.best_index, 
                                                           evaluations, x_trial );
        add_trial_node( );
        update_surrogate_models( );
        update_trustregion( theta );
        if ( EXIT_FLAG != NOEXIT ) break;
        surrogate_nodes.improve_poisedness (evaluations.best_index, evaluations );
        if ( stochastic_optimization ) {
          for ( int i = 0; i < dim; i++ )
            x_trial.at(i) = evaluations.nodes[ evaluations.best_index].at( i ) +
                            delta * norm_dis( rand_generator ) * 1e0;
          evaluations.nodes.push_back( x_trial );
        }
        blackbox_evaluator( );
        if ( EXIT_FLAG != NOEXIT ) break;
        update_surrogate_models( );
      } else {
        if ( verbose == 3 ) { std::cout << "Found feasible trial step" << std::endl; }
        break;
      }
    }
   


    /*----------------------------------------------------------------------------------*/
    /*- STEP 4 - STEP 4 - STEP 4 - STEP 4 - STEP 4 - STEP 4 - STEP 4 - STEP 4 - STEP 4 -*/
    /*----------------------------------------------------------------------------------*/
    if ( EXIT_FLAG == NOEXIT ) { 
      acceptance_ratio = compute_acceptance_ratio ( );
      if ( verbose == 3 ) { std::cout << std::endl; }
      if ( verbose >= 2 ) { 
        std::cout << "*****************************************" << std::endl; 
      }
      if ( acceptance_ratio >= eta_1 && acceptance_ratio < 2e0 ) {
        if ( verbose >= 2 ) { std::cout << " Step successful" << std::endl; }
        update_trustregion( gamma_inc );
      } 
      if ( (acceptance_ratio >= eta_0 && acceptance_ratio < eta_1) || acceptance_ratio >= 2e0 ) {
        if ( verbose >= 2 ) { std::cout << " Step acceptable" << std::endl; }
      }
      if ( acceptance_ratio >= eta_0 ) {
        replace_node_index = surrogate_nodes.replace_node( evaluations.best_index, 
                                                           evaluations, x_trial );
        add_trial_node( );
        update_surrogate_models( );
        evaluations.best_index = evaluations.nodes.size()-1;
      }
      if ( acceptance_ratio < eta_0 ) {
        if ( verbose >= 2 ) { std::cout << " Step rejected" << std::endl; }
        replace_node_index = surrogate_nodes.replace_node( evaluations.best_index, 
                                                           evaluations, x_trial );
        add_trial_node( );
        update_surrogate_models( );
        if ( stochastic_optimization ) {
          for ( int i = 0; i < dim; i++ )
            x_trial.at(i) = evaluations.nodes[ evaluations.best_index].at( i ) +
                            delta * norm_dis( rand_generator ) * 1e0;
          evaluations.nodes.push_back( x_trial );
        }
        blackbox_evaluator( );
        update_trustregion( gamma );
        if ( EXIT_FLAG == NOEXIT ) {
          surrogate_nodes.improve_poisedness( evaluations.best_index, evaluations );
          blackbox_evaluator( );
          if ( EXIT_FLAG != NOEXIT ) {
            std::cout << "*****************************************" << std::endl;
            break;
          }
          update_surrogate_models( );
        }
      }

      x_trial = evaluations.nodes[ evaluations.best_index ];
      criticality_value = surrogate_optimization.compute_criticality_measure( x_trial );
    
//  val = gaussian_processes.evaluate_objective( evaluations);

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
  //      std::cout << "  Current value         :  " << val << std::endl;
        if ( dim > 1) {
          std::cout << "  Current point         : [" << std::setprecision(8) << evaluations.nodes[ evaluations.best_index ].at(0) << std::endl;
          for (int i = 1; i < dim-1; i++)
            std::cout << "                           " << std::setprecision(8) << evaluations.nodes[ evaluations.best_index ].at(i) << std::endl;
          std::cout << "                           " << std::setprecision(8) << evaluations.nodes[ evaluations.best_index ].at(dim-1) << "]"<< std::endl;
        } else
          std::cout << "  Current point         : [" << std::setprecision(8) << evaluations.nodes[ evaluations.best_index ].at(0) << "]"<< std::endl;
        std::cout << "*****************************************" << std::endl << std::endl;
      }

    }

  } while ( EXIT_FLAG == NOEXIT );

//  val = gaussian_processes.evaluate_objective( evaluations);
  val = evaluations.values[0].at(evaluations.best_index);
  x = evaluations.nodes[ evaluations.best_index ];

  if ( verbose >= 2 ) {
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
    std::cout << "  Trust-region radius   :  " << delta << std::endl;
    std::cout << "  Number of evaluations :  " << evaluations.values[0].size() << std::endl;
    std::cout << "---------------------------------------------" << std::endl;
    if ( EXIT_FLAG == 0 )
      std::cout << " Minimal trust region radius reached" << std::endl;
    if ( EXIT_FLAG == 1 )
      std::cout << " Maximal number of evaluations reached" << std::endl;
    if ( EXIT_FLAG == -4 )
      std::cout << " Inconsistent parameter" << std::endl;
    std::cout << "*********************************************" << std::endl << std::endl;
  }



/*
  Eigen::VectorXd x_loc(2);
  Eigen::VectorXd fvals(3);
  std::ofstream outputfile ( "surrogate_data_o.dat" );
  if ( outputfile.is_open( ) ) {
    for (double i = -1.0; i <= 2.0; i+=0.01) {
      x_loc(0) = i;
      for (double j = -1.0; j < 2.0; j+=0.01) {
        x_loc(1) = j;
        fvals(0) = surrogate_models[0].evaluate( x_loc );
        fvals(1) = surrogate_models[1].evaluate( x_loc );
        fvals(2) = surrogate_models[2].evaluate( x_loc );
        outputfile << x_loc(0) << "; " << x_loc(1) << "; " << fvals(0)<< "; " << 
                     fvals(1)<< "; " << fvals(2) << std::endl;
      }
    }
    outputfile.close( );
  } else std::cout << "Unable to open file." << std::endl;
*/

  

  return 1;
}
//--------------------------------------------------------------------------------
