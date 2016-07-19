//#ifndef HNOWPAC
//#define HNOWPAC

#include "BlackBoxData.hpp"
#include "BasisForMinimumFrobeniusNormModel.hpp"
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

template<class TSurrogateModel = MinimumFrobeniusNormModel, 
         class TBasisForSurrogateModel = BasisForMinimumFrobeniusNormModel>
class NOWPAC : protected NoiseDetection<TSurrogateModel> {
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
    void write_to_file( );
    void output_for_plotting( );
    const char *double_format = "  %.16e  ";
    const char *int_format    = "  %d  ";
    const char* output_filename = NULL;
    std::FILE *output_file;
    void *user_data = NULL;
    BlackBoxData evaluations;
    std::vector<double> x_trial;
    double delta, delta_min, delta_max;
    double omega, theta, gamma, gamma_inc, mu;
    double eta_0, eta_1, eps_c;
    double threshold_for_poisedness_constant;
    std::vector<double> inner_boundary_path_constants;
    std::vector<double> max_inner_boundary_path_constants;
    std::vector<double> lower_bound_constraints;
    std::vector<double> upper_bound_constraints;
    double criticality_value;
    double trial_model_value;
    double tmp_dbl;
    double stepsize[2];
    double max_noise;
    int noise_observation_span;
    int nb_allowed_noisy_iterations;
    bool check_for_noise;
    int verbose;
    double acceptance_ratio;
    int replace_node_index;
    int max_number_blackbox_evaluations;
    std::vector<double> update_at_evaluations;
    int update_interval_length;
    int EXIT_FLAG;
    int NOEXIT;
    double tmp_dbl1;
  public:
    ~NOWPAC ( ); 
    NOWPAC ( int ); 
    NOWPAC ( int, const char* ); 
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
NOWPAC<TSurrogateModel, TBasisForSurrogateModel>::NOWPAC ( 
  int dim_input, const char *output_filename_input ) : NOWPAC ( dim_input ) {
  output_filename = output_filename_input;
  output_file = fopen(output_filename, "w");
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
NOWPAC<TSurrogateModel, TBasisForSurrogateModel>::NOWPAC ( int dim_input ) :
  dim ( dim_input ),
  surrogate_basis ( dim_input ),
  NoiseDetection<TSurrogateModel>( surrogate_models, delta, 5, 3)
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
  max_noise = -1e0;
  noise_observation_span = 5;
  nb_allowed_noisy_iterations = 3;
  check_for_noise = true;
  stepsize[0] = 1e0; stepsize[1] = 0e0;
  evaluations.max_nb_nodes = (dim*dim + 3*dim + 2)/2;
//  evaluations.max_nb_nodes = dim +1;
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
  if ( option_name.compare( "geometry_threshold" ) == 0 ) 
    { threshold_for_poisedness_constant = option_value; return; }
  if ( option_name.compare( "eps_b" ) == 0 ) { 
    if ( nb_constraints == 0 ) {
      std::cout << "Unable to set inner boundary path constants" << std::endl;
      std::cout << "Reason: no constraints specified" << std::endl;
      return;
    } else {
      if ( option_value < 0e0 )
        std::cout << "Inner boundary path constants have to be positive" << std::endl;
      for ( int i = 0; i < nb_constraints; i++ ) {
        inner_boundary_path_constants.at( i ) = option_value; 
        max_inner_boundary_path_constants.at( i ) = option_value; 
      }
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
      //evaluations.noise.resize( nb_constraints + 1);
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
  if ( option_name.compare( "eps_b" ) == 0 ) { 
    if ( nb_constraints == 0 ) {
      std::cout << "Unable to set inner boundary path constants" << std::endl;
      std::cout << "Reason: no constraints specified" << std::endl;
      return;
    } else {
      for ( int i = 0; i < option_value.size(); ++i ) {
        inner_boundary_path_constants.push_back( option_value.at(i) ); 
        max_inner_boundary_path_constants.push_back( option_value.at(i) ); 
      }
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
  double const &delta_init_input, double const &delta_min_input )
{
  delta = delta_init_input;
  delta_min = delta_min_input;
  return;
}
//--------------------------------------------------------------------------------

//--------------------------------------------------------------------------------
// sets initial trust region radius,
// minimal trust region radius is not used as stopping criterion
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
// sets maximal trust region radius
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
// sets maximal number of black box evaluations before termination
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
// sets the black box with constraints
//--------------------------------------------------------------------------------
template<class TSurrogateModel, class TBasisForSurrogateModel>
void NOWPAC<TSurrogateModel, TBasisForSurrogateModel>::set_blackbox ( 
  BlackBoxBaseClass &blackbox_input, int nb_constraints_input )
{
  blackbox = &blackbox_input;
  nb_constraints = nb_constraints_input;
  blackbox_values.resize( (nb_constraints+1) );
  evaluations.initialize ( nb_constraints+1, dim );
//  evaluations.values.resize( nb_constraints + 1);
  if ( stochastic_optimization ) {
    blackbox_noise.resize( nb_constraints + 1);
//    evaluations.noise.resize( nb_constraints + 1);
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
    //update_surrogate_models( );
    //trial_model_value = surrogate_models[0].evaluate( evaluations.transform(x_trial) );
  }

//  if ( evaluations.values[0].at(evaluations.best_index) < trial_model_value )
//    return -0.4251981;
/*
  if (( evaluations.values[0].at( evaluations.best_index ) - 
           evaluations.values[0].back() ) /
         ( evaluations.values[0].at( evaluations.best_index ) - 
           trial_model_value ) < 0e0 ) {

     
     std::cout << " best value = " << evaluations.values[0].at(evaluations.best_index) << std::endl;
     std::cout << " next value = " << evaluations.values[0].back() << std::endl; 
     std::cout << " surr value = " << trial_model_value << std::endl;
  }
*/

  acceptance_ratio = ( evaluations.values[0].at( evaluations.best_index ) - 
                       evaluations.values[0].back() ) /
                     ( evaluations.values[0].at( evaluations.best_index ) - 
                       trial_model_value );

  if ( acceptance_ratio != acceptance_ratio ) acceptance_ratio = -0.4251981;

  return acceptance_ratio;
}
//--------------------------------------------------------------------------------

//--------------------------------------------------------------------------------
// sets user data that is passed through (S)NOWPAC to the black box function
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
  for (int i = 0; i < nb_constraints+1; ++i) {  
    if ( stochastic_optimization ) 
      evaluations.noise[i].push_back( blackbox_noise.at(i) );
    evaluations.values[i].push_back( blackbox_values.at(i) );
  }  

  if ( set_node_active )
    evaluations.active_index.push_back( (evaluations.nodes).size()-1 );    

  if ( stochastic_optimization && evaluations.nodes.size() > dim ) 
    gaussian_processes.smooth_data ( evaluations );

  write_to_file();

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

  for ( int i = nb_vals_tmp; i < nb_nodes_tmp; ++i ) {
    // evaluate blackbox
    if ( stochastic_optimization )
      blackbox->evaluate( evaluations.nodes[ i ], blackbox_values, 
                          blackbox_noise, user_data );
    else
      blackbox->evaluate( evaluations.nodes[ i ], blackbox_values, 
                          user_data ); 

    // add evaluations to blackbox data
    for (int j = 0; j < nb_constraints+1; ++j) {
      if ( stochastic_optimization )
        evaluations.noise[j].push_back( blackbox_noise.at(j) );
      evaluations.values[j].push_back( blackbox_values.at(j) );
    }  
  }
 
  assert ( evaluations.nodes.size() == evaluations.values[0].size() );

  if ( stochastic_optimization ) 
    gaussian_processes.smooth_data ( evaluations );

  write_to_file();

  return;
}  
//--------------------------------------------------------------------------------

//--------------------------------------------------------------------------------
template<class TSurrogateModel, class TBasisForSurrogateModel>
bool NOWPAC<TSurrogateModel, TBasisForSurrogateModel>::last_point_is_feasible ( ) 
{
  bool point_is_feasible = true;

  bool update_tr = true;

  for (int i = 0; i < nb_constraints; ++i ) {
   tmp_dbl = evaluations.values[i+1].at( evaluations.best_index );
    if ( tmp_dbl < 0e0 ) tmp_dbl = 0e0;
    if ( evaluations.values[i+1].back() > tmp_dbl ) {
//      tmp_dbl = pow( diff_norm( x_trial, evaluations.nodes[ evaluations.best_index ] ) / delta, 2e0 );
      point_is_feasible = false;
      inner_boundary_path_constants.at(i) += 1e0;
      if ( inner_boundary_path_constants.at(i) > max_inner_boundary_path_constants.at(i) )
        inner_boundary_path_constants.at(i) = max_inner_boundary_path_constants.at(i);
/*
      inner_boundary_path_constants.at(i) = ((evaluations.values[i+1].back()-
       0e0*evaluations.values[i+1].at(evaluations.best_index))/
       pow(diff_norm(x_trial, evaluations.nodes[evaluations.best_index]),2e0))*2e0;//1e0 + inner_boundary_path_constants.at(i);
      
      if ( inner_boundary_path_constants.at(i) > 1e1  ) { 
//        if ( update_tr )
//          update_trustregion( theta );
        inner_boundary_path_constants.at(i) = 1e1;
        update_tr = false;
      }
*/
//      inner_boundary_path_constants.at(i) = tmp_dbl * max_inner_boundary_path_constants.at(i);
    } //else {}
//      inner_boundary_path_constants.at(i) = 0e0;
  }

/*
      tmp_dbl = pow( diff_norm( x_trial, evaluations.nodes[ evaluations.best_index ] ) / delta, 2e0 );
      for (int i = 0; i < nb_constraints; ++i) {
//        inner_boundary_path_constants.at(i) = max_inner_boundary_path_constants.at(i);
//        inner_boundary_path_constants.at(i) = tmp_dbl * inner_boundary_path_constants.at(i);
        inner_boundary_path_constants.at(i) = tmp_dbl * max_inner_boundary_path_constants.at(i);
      }
*/
//      tmp_dbl = pow( diff_norm( x_trial, evaluations.nodes[ evaluations.best_index ] ) / delta, 2e0 );
//      std::cout << " step size scale = " << sqrt(tmp_dbl) << std::endl; 


  if ( !point_is_feasible ) {
    std::cout << "Point is not feasible" << std::endl;
//    std::cout << tmp_dbl << std::endl;
    for ( int i = 0; i < nb_constraints; ++i )
      std::cout << inner_boundary_path_constants[i] << ", ";
    std::cout << std::endl;
    for ( int i = 0; i < nb_constraints; ++i )
      std::cout << evaluations.values[i+1].back() << ", ";
    std::cout << std::endl;
  }

  return point_is_feasible;
}
//--------------------------------------------------------------------------------

//--------------------------------------------------------------------------------
template<class TSurrogateModel, class TBasisForSurrogateModel>
void NOWPAC<TSurrogateModel, TBasisForSurrogateModel>::update_surrogate_models ( )
{
  bool best_index_okay = false;
  for (int i = 0; i < evaluations.active_index.size(); ++i) {
    if ( evaluations.active_index[i] == evaluations.best_index) {
      best_index_okay = true;
      break;
    }    
  }

  assert( best_index_okay );

  surrogate_basis.compute_basis_coefficients ( evaluations.get_scaled_active_nodes( 
    evaluations.nodes[ evaluations.best_index ], delta) );

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
      }
    }
  }

 if ( tmp_dbl >= 0e0 ) {
   if ( max_noise > 2e0*tmp_dbl ) max_noise = 2e0*tmp_dbl;
 } 
 // max_noise = pow(sqrt(1e3)*max_noise/2e0, 2e0);

 if ( check_for_noise && scaling_factor >= 1e0) this->reset_noise_detection();

  delta *= scaling_factor;
//  std::cout << std::endl << "------------------------- " << std::endl;
//  std::cout << "MAXNOISE " << max_noise << " sqrt = " << sqrt( max_noise ) << std::endl;
//  std::cout <<  "------------------------- " << std::endl;
  if ( stochastic_optimization ) {
    double ar_tmp = acceptance_ratio;
    if ( ar_tmp < 0e0 ) ar_tmp = -ar_tmp;
    if (ar_tmp < 1e0) ar_tmp = 1e0;
    if (ar_tmp > 2e0) ar_tmp = 2e0;
      ar_tmp = 1e0;//sqrt(2.0);
    if ( delta < sqrt(1e0*max_noise)*1e0*ar_tmp  ) 
      delta = sqrt(1e0*max_noise) * 1e0 * ar_tmp; 
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
  if ( evaluations.active_index.size( ) < evaluations.max_nb_nodes &&
       evaluations.active_index[ replace_node_index] != evaluations.best_index ) {
    evaluations.active_index.push_back ( evaluations.nodes.size()-1 );
  } else {
    evaluations.active_index[ replace_node_index ] = evaluations.nodes.size()-1;
  }

  if ( stochastic_optimization )
    gaussian_processes.smooth_data ( evaluations );
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
    fprintf(output_file, "\n");
    fflush(output_file);

}
//--------------------------------------------------------------------------------



//--------------------------------------------------------------------------------
//XXX--------------------------------------------------------------------------------
template<class TSurrogateModel, class TBasisForSurrogateModel>
void NOWPAC<TSurrogateModel, TBasisForSurrogateModel>::output_for_plotting ( ) 
{
  return;
  std::vector<double> x_loc(dim);
  std::vector<double> fvals(nb_constraints+1);
  std::ofstream outputfile ( "surrogate_data.dat" );
  int xcoord = 0;
  int ycoord = 1;
  if ( outputfile.is_open( ) ) {
    for ( int i = 0; i < dim; ++i)
      x_loc.at(i) = 0e0;
    for (double i = -1.0; i <= 1.0; i+=0.01) {
      x_loc.at(xcoord) = i;
      for (double j = -1.0; j < 1.0; j+=0.01) {
        x_loc.at(ycoord) = j;
        fvals.at(0) = surrogate_models[0].evaluate( x_loc );
        outputfile << x_loc.at(xcoord) << "; " << x_loc.at(ycoord) << "; " << fvals.at(0)<<"; ";
//        std::cout << " nb_constraints = " << nb_constraints << std::endl;
        for ( int k = 0; k < nb_constraints; ++k) {
          fvals.at(k+1) = surrogate_models[k+1].evaluate( x_loc );
          outputfile << fvals.at(k+1) << "; ";
        }
        outputfile << std::endl;
      }
    }
    outputfile.close( );
  } else std::cout << "Unable to open file." << std::endl;
  outputfile.open ( "data.dat" );
  if ( outputfile.is_open( ) ) {
    std::vector< std::vector<double> > outputnodes;
    outputfile << delta << "; " << evaluations.active_index.size() << "; ";
    for ( int i = 0; i < dim-2; ++i)     
      outputfile << "0 ;";
    outputfile << std::endl;
    outputnodes = evaluations.get_scaled_active_nodes( evaluations.nodes.at(evaluations.best_index), delta); 
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
  } else std::cout << "Unable to open file." << std::endl;

  //std::cin >> EXIT_FLAG;
  system("read -n1 -r -p \"Press any key to continue...\"");
  return;
}
//XXX--------------------------------------------------------------------------------
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
  for (int i = 0; i < nb_constraints+1; ++i )
    surrogate_models.push_back ( surrogate_model_prototype );

  this->initialize_noise_detection( noise_observation_span, nb_allowed_noisy_iterations);

  ImprovePoisedness surrogate_nodes ( surrogate_basis, threshold_for_poisedness_constant,
                                      evaluations.max_nb_nodes, delta, verbose );
  SubproblemOptimization<TSurrogateModel> surrogate_optimization (
    surrogate_models, delta, inner_boundary_path_constants);
  if ( !upper_bound_constraints.empty() ) 
    surrogate_optimization.set_upper_bounds ( upper_bound_constraints );
  if ( !lower_bound_constraints.empty() ) 
    surrogate_optimization.set_lower_bounds ( lower_bound_constraints );

  if ( stochastic_optimization )
    gaussian_processes.initialize ( dim, nb_constraints + 1, delta,
                                    update_at_evaluations, update_interval_length );

//  std::random_device rand_dev;
  int random_seed = 25041981;//rand_dev();
  std::mt19937 rand_generator(random_seed);
  std::normal_distribution<double> norm_dis(0e0,2e-1);
  //std::uniform_real_distribution<double> norm_dis(-2e0,2e0);

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
  for (int i = 0; i < dim; ++i ) {
    x_trial = x;
    x_trial.at(i) += delta;
    blackbox_evaluator( x_trial, true );
  }
  if ( verbose == 3 ) { std::cout << "Building initial models ... "; }
  update_surrogate_models( );
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
        output_for_plotting( ) ;
        update_trustregion( omega );
        update_surrogate_models( ); //xxx
        if ( EXIT_FLAG != NOEXIT ) break;
        if (verbose == 3) {
          std::cout << " Adjusting trust-region radius to " << delta << std::endl;
        }
        surrogate_nodes.improve_poisedness ( evaluations.best_index, evaluations );
        if ( stochastic_optimization ) {
          for (int i = 0; i < dim; ++i )
            x_trial.at(i) = evaluations.nodes[ evaluations.best_index ].at( i ) +
                         delta * norm_dis( rand_generator ) * 1e0;
          evaluations.nodes.push_back( x_trial );
        }
        blackbox_evaluator( );
        if ( EXIT_FLAG != NOEXIT ) break;
        update_surrogate_models( );
        if ( check_for_noise ) this->detect_noise( EXIT_FLAG );
        x_trial = evaluations.nodes[ evaluations.best_index ];
        criticality_value = surrogate_optimization.compute_criticality_measure( x_trial );

/*
      tmp_dbl = pow( diff_norm( x_trial, evaluations.nodes[ evaluations.best_index ] ) / delta, 2e0 );
      for (int i = 0; i < nb_constraints; ++i) {
//        inner_boundary_path_constants.at(i) = max_inner_boundary_path_constants.at(i);
        inner_boundary_path_constants.at(i) = tmp_dbl * inner_boundary_path_constants.at(i);
//        inner_boundary_path_constants.at(i) = tmp_dbl * max_inner_boundary_path_constants.at(i);
      }
      std::cout << " step size scale = " << sqrt(tmp_dbl) << std::endl; 
*/
        output_for_plotting( ) ;
        if (verbose == 3) {
          std::cout << " Value of criticality measure is " << criticality_value << std::endl;
          std::cout << " -----------------------------------" << std::endl << std::flush;
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

      trial_model_value = surrogate_optimization.compute_trial_point ( x_trial );

   //XXX
      if ( trial_model_value > evaluations.values[0].at( evaluations.best_index )) {
//        trial_model_value = evaluations.values[0].at( evaluations.best_index );
   //XXX
        std::cout << std::setprecision(16) << " difference = " << trial_model_value - evaluations.values[0].at( evaluations.best_index ) << std::endl;
        std::cout << " trial model value = " << trial_model_value << std::endl;
//        tmp_dbl1 = surrogate_models[0].evaluate( evaluations.transform( 
//         evaluations.nodes[ evaluations.best_index ]) );
        std::cout << " trial model value = " << tmp_dbl1 << std::endl;
        std::cout << " exact evaluation  = " << evaluations.values[0].at( evaluations.best_index ) << std::endl;
//        assert(false);
      }
      
      if ( verbose == 3 ) { std::cout << "done" << std::endl << std::flush; }
//      trial_model_value = surrogate_models[0].evaluate( evaluations.transform(x_trial) );


      /*-------------------------------------------------------------------------*/
      /*- STEP 3 - STEP 3 - STEP 3 - STEP 3 - STEP 3 - STEP 3 - STEP 3 - STEP 3 -*/
      /*-------------------------------------------------------------------------*/
      if ( verbose == 3 ) { std::cout << "Checking feasibility .... " << std::flush; }
      blackbox_evaluator( x_trial, false );   
      if ( verbose == 3 ) { std::cout << "done" << std::endl; }
      if ( !last_point_is_feasible( ) ) {
        if ( verbose == 3 ) { std::cout << " Feasibility violated" << std::endl << std::flush; }
//        for (int i = 0; i < nb_constraints; ++i) {
//          inner_boundary_path_constants.at(i) *= 2e0;  
 //         if ( inner_boundary_path_constants.at(i) > max_inner_boundary_path_constants.at(i) )
//            inner_boundary_path_constants.at(i) = 1e1* inner_boundary_path_constants.at(i);
//        }


        tmp_dbl = 1e0;
        for ( int i = 0; i < evaluations.active_index.size(); ++i ) {
           if ( this->diff_norm ( evaluations.nodes.at( evaluations.active_index.at(i) ),
                                  x_trial ) < 1e-2 * delta ) {
             tmp_dbl = -1e0;
             break;
           }
        }
        if ( tmp_dbl > 0e0 ) {
          replace_node_index = surrogate_nodes.replace_node( evaluations.best_index, 
                                                             evaluations, x_trial );
          add_trial_node( );
        }
        output_for_plotting( ) ;
        update_trustregion( theta );
        update_surrogate_models( );

        
        if ( EXIT_FLAG != NOEXIT ) break;
        surrogate_nodes.improve_poisedness (evaluations.best_index, evaluations );
        if ( stochastic_optimization ) {
          for (int i = 0; i < dim; ++i )
            x_trial.at(i) = evaluations.nodes[ evaluations.best_index ].at( i ) +
                            delta * norm_dis( rand_generator );
          evaluations.nodes.push_back( x_trial );
        }
        blackbox_evaluator( );
        if ( EXIT_FLAG != NOEXIT ) break;
//        update_trustregion( theta );
        update_surrogate_models( );

        if ( check_for_noise ) this->detect_noise( EXIT_FLAG );

//        output_for_plotting ( );
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


      output_for_plotting();

      stepsize[0] = ( ( this->diff_norm( x_trial, evaluations.nodes[ evaluations.best_index ] ) / delta ) 
                      + stepsize[1])/ 2e0;
      stepsize[1] = this->diff_norm( x_trial, evaluations.nodes[ evaluations.best_index ] ) / delta;
      for (int i = 0; i < nb_constraints; ++i) 
        inner_boundary_path_constants.at(i) = pow(stepsize[0], 1e0) * 
                                              max_inner_boundary_path_constants.at(i);
//        inner_boundary_path_constants.at(i) *= pow(stepsize[0], 2e0);
//      tmp_dbl = pow( diff_norm( x_trial, evaluations.nodes[ evaluations.best_index ] ) / delta, 2e0 )
//      for (int i = 0; i < nb_constraints; ++i) {
//        inner_boundary_path_constants.at(i) = max_inner_boundary_path_constants.at(i);
//        inner_boundary_path_constants.at(i) = tmp_dbl * inner_boundary_path_constants.at(i);
//        inner_boundary_path_constants.at(i) = tmp_dbl * max_inner_boundary_path_constants.at(i);
//      }

      if ( verbose == 3 ) { std::cout << std::endl; }
      if ( verbose >= 2 ) { 
        std::cout << "*****************************************" << std::endl; 
      }
      fflush(stdout);
      if ( acceptance_ratio >= eta_1 && acceptance_ratio < 2e0 ) {
        if ( verbose >= 2 ) { std::cout << " Step successful" << std::endl << std::flush; }
        update_trustregion( gamma_inc );
      } 
      fflush(stdout);
      if ( (acceptance_ratio >= eta_0 && acceptance_ratio < eta_1) || acceptance_ratio >= 2e0 ) {
        if ( verbose >= 2 ) { std::cout << " Step acceptable" << std::endl << std::flush; }
      }

      if ( acceptance_ratio >= eta_0 ) {
        replace_node_index = surrogate_nodes.replace_node( -1, 
                                                           evaluations, x_trial );
        add_trial_node( );
        evaluations.best_index = evaluations.nodes.size()-1;
        update_surrogate_models( );
      }
      if ( acceptance_ratio < eta_0 ) {
        if ( verbose >= 2 ) { std::cout << " Step rejected" << std::endl << std::flush; }
        tmp_dbl = 1e0;
        for ( int i = 0; i < evaluations.active_index.size(); ++i ) {
           if ( this->diff_norm ( evaluations.nodes.at( evaluations.active_index.at(i) ),
                                  evaluations.nodes.at( evaluations.nodes.size()-1 ) ) < 1e-2 * delta ) {
           //if ( diff_norm ( evaluations.nodes.at( evaluations.active_index.at(i) ),
           //                 x_trial ) < 1e-2 * delta ) {
             tmp_dbl = -1e0;
             break;
          }
        } 
        if ( tmp_dbl > 0e0 ) { // XXX --- XXX
          replace_node_index = surrogate_nodes.replace_node( evaluations.best_index, 
                                                             evaluations, x_trial );
          assert ( evaluations.active_index[replace_node_index] != evaluations.best_index);
          add_trial_node( );
          update_surrogate_models( );
        }
        if ( stochastic_optimization ) {
          for ( int i = 0; i < dim; ++i )
            x_trial.at(i) = evaluations.nodes[ evaluations.best_index ].at( i ) +
                            delta * norm_dis( rand_generator ) * 1e0;
          evaluations.nodes.push_back( x_trial );
        }
        blackbox_evaluator( );
        update_trustregion( gamma );
        if ( EXIT_FLAG == NOEXIT ) {
          update_surrogate_models( );
          surrogate_nodes.improve_poisedness( evaluations.best_index, evaluations );
          blackbox_evaluator( );
          if ( EXIT_FLAG != NOEXIT ) {
            std::cout << "*****************************************" << std::endl << std::flush;
            break;
          }
          update_surrogate_models( );
          if ( check_for_noise ) this->detect_noise( EXIT_FLAG );
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
        std::cout << "*****************************************" << std::endl << std::endl << std::flush;
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
    if ( EXIT_FLAG == -2 )
      std::cout << " Noise detected" << std::endl;
    if ( EXIT_FLAG == -4 )
      std::cout << " Inconsistent parameter" << std::endl;
    std::cout << "*********************************************" << std::endl << std::endl << std::flush;
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
        fvals(0) = surrogate_models[0].evaluate( evaluations.transform(x_loc) );
        fvals(1) = surrogate_models[1].evaluate( evaluations.transform(x_loc) );
        fvals(2) = surrogate_models[2].evaluate( evaluations.transform(x_loc) );
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
