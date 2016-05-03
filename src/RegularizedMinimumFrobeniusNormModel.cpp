#include "RegularizedMinimumFrobeniusNormModel.hpp"
#include <iostream>

//--------------------------------------------------------------------------------
RegularizedMinimumFrobeniusNormModel::RegularizedMinimumFrobeniusNormModel ( 
                           BasisForMinimumFrobeniusNormModel &basis_input) :
                           SurrogateModelBaseClass ( basis_input )
{
  model_gradient.resize( basis_input.dimension ( ) );
}
//--------------------------------------------------------------------------------

//--------------------------------------------------------------------------------
std::vector<double> &RegularizedMinimumFrobeniusNormModel::gradient (
  std::vector<double> const &x ) 
{
 // basis->compute_basis_gradients ( x );
  scale( function_values.at(0), basis->gradient(x,0), model_gradient );
  for (int i = 1; i < size; i++)
    add( function_values.at(i), basis->gradient(x,i), model_gradient );
  return model_gradient;
}
//--------------------------------------------------------------------------------

//--------------------------------------------------------------------------------
double RegularizedMinimumFrobeniusNormModel::evaluate (
  std::vector<double> const &x )
{
  return dot_product( function_values, basis->evaluate( x ) );
}
//--------------------------------------------------------------------------------

//--------------------------------------------------------------------------------
void RegularizedMinimumFrobeniusNormModel::set_function_values( 
  std::vector<double> const &values, std::vector<double> const &noise,
  std::vector<int> const &surrogate_nodes_index )
{ 
  size = surrogate_nodes_index.size();
  function_values.resize( size );
  noise_values.resize( size );
  for (int i = 0; i < size; ++i) {
    function_values.at(i) = values.at( surrogate_nodes_index[i] );
    noise_values.at(i) = noise.at( surrogate_nodes_index[i] );
  }
  regularize_coefficients( );
  return;
}
//--------------------------------------------------------------------------------

//--------------------------------------------------------------------------------
void RegularizedMinimumFrobeniusNormModel::regularize_coefficients( ) 
{

  lb.clear();
  ub.clear();
  parameters.clear();
  for ( size_t i = 0; i < function_values.size( ); i++ ) {
    lb.push_back( function_values.at(i) - 5e-1*noise_values.at( i ) );
    ub.push_back( function_values.at(i) + 5e-1*noise_values.at( i ) );
    parameters.push_back( function_values.at( i ) );
  }

  nlopt::opt opt(nlopt::LN_BOBYQA, function_values.size( ) );
  opt.set_lower_bounds ( lb );
  opt.set_upper_bounds ( ub );
  opt.set_min_objective( regularization_objective, basis );
  opt.set_xtol_abs ( 1e-6 );
  opt.set_xtol_rel ( 1e-6 );
//  opt.set_maxtime( 1.5);
  try {
    opt.optimize ( parameters, res );
    for ( size_t i = 0; i < function_values.size( ); i++ )
      function_values.at( i ) = parameters [ i ];
  } catch(...) {}
  return;
}
//--------------------------------------------------------------------------------

//--------------------------------------------------------------------------------
double RegularizedMinimumFrobeniusNormModel::regularization_objective(
  std::vector<double> const &x, std::vector<double> &grad, void *data)
{

  BasisForMinimumFrobeniusNormModel *d = 
    reinterpret_cast<BasisForMinimumFrobeniusNormModel*>(data); 

  Eigen::VectorXd g( d->dimension() );
  Eigen::MatrixXd H( d->dimension(), d->dimension() );
  Eigen::MatrixXd H_total (d->dimension(), d->dimension());
  H_total = Eigen::MatrixXd::Zero( d->dimension(), d->dimension() );
  double objective = 0e0;

  for ( size_t i = 0; i < x.size(); i++ ) {
    d->get_mat_vec_representation( i, g, H );
    H_total += x[i] * H; 
  }

  for ( int i = 0; i < d->dimension(); i++ ) {
    for ( int j = 0; j < d->dimension(); j++ ) {
      objective += pow( H_total(i, j), 2e0);
    }
  }  

  return objective;
}
//--------------------------------------------------------------------------------
