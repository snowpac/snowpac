#include "RegularizedMinimumFrobeniusNormModel.hpp"
#include <iostream>

//--------------------------------------------------------------------------------
RegularizedMinimumFrobeniusNormModel::RegularizedMinimumFrobeniusNormModel ( 
                           BasisForMinimumFrobeniusNormModel &basis_input) :
                           SurrogateModelBaseClass ( basis_input )
{
  rd.dim = basis_input.dimension();
  rd.g.resize(rd.dim);
  rd.H.resize(rd.dim);
  rd.g_total.resize(rd.dim);
  rd.H_total.resize(rd.dim);
  for ( unsigned int j = 0; j < rd.dim; ++j ) {
    rd.H[j].resize(j+1);
    rd.H_total[j].resize(j+1);
  }
  rd.basis = &basis_input;
  rd.vo = &vo;
  model_gradient.resize( basis_input.dimension ( ) );
}
//--------------------------------------------------------------------------------

//--------------------------------------------------------------------------------
std::vector<double> &RegularizedMinimumFrobeniusNormModel::gradient (
  std::vector<double> const &x ) 
{
 // basis->compute_basis_gradients ( x );
  scale( function_values.at(0), basis->gradient(x,0), model_gradient );
  for (unsigned int i = 1; i < size; ++i)
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
  std::vector<int> const &surrogate_nodes_index, int best_index )
{ 
  size = surrogate_nodes_index.size();
  function_values.resize( size );
  noise_values.resize( size );
  rd.best_index = -1;
  for (unsigned int i = 0; i < size; ++i) {
    if ( surrogate_nodes_index[i] == best_index ) rd.best_index = i;
    function_values.at(i) = values.at( surrogate_nodes_index[i] );
    noise_values.at(i) = noise.at( surrogate_nodes_index[i] );
  }
  assert ( rd.best_index >= 0 ); 
  double fval = values.at( best_index );
  regularize_coefficients( );
  fval -= function_values.at( rd.best_index );
  for (unsigned int i = 0; i < size; ++i) {
    function_values.at( i ) += fval;
  }
//  function_values.at( rd.best_index ) = fval;
  return;
}
//--------------------------------------------------------------------------------

//--------------------------------------------------------------------------------
void RegularizedMinimumFrobeniusNormModel::regularize_coefficients( ) 
{
  lb.clear();
  ub.clear();
  //parameters.clear();
  for ( unsigned int i = 0; i < function_values.size( ); ++i ) {
    lb.push_back( function_values.at(i) - 1e-1*noise_values.at( i ) );
    ub.push_back( function_values.at(i) + 1e-1*noise_values.at( i ) );
//    parameters.push_back( function_values.at( i ) );
  }

  nlopt::opt opt(nlopt::LN_BOBYQA, function_values.size( ) );
  opt.set_lower_bounds ( lb );
  opt.set_upper_bounds ( ub );
//  opt.set_min_objective( regularization_objective, basis );
  opt.set_min_objective( regularization_objective, &rd );
  opt.set_xtol_abs ( 1e-6 );
  opt.set_xtol_rel ( 1e-6 );
  opt.set_maxtime( 1.5);
  try {
    opt.optimize ( function_values, res );
//    for ( unsigned int i = 0; i < function_values.size( ); ++i )
//      function_values.at( i ) = parameters [ i ];
  } catch(...) {}
  return;
}
//--------------------------------------------------------------------------------

//--------------------------------------------------------------------------------
double RegularizedMinimumFrobeniusNormModel::regularization_objective(
  std::vector<double> const &x, std::vector<double> &grad, void *data)
{

//  BasisForMinimumFrobeniusNormModel *d = 
//    reinterpret_cast<BasisForMinimumFrobeniusNormModel*>(data); 

  RegularizationData *d = reinterpret_cast<RegularizationData*>(data); 

  int size = x.size();

  double objective = 0e0;

  for ( unsigned int i = 0; i < size; ++i ) {
//    if ( i == d->best_index ) continue;
    d->g = d->basis->gradient( i );
    d->H = d->basis->hessian( i );
    if ( i == 0 ) {
      for ( unsigned int j = 0; j < d->dim; ++j ) {
        d->vo->scale( x[i], d->H[j], d->H_total[j] ); 
        d->vo->scale( x[i], d->g, d->g_total );
      }
    } else {
      for ( unsigned int j = 0; j < d->dim; ++j ) {
        d->vo->add( x[i], d->H[j], d->H_total[j] ); 
        d->vo->add( x[i], d->g, d->g_total );
      } 
    }
  }

  for ( unsigned int i = 0; i < d->dim; ++i ) {
    for ( unsigned int j = 0; j < i; ++j ) {
      objective += 2e0*pow( d->H_total.at(i).at(j), 2e0);
    }
    objective += pow( d->H_total.at(i).at(i), 2e0);
    objective += 1e-1*pow( d->g_total.at(i), 2e0);
  }  



  return objective;
}
//--------------------------------------------------------------------------------
