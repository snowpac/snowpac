#include "MinimumFrobeniusNormModel.hpp"
#include <iostream>

//--------------------------------------------------------------------------------
MinimumFrobeniusNormModel::MinimumFrobeniusNormModel ( 
                           BasisForMinimumFrobeniusNormModel &basis_input) :
                           SurrogateModelBaseClass ( basis_input )
{
  dim = basis_input.dimension();
  model_gradient.resize( dim );
  model_hessian.resize( dim );
  for ( int i = 0; i < dim; ++i )
    model_hessian[i].resize( dim );
}
//--------------------------------------------------------------------------------

//--------------------------------------------------------------------------------
std::vector<double> &MinimumFrobeniusNormModel::gradient (
  std::vector<double> const &x ) 
{
  assert( size == function_values.size() );
  scale( function_values.at(0), basis->gradient(x,0), model_gradient );
  for ( unsigned int i = 1; i < size; ++i)
    add( function_values.at(i), basis->gradient(x,i), model_gradient );
  return model_gradient;
}
//--------------------------------------------------------------------------------

//--------------------------------------------------------------------------------
std::vector<double> &MinimumFrobeniusNormModel::gradient ( ) 
{
  scale( function_values.at(0), basis->gradient(0), model_gradient );
  for ( unsigned int i = 1; i < size; ++i)
    add( function_values.at(i), basis->gradient(i), model_gradient );
  return model_gradient;
}
//--------------------------------------------------------------------------------

//--------------------------------------------------------------------------------
std::vector< std::vector<double> > &MinimumFrobeniusNormModel::hessian (
  std::vector<double> const &x ) 
{
  return hessian( );
}
//--------------------------------------------------------------------------------

//--------------------------------------------------------------------------------
std::vector< std::vector<double> > &MinimumFrobeniusNormModel::hessian ( ) 
{
  for ( unsigned int j = 0; j < dim; ++j ) {
  scale( function_values.at(0), basis->hessian(0), model_hessian );
  for ( unsigned int i = 1; i < size; ++i)
    add( function_values.at(i), basis->hessian(i), model_hessian );
  }
  return model_hessian;
}
//--------------------------------------------------------------------------------

//--------------------------------------------------------------------------------
double MinimumFrobeniusNormModel::evaluate (
  std::vector<double> const &x ) 
{
  return dot_product( function_values, basis->evaluate( x ) );
}
//--------------------------------------------------------------------------------

//--------------------------------------------------------------------------------
void MinimumFrobeniusNormModel::set_function_values( 
  std::vector<double> const &values )
{ 
  size = values.size();
  function_values = values;
//  function_values.resize( values.size() );
//  for ( unsigned int i = 0; i < size; ++i )
//    function_values.at(i) = values.at( i );
  return;
}
//--------------------------------------------------------------------------------
