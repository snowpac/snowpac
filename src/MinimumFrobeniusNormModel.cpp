#include "MinimumFrobeniusNormModel.hpp"
#include <iostream>

//--------------------------------------------------------------------------------
MinimumFrobeniusNormModel::MinimumFrobeniusNormModel ( 
                           BasisForMinimumFrobeniusNormModel &basis_input) :
                           SurrogateModelBaseClass ( basis_input )
{
  model_gradient.resize( basis_input.dimension ( ) );
}
//--------------------------------------------------------------------------------

//--------------------------------------------------------------------------------
std::vector<double> &MinimumFrobeniusNormModel::gradient (
  std::vector<double> const &x ) 
{
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
