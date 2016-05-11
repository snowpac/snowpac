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
  for (int i = 1; i < size; i++)
    add( function_values.at(i), basis->gradient(x,i), model_gradient );
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
  std::vector<double> const &values, std::vector<double> const &noise,
  std::vector<int> const &surrogate_nodes_index, int best_index )
{ 
  size = surrogate_nodes_index.size();
  function_values.resize( size );
  for (int i = 0; i < size; i++)
    function_values.at(i) = values.at( surrogate_nodes_index[i] );
  return;
}
//--------------------------------------------------------------------------------
