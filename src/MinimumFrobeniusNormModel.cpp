#include "MinimumFrobeniusNormModel.hpp"
#include <iostream>

//--------------------------------------------------------------------------------
MinimumFrobeniusNormModel::MinimumFrobeniusNormModel ( 
                           BasisForSurrogateModelBaseClass &basis_input) :
                           SurrogateModelBaseClass ( basis_input )
{
  dim = basis_input.dimension();
  matvecproduct.resize( dim );
  model_gradient.resize( dim );
  model_hessian.resize( dim );
  for ( int i = 0; i < dim; ++i )
    model_hessian[i].resize( dim );
}
//--------------------------------------------------------------------------------

//--------------------------------------------------------------------------------
std::vector<double> &MinimumFrobeniusNormModel::gradient ( ) 
{
  return model_gradient;
}
//--------------------------------------------------------------------------------

//--------------------------------------------------------------------------------
std::vector< std::vector<double> > &MinimumFrobeniusNormModel::hessian ( ) 
{
  return model_hessian;
}
//--------------------------------------------------------------------------------

//--------------------------------------------------------------------------------
double MinimumFrobeniusNormModel::evaluate (  std::vector<double> const &x ) 
{
  double value = model_constant;
  value += VectorOperations::dot_product( x, model_gradient );
  mat_vec_product( model_hessian, x, matvecproduct );
  value += 0.5*VectorOperations::dot_product( x, matvecproduct );
  return value;
}
//--------------------------------------------------------------------------------

//--------------------------------------------------------------------------------
void MinimumFrobeniusNormModel::set_function_values( 
  std::vector<double> const &values )
{ 
  size = values.size();

  model_constant = values.at(0) * basis->value( 0 );
  scale( values.at(0), basis->gradient(0), model_gradient );
  scale( values.at(0), basis->hessian(0), model_hessian );
  for (int i = 1; i < size; ++i) {
    model_constant += values.at(i) * basis->value(i); 
    add( values.at(i), basis->gradient(i), model_gradient );
    add( values.at(i), basis->hessian(i), model_hessian );
  }

/*
  std::cout << " ---------------------- " << std::endl;
  std::cout << " ---------------------- " << std::endl;
  for ( int i = 0; i < size; ++i)
    std::cout << "f["<<i<<"]=" << values[i] << std::endl;
  std::cout << " ---------------------- " << std::endl;
  std::cout << "c = " << model_constant << std::endl;
  for ( int i = 0; i < dim; ++i)
    std::cout << "g[" <<i << "] = " << model_gradient[i] << std::endl;
  for ( int i = 0; i < dim; ++i) {
    for ( int j = 0; j < dim; ++j )
      std::cout << "H[" <<i << "]["<< j<<"] = " << model_hessian[i][j] << std::endl;
  }
  std::cout << " ---------------------- " << std::endl;
  std::cout << " ---------------------- " << std::endl;
*/
//  function_values.resize( values.size() );
//  for ( unsigned int i = 0; i < size; ++i )
//    function_values.at(i) = values.at( i );
  return;
}
//--------------------------------------------------------------------------------
