#include "LegendreBasisForMinimumFrobeniusNormModel.hpp"
#include <iostream>
#include <fstream>
#include <iomanip>


//--------------------------------------------------------------------------------
LegendreBasisForMinimumFrobeniusNormModel::LegendreBasisForMinimumFrobeniusNormModel 
  ( int dim_input ) :
  BasisForSurrogateModelBaseClass ( dim_input ),
  QuadraticLegendre ( dim_input ) 
{
  nb_basis_functions = ( dim_input * ( dim_input + 3 ) + 2 ) / 2;
  nb_nodes = 0;
}
//--------------------------------------------------------------------------------

//--------------------------------------------------------------------------------
void LegendreBasisForMinimumFrobeniusNormModel::set_nb_nodes( int nb_nodes_input ) {
  if ( nb_nodes == nb_nodes_input ) return;
  nb_nodes = nb_nodes_input;

  basis_values.clear();
  basis_values.resize ( nb_nodes );
  A_sysmat = Eigen::MatrixXd::Zero(nb_basis_functions + nb_nodes, 
                                   nb_basis_functions + nb_nodes);

  for ( int i = 0; i < nb_basis_functions; ++i)
    if (i > BasisForSurrogateModelBaseClass::dim) A_sysmat(i, i) = 1e0;
  F_rhsmat = Eigen::MatrixXd::Zero(nb_basis_functions + nb_nodes, nb_nodes);

  for ( int i = 0; i < nb_nodes; ++i ) 
    F_rhsmat(nb_basis_functions + i, i) = 1e0;
  basis_constants.clear();
  basis_constants.resize( nb_nodes );
  basis_gradients.clear();
  basis_gradients.resize( nb_nodes );
  basis_Hessians.clear();
  basis_Hessians.resize( nb_nodes );
  basis_coefficients.clear();
  basis_coefficients.resize( nb_nodes );


  for ( int i = 0; i < nb_nodes; ++i ) {
    basis_coefficients[i] = Eigen::VectorXd::Zero( nb_basis_functions ) ;
    basis_gradients[i].resize( BasisForSurrogateModelBaseClass::dim );
    basis_Hessians[i].resize( BasisForSurrogateModelBaseClass::dim );
    for ( int j = 0; j < BasisForSurrogateModelBaseClass::dim; ++j) 
      basis_Hessians[i][j].resize( BasisForSurrogateModelBaseClass::dim );
  }

  return;
}
//--------------------------------------------------------------------------------


//--------------------------------------------------------------------------------
void LegendreBasisForMinimumFrobeniusNormModel::compute_basis_coefficients ( 
  std::vector< std::vector<double> > const &nodes )
{

  set_nb_nodes ( nodes.size( ) );


  // system matrix for computing coeffs of Lagrange interpolation models
  for ( int i = 0; i < nb_basis_functions; ++i ) {    
    for ( int j = 0; j < nb_nodes; ++j ) {
      A_sysmat(i, j+nb_basis_functions) = evaluate_basis( i, nodes[j] );
      A_sysmat(j+nb_basis_functions, i) = A_sysmat(i, j+nb_basis_functions);
    }
  }    


  S_coeffsolve = A_sysmat.colPivHouseholderQr().solve(F_rhsmat);

  if ( (A_sysmat * S_coeffsolve - F_rhsmat).norm() > 1e-5) {
    for ( int i = 0; i < nb_nodes; ++i ) {
      for ( int j = 0; j < BasisForSurrogateModelBaseClass::dim; ++j )
        S_coeffsolve(BasisForSurrogateModelBaseClass::dim +j+1, i) = 0e0;
      if ( norm( nodes[i]) < 1e-16 ) 
        S_coeffsolve(0, i) = 1e0;
      else
        S_coeffsolve(0, i) = 0e0;
    }
  }

  for ( int i = 0; i < nb_nodes; ++i ) {
    basis_coefficients[i] = S_coeffsolve.block(0,0,nb_basis_functions, nb_nodes).col(i);  
    compute_mat_vec_representation ( i );
  } 





  return;
}
//--------------------------------------------------------------------------------


//--------------------------------------------------------------------------------
double &LegendreBasisForMinimumFrobeniusNormModel::value ( int basis_number ) 
{
  return basis_constants[ basis_number ];
}
//--------------------------------------------------------------------------------

//--------------------------------------------------------------------------------
std::vector<double> &LegendreBasisForMinimumFrobeniusNormModel::gradient (int basis_number)
{ 
  return basis_gradients.at( basis_number );
}
//--------------------------------------------------------------------------------

//--------------------------------------------------------------------------------
std::vector< std::vector<double> > &LegendreBasisForMinimumFrobeniusNormModel::hessian ( 
  int basis_number ) 
{
  return basis_Hessians[ basis_number ];
}
//--------------------------------------------------------------------------------


//--------------------------------------------------------------------------------
void LegendreBasisForMinimumFrobeniusNormModel::compute_mat_vec_representation ( int basis_number )
// computes the representation m(x) = c + g.dot(x) + 0.5*x.dot(H*x) in scaled form
// x_best = 0 and delta = 1;
{
  counter = 0;
  basis_constants[basis_number] = basis_coefficients[basis_number](0);
  for (int j = 0; j < BasisForSurrogateModelBaseClass::dim; ++j) {
    basis_constants[basis_number] -= 0.5*basis_coefficients[basis_number]( 
                                         j+1+BasisForSurrogateModelBaseClass::dim);
    basis_gradients[basis_number].at(j) = basis_coefficients[ basis_number ]( j + 1 );
    basis_Hessians[basis_number].at(j).at(j) = 3e0*basis_coefficients[ basis_number ]
                                               (j+1+BasisForSurrogateModelBaseClass::dim);
    for (int k = j+1; k < BasisForSurrogateModelBaseClass::dim; ++k) {
      basis_Hessians[basis_number].at(j).at(k) = basis_coefficients[ basis_number ]
                                                 (2*BasisForSurrogateModelBaseClass::dim+1+counter);
      basis_Hessians[basis_number].at(k).at(j) = basis_coefficients[ basis_number ]
                                                 (2*BasisForSurrogateModelBaseClass::dim+1+counter);
      counter++;
    }
  }
  
  return;
}
//--------------------------------------------------------------------------------


//--------------------------------------------------------------------------------
std::vector<double> &LegendreBasisForMinimumFrobeniusNormModel::evaluate ( 
  std::vector<double> const &x ) 
{  
 
  //assert( nb_nodes == basis_coefficients.size());

  for ( int i = 0; i < nb_nodes; ++i ) {
    basis_values.at( i ) = 0e0;
    for ( int j = 0; j < nb_basis_functions; ++j ) {
      basis_values.at( i ) += basis_coefficients[ i ]( j ) * evaluate_basis ( j, x );
    }
  }

  return basis_values;
}
//--------------------------------------------------------------------------------


//--------------------------------------------------------------------------------
double LegendreBasisForMinimumFrobeniusNormModel::evaluate (
  std::vector<double> const &x, int basis_number)
{ 

  double basis_value = 0e0;
  for ( int j = 0; j < nb_basis_functions; ++j ) {
    basis_value += basis_coefficients[ basis_number ]( j ) * 
                   evaluate_basis ( j, x );
  }
  return basis_value;
}
//--------------------------------------------------------------------------------


