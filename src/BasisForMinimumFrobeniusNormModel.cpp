#include "BasisForMinimumFrobeniusNormModel.hpp"
#include <iostream>
#include <iomanip>

//--------------------------------------------------------------------------------
BasisForMinimumFrobeniusNormModel::BasisForMinimumFrobeniusNormModel ( int dim_input,
                                                                       double &delta_input ) : 
                                   BasisForSurrogateModelBaseClass ( dim_input ),
                                   QuadraticMonomial ( dim_input ) 
{
  clear_cache_values = true;
  clear_cache_gradients = true;
  cache_point_basis_values.resize( 0 );
  cache_point_basis_gradients.resize( 0 );
  scaled_node.resize( dim_input );
  best_node.resize( dim_input );
  tmp_vec.resize( dim_input );
  nb_basis_functions = ( dim_input * ( dim_input + 3 ) + 2 ) / 2;
  nb_nodes = 0;
  delta = &delta_input;
}
//--------------------------------------------------------------------------------

//--------------------------------------------------------------------------------
void BasisForMinimumFrobeniusNormModel::set_nb_nodes( int nb_nodes_input ) {
  if ( nb_nodes == nb_nodes_input ) return;
  nb_nodes = nb_nodes_input;

  basis_values.clear();
  basis_values.resize ( nb_nodes );
  A_sysmat = Eigen::MatrixXd::Zero(nb_basis_functions + nb_nodes, 
                                   nb_basis_functions + nb_nodes);

  for (int i = 0; i < nb_basis_functions; i++)
    if (i > BasisForSurrogateModelBaseClass::dim) A_sysmat(i, i) = 1e0;
  F_rhsmat = Eigen::MatrixXd::Zero(nb_basis_functions + nb_nodes, nb_nodes);

  for (int i = 0; i < nb_nodes; i++) F_rhsmat(nb_basis_functions + i, i) = 1e0;
//  if ( nb_nodes > basis_coefficients.size ( ) ) {
    basis_gradients.clear();
    basis_gradients.resize( nb_nodes );
    scaled_gradients.clear();
    scaled_gradients.resize( nb_nodes );
    scaled_Hessians.clear();
    scaled_Hessians.resize( nb_nodes );
    basis_coefficients.clear();
    basis_coefficients.resize( nb_nodes );
//    for (int i = basis_coefficients.size ( ); i < nb_nodes; i++) {
    for (int i = 0; i < nb_nodes; i++) {
      basis_gradients[i].resize( BasisForSurrogateModelBaseClass::dim );
      basis_coefficients[i] = Eigen::VectorXd::Zero( nb_basis_functions ) ;
      scaled_gradients[i].resize( BasisForSurrogateModelBaseClass::dim );
      scaled_Hessians[i].resize( BasisForSurrogateModelBaseClass::dim );
      for ( int j = 0; j < BasisForSurrogateModelBaseClass::dim; j++) {
        scaled_Hessians[i][j].resize( BasisForSurrogateModelBaseClass::dim );
      }
    }
//      scaled_gradients.push_back ( 
//        Eigen::VectorXd::Zero ( BasisForSurrogateModelBaseClass::dim ) );
//      scaled_Hessians.push_back ( 
//        Eigen::MatrixXd::Zero ( BasisForSurrogateModelBaseClass::dim,
//                                BasisForSurrogateModelBaseClass::dim ) );
//    }
//  } else if ( nb_nodes < basis_coefficients.size ( ) ) {
//    basis_coefficients.resize ( nb_nodes ); 
//    basis_gradients.resize ( nb_nodes );
//    scaled_gradients.resize ( nb_nodes );
//    scaled_Hessians.resize ( nb_nodes );
//  }
  //basis_values = Eigen::VectorXd::Zero( nb_nodes ); 

  return;
}
//--------------------------------------------------------------------------------


//--------------------------------------------------------------------------------
void BasisForMinimumFrobeniusNormModel::compute_basis_coefficients ( 
  BlackBoxData const &evaluations )
{
  set_nb_nodes ( evaluations.surrogate_nodes_index.size( ) );

  // system matrix for computing coeffs of Lagrange interpolation models
  for (int i = 0; i < nb_basis_functions; i++) {    
    for (int j = 0; j < nb_nodes; j++) {
      rescale( 1e0 / (*delta), 
               evaluations.nodes[ evaluations.surrogate_nodes_index[j] ], 
               evaluations.nodes[ evaluations.best_index ],
               scaled_node );
      A_sysmat(i, j+nb_basis_functions) = evaluate_monomial( i, scaled_node );
      A_sysmat(j+nb_basis_functions, i) = A_sysmat(i, j+nb_basis_functions);
    }
  }    

  // solve for coefficients
  tmp_dbl = pow( *delta, 2e0 );
  best_node = evaluations.nodes[evaluations.best_index];
  S_coeffsolve = A_sysmat.fullPivHouseholderQr().solve(F_rhsmat);
  for ( int i = 0; i < nb_nodes; i++ ) {
    basis_coefficients[i] = S_coeffsolve.block(0,0,nb_basis_functions, nb_nodes).col(i);  
    compute_mat_vec_representation ( i );
    for ( int j = 0; j < BasisForSurrogateModelBaseClass::dim; ++j )
      tmp_vec.at(j) =  VectorOperations::dot_product( scaled_Hessians[i][j], best_node ) / 
                       tmp_dbl;
//    tmp_vec = (scaled_Hessians[ i ] * best_node ) /  
//              pow( *delta, 2e0 );    //XXX

    basis_coefficients[i](0) = basis_coefficients[i](0) - 
      VectorOperations::dot_product( scaled_gradients[i],  best_node ) / (*delta) +
      5e-1*VectorOperations::dot_product( tmp_vec, best_node);
    for ( int j = 0; j < BasisForSurrogateModelBaseClass::dim; j++ )
      basis_coefficients[i]( j+1 ) = scaled_gradients[i].at( j ) / ( *delta ) - tmp_vec.at( j );
    for ( int j = BasisForSurrogateModelBaseClass::dim+1; 
          j < basis_coefficients[i].size(); j++ )
      basis_coefficients[i](j) /= tmp_dbl;
  } 
  
  clear_cache_values = true;
  clear_cache_gradients = true;

  return;
}
//--------------------------------------------------------------------------------

//--------------------------------------------------------------------------------
void BasisForMinimumFrobeniusNormModel::compute_gradients ( 
 std::vector<double> const &x )
{

  if ( cache_point_basis_gradients.size() > 0 ) {
    if ( diff_norm( cache_point_basis_gradients, x ) < 1e-20 &&
         !clear_cache_gradients ) 
      return;
  }
  for (int m = 0; m < nb_nodes; m++) {
    for (int i = 0; i < BasisForSurrogateModelBaseClass::dim; i++) {
      basis_gradients[m].at(i) = basis_coefficients[m](i+1);
      basis_gradients[m].at(i) += basis_coefficients[m](
                               BasisForSurrogateModelBaseClass::dim+1+i) *
                               x.at(i);
      counter = 0;
      for (int j = 0; j < BasisForSurrogateModelBaseClass::dim; j++) {
        for (int k = j+1; k < BasisForSurrogateModelBaseClass::dim; k++) {
          if (j == i) 
            basis_gradients[m].at(i) += basis_coefficients[m](
                                     2*BasisForSurrogateModelBaseClass::dim+
                                     1+counter) * x.at(k);
          if (k == i) 
            basis_gradients[m].at(i) += basis_coefficients[m](
                                     2*BasisForSurrogateModelBaseClass::dim+
                                     1+counter) * x.at(j);
          counter++;
        }
      }
    }
  }

  clear_cache_gradients = false;
  cache_point_basis_gradients = x;

  return;
}
//--------------------------------------------------------------------------------

//--------------------------------------------------------------------------------
void BasisForMinimumFrobeniusNormModel::compute_mat_vec_representation ( int basis_number )
// computes the representation m(x) = c + g.dot(x) + 0.5*x.dot(H*x) in scaled form
// x_best = 0 and delta = 1;
{
  counter = 0;
  for (int j = 0; j < BasisForSurrogateModelBaseClass::dim; j++) {
    scaled_gradients[basis_number].at(j) = basis_coefficients[ basis_number ]( j + 1 );
    scaled_Hessians[basis_number].at(j).at(j) = basis_coefficients[ basis_number ]
                                         (j+1+BasisForSurrogateModelBaseClass::dim);
    for (int k = j+1; k < BasisForSurrogateModelBaseClass::dim; k++) {
      scaled_Hessians[basis_number].at(j).at(k) = basis_coefficients[ basis_number ]
                                           (2*BasisForSurrogateModelBaseClass::dim+1+counter);
      scaled_Hessians[basis_number].at(k).at(j) = basis_coefficients[ basis_number ]
                                           (2*BasisForSurrogateModelBaseClass::dim+1+counter);
      counter++;
    }
  }
  
  return;
}
//--------------------------------------------------------------------------------

//--------------------------------------------------------------------------------
void BasisForMinimumFrobeniusNormModel::get_mat_vec_representation ( int basis_number,
  std::vector<double> &g, std::vector< std::vector<double> > &H )
// returns the representation m(x) = c + g.dot(x) + 0.5*x.dot(H*x) in scaled form
// x_best = 0 and delta = 1;
{
  g = scaled_gradients [ basis_number ];
  H = scaled_Hessians [ basis_number ];
  return;
}
//--------------------------------------------------------------------------------

//--------------------------------------------------------------------------------
std::vector<double> &BasisForMinimumFrobeniusNormModel::evaluate ( 
  std::vector<double> const &x ) 
{  
  if ( cache_point_basis_values.size() > 0 ) {
    if ( diff_norm ( cache_point_basis_values, x ) < 1e-20 &&
          !clear_cache_values ) 
      return basis_values;
  }

  for ( int i = 0; i < nb_nodes; i++ ) {
    basis_values.at( i ) = 0e0;
    for ( int j = 0; j < nb_basis_functions; j++ ) {
      basis_values.at( i ) += basis_coefficients[ i ]( j ) * evaluate_monomial ( j, x );
    }
  }

  clear_cache_values = false;
  cache_point_basis_values = x;

  return basis_values;
}
//--------------------------------------------------------------------------------

//--------------------------------------------------------------------------------
double BasisForMinimumFrobeniusNormModel::evaluate (
  std::vector<double> const &x, int basis_number)
{ 
  if ( cache_point_basis_values.size() > 0 ) {
    if ( diff_norm( cache_point_basis_values, x ) < 1e-20 &&
         !clear_cache_values ) 
      return basis_values.at ( basis_number );
  }
  double basis_value = 0e0;
  for ( int j = 0; j < nb_basis_functions; j++ ) {
    basis_value += basis_coefficients[ basis_number ]( j ) * 
                   evaluate_monomial ( j, x );
  }

  return basis_value;
}
//--------------------------------------------------------------------------------

//--------------------------------------------------------------------------------
std::vector<double> &BasisForMinimumFrobeniusNormModel::gradient (
  std::vector<double> const &x, int basis_number)
{ 
  compute_gradients ( x );

  return basis_gradients.at( basis_number );
}
//--------------------------------------------------------------------------------
