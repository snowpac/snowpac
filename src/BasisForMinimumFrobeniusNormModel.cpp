#include "BasisForMinimumFrobeniusNormModel.hpp"
#include <iostream>
#include <fstream>
#include <iomanip>


#define mybool false 

//--------------------------------------------------------------------------------
BasisForMinimumFrobeniusNormModel::BasisForMinimumFrobeniusNormModel ( int dim_input ) :
                                   BasisForSurrogateModelBaseClass ( dim_input ),
                                   QuadraticMonomial ( dim_input ) 
{
  cache_values_exist = false;
  cache_gradients_exist = false;
  cache_point_basis_values.resize( 0 );
  cache_point_basis_gradients.resize( 0 );
  nb_basis_functions = ( dim_input * ( dim_input + 3 ) + 2 ) / 2;
  nb_nodes = 0;
}
//--------------------------------------------------------------------------------

//--------------------------------------------------------------------------------
void BasisForMinimumFrobeniusNormModel::set_nb_nodes( int nb_nodes_input ) {
  if ( nb_nodes == nb_nodes_input ) return;
  nb_nodes = nb_nodes_input;

//  if ( nb_basis_functions <= nb_nodes ) nb_basis_functions = 0;

  basis_values.clear();
  basis_values.resize ( nb_nodes );
  A_sysmat = Eigen::MatrixXd::Zero(nb_basis_functions + nb_nodes, 
                                   nb_basis_functions + nb_nodes);

  for ( int i = 0; i < nb_basis_functions; ++i)
    if (i > BasisForSurrogateModelBaseClass::dim) A_sysmat(i, i) = 1e0;
  F_rhsmat = Eigen::MatrixXd::Zero(nb_basis_functions + nb_nodes, nb_nodes);

  for ( int i = 0; i < nb_nodes; ++i ) 
    F_rhsmat(nb_basis_functions + i, i) = 1e0;
  gradients.clear();
  gradients.resize( nb_nodes );
  basis_gradients.clear();
  basis_gradients.resize( nb_nodes );
  basis_Hessians.clear();
  basis_Hessians.resize( nb_nodes );
  basis_coefficients.clear();
  basis_coefficients.resize( nb_nodes );

//  if ( nb_basis_functions == 0 ) nb_basis_functions = nb_nodes;

  for ( int i = 0; i < nb_nodes; ++i ) {
    basis_coefficients[i] = Eigen::VectorXd::Zero( nb_basis_functions ) ;
    gradients[i].resize( BasisForSurrogateModelBaseClass::dim );
    basis_gradients[i].resize( BasisForSurrogateModelBaseClass::dim );
    basis_Hessians[i].resize( BasisForSurrogateModelBaseClass::dim );
    for ( int j = 0; j < BasisForSurrogateModelBaseClass::dim; ++j) 
      basis_Hessians[i][j].resize( BasisForSurrogateModelBaseClass::dim );
  }


  return;
}
//--------------------------------------------------------------------------------


//--------------------------------------------------------------------------------
void BasisForMinimumFrobeniusNormModel::compute_basis_coefficients ( 
  std::vector< std::vector<double> > const &nodes )
{

  cache_point_basis_values.resize( 0 );
  cache_point_basis_gradients.resize( 0 );
  cache_values_exist = false;
  cache_gradients_exist = false;


  set_nb_nodes ( nodes.size( ) );

//  assert ( nb_nodes <= 6 );


  // system matrix for computing coeffs of Lagrange interpolation models
  for ( int i = 0; i < nb_basis_functions; ++i ) {    
    for ( int j = 0; j < nb_nodes; ++j ) {
      A_sysmat(i, j+nb_basis_functions) = evaluate_monomial( i, nodes[j] );
      A_sysmat(j+nb_basis_functions, i) = A_sysmat(i, j+nb_basis_functions);
    }
  }    

  // solve for coefficients
//  S_coeffsolve = A_sysmat.householderQr().solve(F_rhsmat);

/*
  Eigen::FullPivHouseholderQR<Eigen::MatrixXd> CPHQR;
  CPHQR.setThreshold(1e-36);
  CPHQR = A_sysmat.fullPivHouseholderQr();
  S_coeffsolve = CPHQR.solve(F_rhsmat);
*/

  S_coeffsolve = A_sysmat.colPivHouseholderQr().solve(F_rhsmat);

/*
    std::cout << "----------------" << std::endl; 
    for ( int i = 0; i < nb_nodes; ++i ) {
      for ( int j = i+1; j < nb_nodes; ++j) {
//        if ( diff_norm(nodes[i], nodes[j]) < 1e-3 ) {
          std::cout << i << " and " << j << " diff = " << diff_norm(nodes[i], nodes[j]) <<std::endl;
//          for ( int k = 0; k < BasisForSurrogateModelBaseClass::dim; ++k )
//            std::cout << nodes[i][k] << ", ";
          std::cout << std::endl;
          if ( i == 0 && j == 2 ) {
          for ( int k = 0; k < BasisForSurrogateModelBaseClass::dim; ++k )
            std::cout << nodes[i][k] << ", ";
          std::cout << std::endl;
          for ( int k = 0; k < BasisForSurrogateModelBaseClass::dim; ++k )
            std::cout << nodes[j][k] << ", ";
          std::cout << std::endl;
          }
      //  }
      }
    }
*/
//  if ( (A_sysmat.block(0,0,nb_basis_functions,nb_basis_functions+nb_nodes) * 
//        S_coeffsolve - F_rhsmat.block(0,0,nb_basis_functions, nb_nodes)).norm() > 1e-6 ) {
//    std::cout << "----------------" << std::endl; 
//    std::cout << (A_sysmat * S_coeffsolve - F_rhsmat).norm() << std::endl;
  //  std::cout << (A_sysmat.block(0,0,nb_basis_functions,nb_basis_functions+nb_nodes) * 
  //      S_coeffsolve - F_rhsmat.block(0,0,nb_basis_functions, nb_nodes)).norm() << std::endl;
//    std::cout << A_sysmat * S_coeffsolve - F_rhsmat << std::endl;
//    std::cout << "----------------" << std::endl; 
//    assert( false );
//  }


//  S_coeffsolve = A_sysmat.colPivHouseholderQr().solve(F_rhsmat);
  for ( int i = 0; i < nb_nodes; ++i ) {
    basis_coefficients[i] = S_coeffsolve.block(0,0,nb_basis_functions, nb_nodes).col(i);  
    compute_mat_vec_representation ( i );
  } 


   if ( (A_sysmat * S_coeffsolve - F_rhsmat).norm() > 1e-5 ) {
/*
     std::vector<double> fvals;
     for ( int i = 0; i < nb_nodes; ++i) { 
       fvals = evaluate( nodes[i] );
       for ( int j = 0; j < nb_nodes; ++j )
         std::cout << std::setprecision(2) << fabs(fvals[j]) << ", " ;
       std::cout << std::endl;
     }
*/

    std::ofstream outputfile ( "points.dat" );
    if ( outputfile.is_open() ) {
      for ( int i = 0; i < nb_nodes; ++i ) {
        for ( int j = 0; j < BasisForSurrogateModelBaseClass::dim; ++j )
          outputfile << nodes[i][j] << "; ";
        outputfile << std::endl;
      }
    } 
    outputfile.close();

  
    std::cout << " nb nodes = " << nb_nodes << std::endl;
    std::cout << "----------------" << std::endl; 
    std::cout << (A_sysmat * S_coeffsolve - F_rhsmat).norm() << std::endl;
    std::cout << "----------------" << std::endl; 

/*
  std::vector<double> vals;
  for ( int j = 0; j < nb_nodes; ++j ) {
    vals = evaluate( nodes[ j ]);
    for ( int i = 0; i < nb_nodes; ++i ) {
      std::cout << vals.at(i) << ", ";
    }    
    std::cout << std::endl;
  }
*/

     assert(false);
  }



  return;
}
//--------------------------------------------------------------------------------


//--------------------------------------------------------------------------------
void BasisForMinimumFrobeniusNormModel::compute_gradients ( 
 std::vector<double> const &x )
{

  if ( cache_point_basis_gradients.size() > 0 && cache_gradients_exist && mybool ) {
    if ( diff_norm( cache_point_basis_gradients, x ) < 1e-20 ) 
      return;
  }
  for (int m = 0; m < nb_nodes; ++m) {
    for (int i = 0; i < BasisForSurrogateModelBaseClass::dim; ++i) {
      gradients[m].at(i) = basis_coefficients[m](i+1);
      gradients[m].at(i) += basis_coefficients[m](
                            BasisForSurrogateModelBaseClass::dim+1+i) * x.at(i);
      counter = 0;
      for (int j = 0; j < BasisForSurrogateModelBaseClass::dim; ++j) {
        for (int k = j+1; k < BasisForSurrogateModelBaseClass::dim; ++k) {
          if (j == i) 
            gradients[m].at(i) += basis_coefficients[m](
                                  2*BasisForSurrogateModelBaseClass::dim+
                                  1+counter) * x.at(k);
          if (k == i) 
            gradients[m].at(i) += basis_coefficients[m](
                                  2*BasisForSurrogateModelBaseClass::dim+
                                  1+counter) * x.at(j);
          counter++;
        }
      }
    }
  }

  cache_gradients_exist = true;
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
  for (int j = 0; j < BasisForSurrogateModelBaseClass::dim; ++j) {
    basis_gradients[basis_number].at(j) = basis_coefficients[ basis_number ]( j + 1 );
    basis_Hessians[basis_number].at(j).at(j) = basis_coefficients[ basis_number ]
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
void BasisForMinimumFrobeniusNormModel::get_mat_vec_representation ( int basis_number,
  std::vector<double> &g, std::vector< std::vector<double> > &H )
// returns the representation m(x) = c + g.dot(x) + 0.5*x.dot(H*x) in scaled form
// x_best = 0 and delta = 1;
{
  g = basis_gradients [ basis_number ];
  H = basis_Hessians [ basis_number ];
  return;
}
//--------------------------------------------------------------------------------

//--------------------------------------------------------------------------------
std::vector<double> &BasisForMinimumFrobeniusNormModel::evaluate ( 
  std::vector<double> const &x ) 
{  
  if ( cache_point_basis_values.size() > 0 && cache_values_exist && mybool ) {
    if ( diff_norm ( cache_point_basis_values, x ) < 1e-20 ) 
      return basis_values;
  }
 
  assert( nb_nodes == basis_coefficients.size());

  for ( int i = 0; i < nb_nodes; ++i ) {
    basis_values.at( i ) = 0e0;
    for ( int j = 0; j < nb_basis_functions; ++j ) {
      basis_values.at( i ) += basis_coefficients[ i ]( j ) * evaluate_monomial ( j, x );
    }
  }

  cache_values_exist = true;
  cache_point_basis_values = x;

  return basis_values;
}
//--------------------------------------------------------------------------------

//--------------------------------------------------------------------------------
double BasisForMinimumFrobeniusNormModel::evaluate (
  std::vector<double> const &x, int basis_number)
{ 
  if ( cache_point_basis_values.size() > 0 && cache_values_exist && mybool ) {
    if ( diff_norm( cache_point_basis_values, x ) < 1e-20 ) 
      return basis_values.at ( basis_number );
  }
  double basis_value = 0e0;
  for ( int j = 0; j < nb_basis_functions; ++j ) {
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
  return gradients.at( basis_number );
}
//--------------------------------------------------------------------------------

//--------------------------------------------------------------------------------
std::vector<double> &BasisForMinimumFrobeniusNormModel::gradient (int basis_number)
{ 
  return basis_gradients.at( basis_number );
}
//--------------------------------------------------------------------------------
