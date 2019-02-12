#include "QuadraticMinimization.hpp"
#include <iostream>

//--------------------------------------------------------------------------------
QuadraticMinimization::QuadraticMinimization ( int n ) : 
  TriangularMatrixOperations ( n )
{
  dim = n;
  //set parameters for stopping criteria
  sigma1 = 1e-4; // 1e-3
  sigma2 = 1e-2; // 1e-1
  //allocate memory for auxiliary variables for the More/Sorensen algorithm
  M.resize ( dim );
  for (int i = 0; i < dim; ++i)
    M.at(i).resize(dim);
  z_hat.resize ( dim );
  u.resize( dim );
}
//--------------------------------------------------------------------------------


//--------------------------------------------------------------------------------
void QuadraticMinimization::minimize ( std::vector<double> &y, 
  std::vector<double> const &g, std::vector< std::vector<double> > const &H ) { 
  //this algorithm computes a solution of the optimization problem
  // y = argmin (0.5*xHx + g'x) subject to ||x|| <= delta (here: delta is fixed to 1)
  //reference: More/Sorensen 1983

  double delta_loc = 1e0;

  // initialize variables
  tau = 0e0;
  tmp_save = 0e0;

  //initialize solution
  VectorOperations::set_zero( y );
  norm_y = 0e0;
  norm_g = VectorOperations::norm( g );

  //initialize safeguarding parameters
  norm_g_d_delta = norm_g / delta_loc;
  l1_norm_B      = 0e0;
  lam_S = H.at(0).at(0);
  for (int i = 0; i < dim; ++i) {
    tmp_dbl = 0e0;
    for (int j = 0; j < dim; ++j) 
      tmp_dbl += fabs(H.at(j).at(i));
    if (tmp_dbl > l1_norm_B) l1_norm_B = tmp_dbl;
    if ( H.at(i).at(i) < lam_S ) lam_S = H.at(i).at(i);
  }
  lam_S = -lam_S;
  lam_L = norm_g_d_delta - l1_norm_B;
  lam_U = norm_g_d_delta + l1_norm_B;
  lam   = lam_U;
  if (lam_L < lam_S) lam_L = lam_S;
  if (lam_L < 0e0)   lam_L = 0e0;
  counter = 0;

  while (counter < 10000 || true ) {
    counter++;

    //safeguard lambda
    if (lam < lam_L) lam = lam_L;
    if (lam > lam_U) lam = lam_U;
    if (lam <= lam_S) {
      lam = 1e-3*lam_U;
      if (lam < sqrt(lam_L*lam_U)) lam = sqrt(lam_L*lam_U);
    }
    M = H;
    for ( int i = 0; i < dim; ++i ) 
      M.at(i).at(i) += lam; 
    CholeskyFactorization::compute(M, p, offset, dim);
    if (p == 0) {  //i.e. if M = H+lam*I is positive definite
      //solve the linear system (H + lambda*I)y - -g
      VectorOperations::scale( -1e0, g, y);
      TriangularMatrixOperations::forward_substitution( M, y );
      TriangularMatrixOperations::backward_substitution( M, y );
      norm_y = VectorOperations::norm( y );
      if (norm_y < delta_loc) {
        //compute z_hat
        TriangularMatrixOperations::compute_large_norm_solution(M, z_hat);
        TriangularMatrixOperations::backward_substitution( M, z_hat );
        VectorOperations::scale( 1e0/VectorOperations::norm(z_hat), z_hat, z_hat);
        //compute tau
        tau   = pow( delta_loc, 2e0 ) - norm_y*norm_y;
        yzhat = VectorOperations::dot_product( y, z_hat );
        if (yzhat < 0) syzhat = -1e0; else syzhat = 1e0;
        tau = tau / (yzhat + syzhat*sqrt(yzhat*yzhat+tau));
      }
    }
    //update lam_L, lam_U, lam_S
    if (p == 0) {
      if (norm_y < delta_loc) {
        if (lam_U > lam) lam_U = lam;
        tmp_save = 0e0;
        for ( int i = 0; i < dim; ++i ) {
          tmp_dbl = 0e0;
          for ( int j = i; j < dim; ++j )
            tmp_dbl += M.at(j).at(i)*z_hat.at(j);
          tmp_save += tmp_dbl * tmp_dbl;
        } 
        tmp_dbl = lam - tmp_save;
        if (lam_S < tmp_dbl) lam_S = tmp_dbl;
      } else {
        if (lam_L < lam) lam_L = lam;
      }
    } else {
      if (lam_L < lam) lam_L = lam;
      A.clear();
      for ( int i = 0; i < p; ++i )
        A.push_back( M.at(i) );
      uu.resize(p);
      VectorOperations::set_zero( uu ); 
      uu.at(p-1) = 1e0;
      TriangularMatrixOperations::backward_substitution( A, uu );
      tmp_dbl = lam + offset/ VectorOperations::dot_product(uu, uu);
      if (lam_S < tmp_dbl) lam_S = tmp_dbl;
    }
    if (lam_L < lam_S) lam_L = lam_S;
    //check convergence criteria
    if (p == 0) {
      tmp_dbl = fabs( delta_loc - norm_y );
      if (tmp_dbl <= sigma1*(delta_loc) || (norm_y <= delta_loc && lam <= 1e-16)) return;

      if (norm_y < delta_loc) {
        tmp_dbl = lam*pow( delta_loc, 2e0 );
        for ( int i = 0; i < dim; ++i ) {
          tmp_norm = 0e0;
          for ( int j = i; j < dim; ++j ) 
            tmp_norm += M.at(j).at(i) * y.at(j);
          tmp_dbl += tmp_norm*tmp_norm;
        }
        if (tmp_dbl < sigma2) tmp_dbl = sigma2;
        tmp_dbl = sigma1*(2e0-sigma2)*tmp_dbl;				
        if (tau*tau*tmp_save <= tmp_dbl) {
          for ( int i = 0; i < dim; ++i )
            y.at(i) += tau*z_hat.at(i);
          return;
        }
      }
    }
    //update lambda
    if (p == 0 && norm_g > 0) {
      u = y;
      TriangularMatrixOperations::forward_substitution( M, u );
      lam += pow(norm_y/VectorOperations::norm(u), 2e0)*((norm_y-delta_loc)/delta_loc); 
    } else lam = lam_S;

  }

  return;

}
//--------------------------------------------------------------------------------
