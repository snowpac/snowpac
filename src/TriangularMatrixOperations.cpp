#include "TriangularMatrixOperations.hpp"
#include <iostream>

//--------------------------------------------------------------------------------
TriangularMatrixOperations::TriangularMatrixOperations ( int n ) 
{
  dim = n;
  w.resize( dim );
  p.resize( dim );
  d.resize( dim );
}
//--------------------------------------------------------------------------------


//--------------------------------------------------------------------------------
void TriangularMatrixOperations::forward_substitution (
  std::vector< std::vector<double> > const &L, std::vector<double> &x )
{
  // solves the system L * y = x for a lower triangular nonsingular matrix L
  // and stores the solution y in x 
  dim = x.size();
  for (int i = 0; i < dim; ++i) {
    tmp = x.at(i);
    for (int j = 0; j <= i-1; ++j)
      tmp -= L.at(i).at(j)*x.at(j);
    x.at(i) = tmp / L.at(i).at(i);
  }

  return;
}
//--------------------------------------------------------------------------------

//--------------------------------------------------------------------------------
void TriangularMatrixOperations::forward_substitution_not_inplace (
        std::vector< std::vector<double> > const &L, std::vector<double> const &b, std::vector<double> &x )
{
    // solves the system L * y = x for a lower triangular nonsingular matrix L
    // and stores the solution y in x
    dim = x.size();
    for (int i = 0; i < dim; ++i) {
        tmp = b.at(i);
        for (int j = 0; j <= i-1; ++j)
            tmp -= L.at(i).at(j)*x.at(j);
        x.at(i) = tmp / L.at(i).at(i);
    }

    return;
}
//--------------------------------------------------------------------------------

//--------------------------------------------------------------------------------
void TriangularMatrixOperations::backward_substitution (
  std::vector< std::vector<double> > const &L, std::vector<double> &x )
{
  // solves the system L' * y = x for a lower triangular nonsingular matrix L
  // and stores the solution y in x
  dim = x.size();
  for (int i = dim-1; i >= 0; --i) {
    tmp = x[i];
    for (int j = i+1; j < dim; ++j)
      tmp -= L[j][i]*x[j];
    x[i] = tmp / L[i][i];
  }
  return;
}
//--------------------------------------------------------------------------------

void TriangularMatrixOperations::backward_substitution_not_inplace(std::vector<std::vector<double> > const &L,
                                                                   std::vector<double> const &b, std::vector<double> &x)
{
    // solves the system L' * y = x for a lower triangular nonsingular matrix L
    // and stores the solution y in x
    dim = x.size();
    for (int i = dim-1; i >= 0; --i) {
        tmp = b.at(i);
        for (int j = i+1; j < dim; ++j)
            tmp -= L[j][i]*x[j];
        x[i] = tmp / L[i][i];
    }
    return;

}

//--------------------------------------------------------------------------------
void TriangularMatrixOperations::compute_large_norm_solution ( 
  std::vector< std::vector<double> > const &L, std::vector<double> &y ) {
  //this algorithm computes an approximation of the largest norm solution y of
  //Ly = p for any vector ||p||_2 = 1. Reference: Cline et. al (1982)

  //define the weights and initialize solution
  for (int i = 0; i < dim; ++i) {
    p.at(i) = 0e0;
    w.at(i) = 1e0/L.at(i).at(i);
  }
  yy = 0e0;

  for (int k = 0; k < dim; ++k) {
    //step 1: compute cosine/sine pairs
    if (k == 0) {
      c    = 1e0;
      s    = 0e0;
      y.at(0) = c/L.at(0).at(0);
      yy   = 0e0;
    } else {
      yy = yy + y.at(k-1)*y.at(k-1);
      tDDt = 0e0;
      pDDp = 0e0;
      pDDt = 0e0;
      for ( int i = k+1; i < dim; ++i ) {
        tDDt += pow( L.at(i).at(k)*w.at(i), 2e0 );
        pDDp += pow( p.at(i)*w.at(i), 2e0 );
        pDDt += L.at(i).at(k) * w.at(i) * p.at(i) * w.at(i);
      }
      beta  = (yy + pDDp)*pow(L.at(k).at(k),2e0) + (p.at(k)*p.at(k) - 1e0)*(1e0+tDDt) -
              2e0*p.at(k)*L.at(k).at(k)*pDDp;
      alpha = p.at(k)*(1e0+tDDt) - L.at(k).at(k)*pDDt;
      if (alpha != 0e0) {
        r  = beta / (2e0 * alpha);
        mu = r + sqrt(1e0 + r*r);
        s1 = 1/sqrt(1e0 + mu*mu);
        c1 = s1*mu; 
        mu = r - sqrt(1e0 + r*r);
        s2 = 1/sqrt(1e0 + mu*mu);
        c2 = s2*mu; 
      } else {
        s1 = 0e0;
        c1 = 1e0;
        s2 = 1e0;
        c2 = 0e0;
      }
      y1   = (c1 - s1*p.at(k))/L.at(k).at(k);
      y2   = (c2 - s2*p.at(k))/L.at(k).at(k);
      phi1 = s1*s1*yy + y1*y1;
      phi2 = s2*s2*yy + y2*y2;
      if (k < dim-1) {
        phi1 += s1*s1*pDDp + y1*y1*tDDt + 2e0*s1*y1*pDDt;
        phi2 += s2*s2*pDDp + y2*y2*tDDt + 2e0*s2*y2*pDDt;
      }
      if (phi1 > phi2) {
        s    = s1;
        c    = c1;
        y.at(k) = y1;
      } else {
        s    = s2;
        c    = c2;
        y.at(k) = y2;			
      }
    }
    //step 2:
    //note: updating y(k) has already been incorporated in step 1 
    d.at(k) = c;

    for (int i = 0; i < k; ++i) {
      d.at(i) *= s;
      y.at(i) *= s;
    }

    for (int i = k+1; i < dim; ++i)
      p.at(i) = s*p.at(i) + L.at(i).at(k)*y.at(k);
  
  }
  return;     
}
//--------------------------------------------------------------------------------

