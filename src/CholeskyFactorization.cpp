#include "CholeskyFactorization.hpp"
#include <iostream>

//--------------------------------------------------------------------------------
void CholeskyFactorization::compute(
  std::vector< std::vector<double> > &L, int &p, double &offset, int n) 
{
  //initially the matrix M to be factorized is stored in L. 
  //computes an lower tridiagonal matrix L such that M = L*L' if M is positive definite
  //otherwise it returns R such that M(0:(p-1), 0:(p-1)) = L(0:(p-1), 0:(p-1))L(0:(p-1), 0:(p-1))'
  //and the value p such that (M + offset*ep*ep') has a zero eigenvalue, where
  //ep is the p-th canonical unit vector. The variable p = 0, if M is positive definite.
 
  //initialize variables
  p      = 0;
  offset = 0e0;
	
  for (int i = 0; i < n; ++i) {
    //check for special case
    if (L.at(i).at(i) == 0e0) {
      p      = i+1;
      offset = 0e0;		
      return;
    }
    for (int j = 0; j <= i; ++j) {
      s = L.at(i).at(j);
      for (int k = 0; k <= j-1; ++k) 
        s = s - L.at(i).at(k)*L.at(j).at(k);
      if (i > j) {
        L.at(i).at(j) = s / L.at(j).at(j);
      } else {				
        if (s > 0e0) L.at(i).at(i) = sqrt(s);
        else {
          p      = i+1; 
          offset = -s;					
          L.at(i).at(i) = 1e0;
          return;
        }
      }
    }
// Pseudo code:  if (n-i-1 > 0) (L.row(i)).tail(n-i-1) = [0, ..., 0]; 
  }	
  return;
}
