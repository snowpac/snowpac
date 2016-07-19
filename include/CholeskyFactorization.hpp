#ifndef HCholeskyFactorization
#define HCholeskyFactorization

#include <vector>
#include "math.h"


//! Cholesky factorization
/*!
 Computes the Cholesky factorization of a matrix M if M is positive definite,
 i.e. M = LL' with a lower triangular matrix L.\nx
 If M is positive defineite the matrix M is over-written by L and p and offset are set to zero.\n
 If M is not positive definite with a zero eigenvalue, a partial Cholesky decomposition
 in the upper left (p-1)x(p-1) block of M is returned.\n 
 Additionally, p and offset are 
 set such that M + rho ep ep' (with the p-th canonical unit vector ep) has a zero
 eigenvalue.
*/
class CholeskyFactorization {
  private:
    //!auxiliary variable
    double s;
  protected:
    //! Computes Cholesky factorization
    /*!
     \param L matrix M to be factorizized on input and the factorized matrix L on output
     \param p  index of the diagonal element such that M+offset ep ep^T has a zero eigenvalue
     \param offset offset to shift non-positive eigenvalue of M to zero.
     \param n dimension of the matrix L
    */
    void compute ( std::vector< std::vector<double> >&, int&, double&, int );
};

#endif
