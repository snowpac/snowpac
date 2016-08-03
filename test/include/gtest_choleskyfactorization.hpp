#include "CholeskyFactorization.hpp"
#include "gtest/gtest.h"
#include <Eigen/Core>

//--------------------------------------------------------------------------------
class Wrapper_CholeskyFactorization : public CholeskyFactorization 
{
  public:
    int cholesky_test1 () 
    { 
      int      n = 3;
      double   rho = 0e0;
      int      pos = 0;
      std::vector< std::vector<double> > M(n);
      for ( int i = 0; i < n; ++i )
        M[i].resize( i+1 );
      M[0][0] = 14;
      M[1][0] = 19;
      M[1][1] = 45;
      M[2][0] = 34;
      M[2][1] = 41;
      M[2][2] = 94;

      CholeskyFactorization::compute(M, pos, rho, n);

      if ( fabs( M[0][0] - 3.741657386773941 ) > 1e-6 ) return 0;
      if ( fabs( M[1][0] - 5.077963596336064 ) > 1e-6 ) return 0;
      if ( fabs( M[1][1] - 4.383410283590359 ) > 1e-6 ) return 0;
      if ( fabs( M[2][0] - 9.086882225022430 ) > 1e-6 ) return 0;
      if ( fabs( M[2][1] + 1.173254797094819 ) > 1e-6 ) return 0;
      if ( fabs( M[2][2] - 3.170495956418396 ) > 1e-6 ) return 0;

      return 1; 
    }

    int cholesky_test2 () 
    { 
      int      n = 3;
      double   rho = 0e0;
      int      pos = 0;
      std::vector< std::vector<double> > M(n);
      for ( int i = 0; i < n; ++i )
        M[i].resize( i+1 );
      M[0][0] = 10;
      M[1][0] = 19;
      M[1][1] = 41;
      M[2][0] = 34;
      M[2][1] = 41;
      M[2][2] = 90;

      CholeskyFactorization::compute(M, pos, rho, n);

      if ( fabs( M[0][0] - 3.162277660168 ) > 1e-6 ) return 0;
      if ( fabs( M[1][0] - 6.008327554320 ) > 1e-6 ) return 0;
      if ( fabs( M[1][1] - 2.213594362118 ) > 1e-6 ) return 0;
      if ( fabs( M[2][0] - 10.75174404457 ) > 1e-6 ) return 0;
      if ( fabs( M[2][1] + 10.66139325428 ) > 1e-6 ) return 0;
      if ( fabs( M[2][2] - 1.0 ) > 1e-6 ) return 0;
      if ( fabs( rho - 139.2653061224) > 1e-6 ) return 0;
      if ( pos != 3 ) return 0;

      return 1; 
    }

};
//--------------------------------------------------------------------------------


//--------------------------------------------------------------------------------
TEST ( CholeskyFactorizationTest, positive_definite_matrix ) 
{
  Wrapper_CholeskyFactorization W;
  EXPECT_EQ( 1, W.cholesky_test1() );
}
//--------------------------------------------------------------------------------

//--------------------------------------------------------------------------------
TEST ( CholeskyFactorizationTest, indefinite_matrix ) 
{
  Wrapper_CholeskyFactorization W;
  EXPECT_EQ( 1, W.cholesky_test2() );
}
//--------------------------------------------------------------------------------

