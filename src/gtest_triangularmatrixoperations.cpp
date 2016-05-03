#include "TriangularMatrixOperations.hpp"
#include "gtest/gtest.h"

//--------------------------------------------------------------------------------
class Wrapper_TriangularMatrixOperations : public TriangularMatrixOperations {
  public:
    Wrapper_TriangularMatrixOperations ( int n ) :
      TriangularMatrixOperations ( n ) { };
    int forward_test () 
    {
      std::vector< std::vector<double> > L (3);
      L[0].resize(1);
      L[1].resize(2);
      L[2].resize(3);
      L[0][0] = 1;
      L[1][0] = 2;
      L[1][1] = 3;
      L[2][0] = 4;
      L[2][1] = 5;
      L[2][2] = 6;

      std::vector<double> x(3);
      x[0] = 1.0;
      x[1] = 2.0;
      x[2] = 3.0;

      TriangularMatrixOperations::forward_substitution ( L, x );

      if ( fabs( x[0] - 1.0 ) > 1e-6 ) return 0;
      if ( fabs( x[1] - 0.0 ) > 1e-6 ) return 0;
      if ( fabs( x[2] + 0.16666666666666666 ) > 1e-6 ) return 0;

      return 1;
    }

    int backward_test () 
    {
      std::vector< std::vector<double> > L (3);
      L[0].resize(1);
      L[1].resize(2);
      L[2].resize(3);
      L[0][0] = 1;
      L[1][0] = 2;
      L[1][1] = 3;
      L[2][0] = 4;
      L[2][1] = 5;
      L[2][2] = 6;

      std::vector<double> x(3);
      x[0] = 1.0;
      x[1] = 2.0;
      x[2] = 3.0;

      TriangularMatrixOperations::backward_substitution ( L, x );

      if ( fabs( x[0] + 0.6666666666666666 ) > 1e-6 ) return 0;
      if ( fabs( x[1] + 0.1666666666666666 ) > 1e-6 ) return 0;
      if ( fabs( x[2] - 0.5 ) > 1e-6 ) return 0;

      return 1;
    }

};
//--------------------------------------------------------------------------------


//--------------------------------------------------------------------------------
TEST ( TriangularMatrixOperationsTest, forward_substituion ) 
{
  Wrapper_TriangularMatrixOperations W ( 3 );
  EXPECT_EQ( 1, W.forward_test() );
}
//--------------------------------------------------------------------------------

//--------------------------------------------------------------------------------
TEST ( TriangularMatrixOperationsTest, backward_substitution ) 
{
  Wrapper_TriangularMatrixOperations W ( 3 );
  EXPECT_EQ( 1, W.backward_test() );
}
//--------------------------------------------------------------------------------
