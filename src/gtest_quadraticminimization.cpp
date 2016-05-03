#include "QuadraticMinimization.hpp"
#include "gtest/gtest.h"
#include <vector>
#include <Eigen/Core>


//--------------------------------------------------------------------------------
class Wrapper_QuadraticMinimization : public QuadraticMinimization {
  public:
    Wrapper_QuadraticMinimization ( int dim_input ) :
       QuadraticMinimization ( dim_input ) { };

    int quadratic_minimization_test1 ( ) {
      std::vector<double> y(2);
      std::vector<double> g(2);
      std::vector< std::vector<double> > H(2);
      H[0].resize(2);
      H[1].resize(2);
      g[0] = 1.0; g[1] = 2.0;
      H[0][0] = -4.0; H[0][1] = 1.0;   //XX check second
      H[1][0] =  1.0; H[1][1] = 5.0;   //XX check second
      QuadraticMinimization::minimize( y, g, H );
      if ( fabs(y[0] + 0.994835624149) > 1e-6 ) return 0;
      if ( fabs(y[1] + 0.101499306351) > 1e-6 ) return 0;
      return 1;
    }
    int quadratic_minimization_test2 ( ) {
      std::vector<double> y(2);
      std::vector<double> g(2);
      std::vector< std::vector<double> > H(2);
      H[0].resize(2);
      H[1].resize(2);
      g[0] = 1.0; g[1] = 0.0;
      H[0][0] = -1.0; H[0][1] = 0.0;
      H[1][0] =  0.0; H[1][1] = 1.0; 
      QuadraticMinimization::minimize( y, g, H );
      if ( fabs(y[0] + 1.0 ) > 1e-6 ) return 0;
      if ( fabs(y[1] + 0.0 ) > 1e-6 ) return 0;
      return 1;
    }
    int quadratic_minimization_test3 ( ) {
      std::vector<double> y(2);
      std::vector<double> g(2);
      std::vector< std::vector<double> > H(2);
      H[0].resize(2);
      H[1].resize(2);
      g[0] = 0.0; g[1] = 2.0;
      H[0][0] = -4.0; H[0][1] = 0.0;
      H[1][0] =  0.0; H[1][1] = 4.0; 
      QuadraticMinimization::minimize( y, g, H );
      if ( fabs(y[0] - 0.968252226024 ) > 1e-6 && 
           fabs(y[0] - 0.968252226024 ) > 1e-6 ) return 0;
      if ( fabs(y[1] + 0.249975252374 ) > 1e-6 ) return 0;
      return 1;
    }

};
//--------------------------------------------------------------------------------


//--------------------------------------------------------------------------------
TEST ( QuadraticMinimizationTest, minimize_test1 ) 
{
  Wrapper_QuadraticMinimization W ( 2 );
  EXPECT_EQ( 1, W.quadratic_minimization_test1() );
}
//--------------------------------------------------------------------------------

//--------------------------------------------------------------------------------
TEST ( QuadraticMinimizationTest, minimize_test2 ) 
{
  Wrapper_QuadraticMinimization W ( 2 );
  EXPECT_EQ( 1, W.quadratic_minimization_test2() );
}
//--------------------------------------------------------------------------------

//--------------------------------------------------------------------------------
TEST ( QuadraticMinimizationTest, minimize_test3 ) 
{
  Wrapper_QuadraticMinimization W ( 2 );
  EXPECT_EQ( 1, W.quadratic_minimization_test3() );
}
//--------------------------------------------------------------------------------

