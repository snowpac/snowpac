#include "QuadraticMonomial.hpp"
#include "gtest/gtest.h"
#include <vector>
#include <iostream>

//--------------------------------------------------------------------------------
class Wrapper_QuadraticMonomial : public QuadraticMonomial
{
  public:
    Wrapper_QuadraticMonomial ( int dim_input) : QuadraticMonomial ( dim_input ) { }
    int quadraticmonomial_test1 ( )
    {

      std::vector<double> x(1);

      x.at(0) = 0e0;
      if ( fabs( evaluate_basis(0, x) - 1e0 ) > 1e-6 ) return 0;
      if ( fabs( evaluate_basis(1, x)       ) > 1e-6 ) return 0;
      if ( fabs( evaluate_basis(2, x)       ) > 1e-6 ) return 0;

      x.at(0) = 1e0;
      if ( fabs( evaluate_basis(0, x) - 1e0  ) > 1e-6 ) return 0;
      if ( fabs( evaluate_basis(1, x) - 1e0  ) > 1e-6 ) return 0;
      if ( fabs( evaluate_basis(2, x) - 5e-1 ) > 1e-6 ) return 0;

      x.at(0) = 2e0;
      if ( fabs( evaluate_basis(0, x) - 1e0 ) > 1e-6 ) return 0;
      if ( fabs( evaluate_basis(1, x) - 2e0 ) > 1e-6 ) return 0;
      if ( fabs( evaluate_basis(2, x) - 2e0 ) > 1e-6 ) return 0;

      return 1;
    }
    int quadraticmonomial_test2 ( )
    {

      std::vector<double> x(2);
      std::vector<double> solution ( 6 );

      x.at(0) = 0e0; x.at(1) = 0e0;
      solution = { 1e0, 0e0, 0e0, 0e0, 0e0, 0e0 };
      for ( int i = 0; i < 6; ++i )
        if ( fabs( evaluate_basis(i, x) - solution.at(i) ) > 1e-6 ) return 0;

      x.at(0) = 1e0;  x.at(1) = 0e0;
      solution = { 1e0, 1e0, 0e0, 5e-1, 0e0, 0e0 };
      for ( int i = 0; i < 6; ++i )
        if ( fabs( evaluate_basis(i, x) - solution.at(i) ) > 1e-6 ) return -1;

      x.at(0) = 0e0;  x.at(1) = 1e0;
      solution = { 1e0, 0e0, 1e0, 0e0, 5e-1, 0e0 };
      for ( int i = 0; i < 6; ++i )
        if ( fabs( evaluate_basis(i, x) - solution.at(i) ) > 1e-6 ) return -2;

      x.at(0) = 1e0;  x.at(1) = 1e0;
      solution = { 1e0, 1e0, 1e0, 5e-1, 5e-1, 1e0 };
      for ( int i = 0; i < 6; ++i )
        if ( fabs( evaluate_basis(i, x) - solution.at(i) ) > 1e-6 ) return -3;

      x.at(0) = 2e0;  x.at(1) = 0e0;
      solution = { 1e0, 2e0, 0e0, 2e0, 0e0, 0e0 };
      for ( int i = 0; i < 6; ++i )
        if ( fabs( evaluate_basis(i, x) - solution.at(i) ) > 1e-6 ) return -4;

      x.at(0) = 0e0;  x.at(1) = 2e0;
      solution = { 1e0, 0e0, 2e0, 0e0, 2e0, 0e0 };
      for ( int i = 0; i < 6; ++i )
        if ( fabs( evaluate_basis(i, x) - solution.at(i) ) > 1e-6 ) return -5;

      x.at(0) = 2e0;  x.at(1) = 2e0;
      solution = { 1e0, 2e0, 2e0, 2e0, 2e0, 4e0 };
      for ( int i = 0; i < 6; ++i )
        if ( fabs( evaluate_basis(i, x) - solution.at(i) ) > 1e-6 ) return -6;

      return 1;
    }
    int quadraticmonomial_test3 ( )
    {

      std::vector<double> x(3);
      std::vector<double> solution ( 10 );

      x.at(0) = 0e0; x.at(1) = 0e0; x.at(2) = 0e0;
      solution = { 1e0, 0e0, 0e0, 0e0, 0e0, 0e0, 0e0, 0e0, 0e0, 0e0 };
      for ( int i = 0; i < 10; ++i )
        if ( fabs( evaluate_basis(i, x) - solution.at(i) ) > 1e-6 ) return 0;

      x.at(0) = 1e0; x.at(1) = 0e0; x.at(2) = 0e0;
      solution = { 1e0, 1e0, 0e0, 0e0, 5e-1, 0e0, 0e0, 0e0, 0e0, 0e0 };
      for ( int i = 0; i < 10; ++i )
        if ( fabs( evaluate_basis(i, x) - solution.at(i) ) > 1e-6 ) return 0;

      x.at(0) = 0e0; x.at(1) = 1e0; x.at(2) = 0e0;
      solution = { 1e0, 0e0, 1e0, 0e0, 0e0, 5e-1, 0e0, 0e0, 0e0, 0e0 };
      for ( int i = 0; i < 10; ++i )
        if ( fabs( evaluate_basis(i, x) - solution.at(i) ) > 1e-6 ) return 0;

      x.at(0) = 0e0; x.at(1) = 0e0; x.at(2) = 1e0;
      solution = { 1e0, 0e0, 0e0, 1e0, 0e0, 0e0, 5e-1, 0e0, 0e0, 0e0 };
      for ( int i = 0; i < 10; ++i )
        if ( fabs( evaluate_basis(i, x) - solution.at(i) ) > 1e-6 ) return 0;

      x.at(0) = 1e0; x.at(1) = 1e0; x.at(2) = 0e0;
      solution = { 1e0, 1e0, 1e0, 0e0, 5e-1, 5e-1, 0e0, 1e0, 0e0, 0e0 };
      for ( int i = 0; i < 10; ++i )
        if ( fabs( evaluate_basis(i, x) - solution.at(i) ) > 1e-6 ) return 0;

      x.at(0) = 1e0; x.at(1) = 0e0; x.at(2) = 1e0;
      solution = { 1e0, 1e0, 0e0, 1e0, 5e-1, 0e0, 5e-1, 0e0, 1e0, 0e0 };
      for ( int i = 0; i < 10; ++i )
        if ( fabs( evaluate_basis(i, x) - solution.at(i) ) > 1e-6 ) return 0;

      x.at(0) = 0e0; x.at(1) = 1e0; x.at(2) = 1e0;
      solution = { 1e0, 0e0, 1e0, 1e0, 0e0, 5e-1, 5e-1, 0e0, 0e0, 1e0 };
      for ( int i = 0; i < 10; ++i )
        if ( fabs( evaluate_basis(i, x) - solution.at(i) ) > 1e-6 ) return 0;

      x.at(0) = 2e0; x.at(1) = 2e0; x.at(2) = 0e0;
      solution = { 1e0, 2e0, 2e0, 0e0, 2e0, 2e0, 0e0, 4e0, 0e0, 0e0 };
      for ( int i = 0; i < 10; ++i )
        if ( fabs( evaluate_basis(i, x) - solution.at(i) ) > 1e-6 ) return 0;

      x.at(0) = 2e0; x.at(1) = 0e0; x.at(2) = 2e0;
      solution = { 1e0, 2e0, 0e0, 2e0, 2e0, 0e0, 2e0, 0e0, 4e0, 0e0 };
      for ( int i = 0; i < 10; ++i )
        if ( fabs( evaluate_basis(i, x) - solution.at(i) ) > 1e-6 ) return 0;

      x.at(0) = 0e0; x.at(1) = 2e0; x.at(2) = 2e0;
      solution = { 1e0, 0e0, 2e0, 2e0, 0e0, 2e0, 2e0, 0e0, 0e0, 4e0 };
      for ( int i = 0; i < 10; ++i )
        if ( fabs( evaluate_basis(i, x) - solution.at(i) ) > 1e-6 ) return 0;

      x.at(0) = 2e0; x.at(1) = 2e0; x.at(2) = 2e0;
      solution = { 1e0, 2e0, 2e0, 2e0, 2e0, 2e0, 2e0, 4e0, 4e0, 4e0 };
      for ( int i = 0; i < 10; ++i )
        if ( fabs( evaluate_basis(i, x) - solution.at(i) ) > 1e-6 ) return 0;

      return 1;
    }

};
//--------------------------------------------------------------------------------




//--------------------------------------------------------------------------------
TEST ( QuadraticMonomialTest, evaluation_test_1d ) 
{
  Wrapper_QuadraticMonomial W( 1 );
  EXPECT_EQ( 1, W.quadraticmonomial_test1() );
}
//--------------------------------------------------------------------------------

//--------------------------------------------------------------------------------
TEST ( QuadraticMonomialTest, evaluation_test_2d ) 
{
  Wrapper_QuadraticMonomial W( 2 );
  EXPECT_EQ( 1, W.quadraticmonomial_test2() );
}
//--------------------------------------------------------------------------------

//--------------------------------------------------------------------------------
TEST ( QuadraticMonomialTest, evaluation_test_3d ) 
{
  Wrapper_QuadraticMonomial W( 3 );
  EXPECT_EQ( 1, W.quadraticmonomial_test3() );
}
//--------------------------------------------------------------------------------
