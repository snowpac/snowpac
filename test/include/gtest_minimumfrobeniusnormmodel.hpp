#include "gtest/gtest.h"
#include "MonomialBasisForMinimumFrobeniusNormModel.hpp"
#include "MinimumFrobeniusNormModel.hpp"
#include <math.h>

//--------------------------------------------------------------------------------
double function1 ( std::vector<double> x ) {
//  return pow(x[0], 2e0) + pow(x[1], 2e0);
  return pow(x[0], 2e0) + pow(x[1], 2e0) + 3e0*x[0]*x[1] + 0.5*x[0] + 9e0;
}
double gradient1 ( std::vector<double> x, int element ) {
//  return 2e0*x[ element ];
  double returnvalue = 2e0*x[element];
  if ( element == 0 ) returnvalue += 0.5 + 3e0*x[1];
  if ( element == 1 ) returnvalue += 3e0*x[0];
  return returnvalue;
}
double hessian1 ( std::vector<double> x, int element1, int element2 ) {
  double returnvalue = 3e0;
  if ( element1 == element2 ) returnvalue = 2.0;
  return returnvalue;
}
//--------------------------------------------------------------------------------

//--------------------------------------------------------------------------------
class Wrapper_MinimumFrobeniusNormModel {
  public:
    int mfnmodel_test1 ( ) {

      int test_passed = 1; 

      int dim = 2;
      MonomialBasisForMinimumFrobeniusNormModel basis( dim );
      std::vector<double> node( dim );
      std::vector< std::vector<double> > nodes;
      for ( int j = 0; j < dim; ++j )
        node.at(j) = 0e0;
      nodes.push_back( node );
      for ( int i = 0; i < dim; ++i ) {
        for ( int j = 0; j < dim; ++j )
          node.at(j) = 0e0;
        node.at(i) = 1e0;
        nodes.push_back( node );
        for ( int j = 0; j < dim; ++j )
          node.at(j) = 0e0;
        node.at(i) = -1e0;
        nodes.push_back( node );
      }

      for ( int j = 0; j < dim; ++j )
        node.at(j) = -1e-1;
      node.at(0) = 1e-2;
      nodes.push_back( node );

      basis.compute_basis_coefficients ( nodes );

      MinimumFrobeniusNormModel model ( basis );
      std::vector<double> function_values ( nodes.size() );

      for ( unsigned i = 0; i < nodes.size(); ++i ) 
        function_values[i] = function1( nodes[i] );
      
      model.set_function_values( function_values );

      std::vector<double> gradient;
      std::vector< std::vector<double> > hessian;

      gradient = model.gradient();
      for ( int i = 0; i < dim; ++i )
        node[i] = 0e0;       
      for ( int i = 0; i < dim; ++i ) {
        if ( fabs( gradient[i] - gradient1( node, i) ) > 1e-6 ) {
          std::cout << "Something is wrong with the gradient at the origin" << std::endl;
          for ( int j = 0; j < dim; ++j )
            std::cout << "Error in gradient[" << j << "] = " << fabs(gradient[j]) << std::endl;
          test_passed = 0; 
          break;
        }
      }

    
      for ( int i = 0; i < dim; ++i )
        node[i] = 0.0;    
      hessian = model.hessian( );
      for ( int i = 0; i < dim; ++i ) {
        for ( int j = 0; j < dim; ++j ) {
          if ( fabs( hessian[i][j] - hessian1( node, i, j ) ) > 1e-6 ) {
            std::cout << "Something is wrong with the hessian" << std::endl;
            for ( int k = 0; k < dim; ++k ) {
              for ( int n = 0; n < dim; ++n ) {
                std::cout << "Error in hessian[" << k << "," << n << "] = " << fabs(hessian[k][n]) << std::endl;
              }
            }
            test_passed = 0; 
            break;
          }
        }
      }
     
      return test_passed;
    }
//--------------------------------------------------------------------------------
    int mfnmodel_test2 ( ) {

      int test_passed = 1; 

      int dim = 2;
      MonomialBasisForMinimumFrobeniusNormModel basis( dim );
      std::vector<double> node( dim );
      std::vector< std::vector<double> > nodes;
      for ( int j = 0; j < dim; ++j )
        node.at(j) = 0e0;
      nodes.push_back( node );
      for ( int i = 0; i < dim; ++i ) {
        for ( int j = 0; j < dim; ++j )
          node.at(j) = 0e0;
        node.at(i) = 1e0;
        nodes.push_back( node );
        for ( int j = 0; j < dim; ++j )
          node.at(j) = 0e0;
        node.at(i) = -1e0;
        nodes.push_back( node );
      }

      for ( int j = 0; j < dim; ++j )
        node.at(j) = -1e-1;
      node.at(0) = 1e-2;
      nodes.push_back( node );

      basis.compute_basis_coefficients ( nodes );

      MinimumFrobeniusNormModel model ( basis );
      std::vector<double> function_values ( nodes.size() );

      for ( unsigned i = 0; i < nodes.size(); ++i ) 
        function_values[i] = function1( nodes[i] );
      
      model.set_function_values( function_values );

      double evaluation;

      for ( int i = 0; i < dim; ++i )
        node[i] = 0.5;    
      evaluation = model.evaluate( node );
      evaluation -= function1( node );
      if ( fabs( evaluation ) > 1e-6 ) {
        std::cout << "Something is wrong with the model evaluation" << std::endl;
        std::cout << "Error in evaluation = " << evaluation << std::endl;
        test_passed = 0; 
      }


      for ( int i = 0; i < dim; ++i )
        node[0] = 0.0 + ((double)i)/((double)dim) ;    
      evaluation = model.evaluate( node );
      evaluation -= function1( node );
      if ( fabs( evaluation ) > 1e-6 ) {
        std::cout << "Something is wrong with the model evaluation" << std::endl;
        std::cout << "Error in evaluation = " << evaluation << std::endl;
        test_passed = 0; 
      }

     
      return test_passed;
    }
//--------------------------------------------------------------------------------
    int mfnmodel_test3 ( ) {

      int test_passed = 1; 

      int dim = 2;
      MonomialBasisForMinimumFrobeniusNormModel basis( dim );
      std::vector<double> node( dim );
      std::vector< std::vector<double> > nodes;
      for ( int j = 0; j < dim; ++j )
        node.at(j) = 0e0;
      nodes.push_back( node );
      for ( int i = 0; i < dim; ++i ) {
        for ( int j = 0; j < dim; ++j )
          node.at(j) = 0e0;
        node.at(i) = 1e0;
        nodes.push_back( node );
      }

      basis.compute_basis_coefficients ( nodes );

      MinimumFrobeniusNormModel model ( basis );
      std::vector<double> function_values ( nodes.size() );

      for ( unsigned i = 0; i < nodes.size(); ++i ) 
        function_values[i] = function1( nodes[i] );
      
      model.set_function_values( function_values );

      for ( int i = 0; i < dim; ++i ) {
        for ( int j = 0; j < dim; ++j )
          node.at(j) = 0e0;
        node.at(i) = -1e0;
        nodes.push_back( node );
      }
      for ( int j = 0; j < dim; ++j )
        node.at(j) = -1e-1;
      node.at(0) = 1e-2;
      nodes.push_back( node );

      basis.compute_basis_coefficients ( nodes );
      function_values.resize( nodes.size() );
      for ( unsigned i = 0; i < nodes.size(); ++i ) 
        function_values[i] = function1( nodes[i] );

      model.set_function_values( function_values );

      double evaluation;

      for ( int i = 0; i < dim; ++i )
        node[i] = 0.5;    
      evaluation = model.evaluate( node );
      evaluation -= function1( node );
      if ( fabs( evaluation ) > 1e-6 ) {
        std::cout << "Something is wrong with the model evaluation" << std::endl;
        std::cout << "Error in evaluation = " << evaluation << std::endl;
        test_passed = 0; 
      }


      for ( int i = 0; i < dim; ++i )
        node[0] = 0.0 + ((double)i)/((double)dim) ;    
      evaluation = model.evaluate( node );
      evaluation -= function1( node );
      if ( fabs( evaluation ) > 1e-6 ) {
        std::cout << "Something is wrong with the model evaluation" << std::endl;
        std::cout << "Error in evaluation = " << evaluation << std::endl;
        test_passed = 0; 
      }

     
      return test_passed;
    }

};
//--------------------------------------------------------------------------------



//--------------------------------------------------------------------------------
TEST ( MinimumFrobeniusNormModelTest, mfnmodel_test1 ) 
{
  Wrapper_MinimumFrobeniusNormModel W;
  EXPECT_EQ( 1, W.mfnmodel_test1() );
}
TEST ( MinimumFrobeniusNormModelTest, mfnmodel_test2 ) 
{
  Wrapper_MinimumFrobeniusNormModel W;
  EXPECT_EQ( 1, W.mfnmodel_test2() );
}
TEST ( MinimumFrobeniusNormModelTest, mfnmodel_test3 ) 
{
  Wrapper_MinimumFrobeniusNormModel W;
  EXPECT_EQ( 1, W.mfnmodel_test3() );
}

//--------------------------------------------------------------------------------
