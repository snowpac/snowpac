#include "gtest/gtest.h"
#include "BasisForMinimumFrobeniusNormModel.hpp"


//--------------------------------------------------------------------------------
class Wrapper_BasisForMinimumFrobeniusNormModel {
  public:
    int basisformfnmodel_test1 ( ) {
 
      int dim = 2;
      BasisForMinimumFrobeniusNormModel basis( dim );
      std::vector<double> node( dim );
      std::vector< std::vector<double> > nodes;
      for ( unsigned j = 0; j < dim; ++j ) 
        node.at(j) = 0e0;
      nodes.push_back( node );
      for ( unsigned i = 0; i < dim; ++i ) {
        for ( unsigned j = 0; j < dim; ++j ) 
          node.at(j) = 0e0;
        node.at(i) = 1e0;
        nodes.push_back( node );
        for ( unsigned j = 0; j < dim; ++j ) 
          node.at(j) = 0e0;
        node.at(i) = -1e0;
        nodes.push_back( node );
      }

      for ( unsigned j = 0; j < dim; ++j ) 
        node.at(j) = -1e-1;
      node.at(0) = 1e-2;
      nodes.push_back( node );

      basis.compute_basis_coefficients ( nodes );

      std::vector<double> basis_evaluations;
      int test_passed = 1;
      for ( unsigned int i = 0; i < nodes.size(); ++i ) {
        basis_evaluations = basis.evaluate( nodes[i] );
        for ( unsigned int j = 0; j < nodes.size(); ++j ) {
          if ( j == i && fabs( basis_evaluations.at(j) - 1e0 ) > 1e-6 ) {
            std::cout << "Something is wrong at node " << i << " and basis " << j << std::endl;
            std::cout << "Error in evaluation = " << basis_evaluations.at(j) - 1e0 << std::endl;
            test_passed = 0; 
          }
          if ( j != i && fabs( basis_evaluations.at(j) - 0e0 ) > 1e-6 ) {
            std::cout << "Something is wrong at node " << i << " and basis " << j << std::endl;
            std::cout << "Error in evaluation = " << basis_evaluations.at(j) - 1e0 << std::endl;
            test_passed = 0; 
          }
        }
      }

      return test_passed;

    }

};
//--------------------------------------------------------------------------------



//--------------------------------------------------------------------------------
TEST ( BasisForMinimumFrobeniusNormModelTest, basisformfnmodel_test1 ) 
{
  Wrapper_BasisForMinimumFrobeniusNormModel W;
  EXPECT_EQ( 1, W.basisformfnmodel_test1() );
}
//--------------------------------------------------------------------------------
