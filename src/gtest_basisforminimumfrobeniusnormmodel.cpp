
#include "gtest/gtest.h"
#include "BasisForMinimumFrobeniusNormModel.hpp"
#include "BlackBoxData.hpp"

//--------------------------------------------------------------------------------
class Wrapper_BasisForMinimumFrobeniusNormModel {
  public:
    int basisformfnmodel_test1 ( ) {

      double delta = 0.5;
      int dim = 2;
      BasisForMinimumFrobeniusNormModel basis( dim, delta );
      BlackboxData data;
      std::vector<double> node(2);
      node.at(0) = 0.0; node.at(1) = 0e0;
      data.nodes.push_back( node );
      data.surrogate_nodes_index.push_back ( 0 ); 
      node.at(0) = 0.9; node.at(1) = 0.0;
      data.nodes.push_back( node );
      data.surrogate_nodes_index.push_back ( 1 ); 
      node.at(0) = 0.9; node.at(1) = 0.0;
      data.nodes.push_back( node );
      data.surrogate_nodes_index.push_back ( 2 ); 
      node.at(0) = -0.7; node.at(1) = 0.1;
      data.nodes.push_back( node );
      data.surrogate_nodes_index.push_back ( 3 ); 
      node.at(0) = -0.7; node.at(1) = 0.05;
      data.nodes.push_back( node );
      data.surrogate_nodes_index.push_back ( 4 ); 
      data.best_index = 0;
      basis.compute_basis_coefficients ( data );


      std::vector<double> g(2);
      std::vector< std::vector<double> > H(2);
      H[0].resize(2);
      H[1].resize(2);

/*
      for ( int i = 0; i < 5; ++i) {
      basis.get_mat_vec_representation (i , g, H);       
        std::cout << std::setprecision(12) << g << std::endl;
        std::cout << std::setprecision(12) << H <<  std::endl << std::endl;
      }
*/
      basis.get_mat_vec_representation (0 , g, H);       
      if ( fabs ( g[0] - 0.158715923409) > 1e-6 ) return 0;
      if ( fabs ( g[1] + 0.000531451993279 ) > 1e-6 ) return 0;
      if ( fabs ( H[0][0] + 0.793634976627 ) > 1e-6 ) return 0;
      if ( fabs ( H[0][1] - 0.0 ) > 1e-6 ) return 0;
      if ( fabs ( H[1][0] - 0.0 ) > 1e-6 ) return 0;
      if ( fabs ( H[1][1] - 0.00354301328851) > 1e-6 ) return 0;

      basis.get_mat_vec_representation (1 , g, H);       
      if ( fabs ( g[0] - 0.121530891754) > 1e-6 ) return 0;
      if ( fabs ( g[1] - 0.000116255123535 ) > 1e-6 ) return 0;
      if ( fabs ( H[0][0] - 0.173607651137 ) > 1e-6 ) return 0;
      if ( fabs ( H[0][1] - 0.0 ) > 1e-6 ) return 0;
      if ( fabs ( H[1][0] - 0.0 ) > 1e-6 ) return 0;
      if ( fabs ( H[1][1] + 0.000775034156862) > 1e-6 ) return 0;

      basis.get_mat_vec_representation (2 , g, H);       
      if ( fabs ( g[0] - 0.121530891754) > 1e-6 ) return 0;
      if ( fabs ( g[1] - 0.000116255123535 ) > 1e-6 ) return 0;
      if ( fabs ( H[0][0] - 0.173607651137 ) > 1e-6 ) return 0;
      if ( fabs ( H[0][1] - 0.0 ) > 1e-6 ) return 0;
      if ( fabs ( H[1][0] - 0.0 ) > 1e-6 ) return 0;
      if ( fabs ( H[1][1] + 0.000775034156862) > 1e-6 ) return 0;

      basis.get_mat_vec_representation (3 , g, H);       
      if ( fabs ( g[0] - 0.401777706918) > 1e-6 ) return 0;
      if ( fabs ( g[1] - 9.99970105825  ) > 1e-6 ) return 0;
      if ( fabs ( H[0][0] + 0.446419674353 ) > 1e-6 ) return 0;
      if ( fabs ( H[0][1] - 0.0 ) > 1e-6 ) return 0;
      if ( fabs ( H[1][0] - 0.0 ) > 1e-6 ) return 0;
      if ( fabs ( H[1][1] - 0.00199294497479) > 1e-6 ) return 0;

      basis.get_mat_vec_representation (4 , g, H);       
      if ( fabs ( g[0] + 0.803555413835) > 1e-6 ) return 0;
      if ( fabs ( g[1] + 9.99940211651  ) > 1e-6 ) return 0;
      if ( fabs ( H[0][0] - 0.892839348706  ) > 1e-6 ) return 0;
      if ( fabs ( H[0][1] - 0.0 ) > 1e-6 ) return 0;
      if ( fabs ( H[1][0] - 0.0 ) > 1e-6 ) return 0;
      if ( fabs ( H[1][1] + 0.00398588994958) > 1e-6 ) return 0;

  
      return 1;
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
