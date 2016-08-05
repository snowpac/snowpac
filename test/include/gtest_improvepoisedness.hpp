#include "ImprovePoisedness.hpp"
#include "MonomialBasisForMinimumFrobeniusNormModel.hpp"
#include "BlackBoxData.hpp"
#include <iomanip>

//--------------------------------------------------------------------------------
class Wrapper_ImprovePoisedness : public ImprovePoisedness {
public:
    double threshold_poisedness;
    double delta;
    Wrapper_ImprovePoisedness ( double threshold_for_poisedness_constant_input,
                                int max_nb_nodes_input, double &delta_input,
                                MonomialBasisForMinimumFrobeniusNormModel &basis_input,
                                int print_output_input ) :
     ImprovePoisedness ( basis_input, threshold_for_poisedness_constant_input,
                          max_nb_nodes_input, delta_input,
                        print_output_input )
     {
         delta = delta_input;
         threshold_poisedness = threshold_for_poisedness_constant_input;
     }
    int improvepoisedness_test1 ( ) {

      int dim = 2;
      BlackBoxData data;
      data.initialize( 1, dim );
      std::vector<double> node( dim );
      node.at(0) = 0.0; node.at(1) = 0e0;
      data.nodes.push_back( node );
      data.active_index.push_back ( 0 ); 
      node.at(0) = 0.9; node.at(1) = 0.01;
      data.nodes.push_back( node );
      data.active_index.push_back ( 1 ); 
      node.at(0) = 0.9; node.at(1) = 0.005;
      data.nodes.push_back( node );
      data.active_index.push_back ( 2 ); 
      node.at(0) = -0.9; node.at(1) = 0.1;
      data.nodes.push_back( node );
      data.active_index.push_back ( 3 ); 
      node.at(0) = -0.9; node.at(1) = -0.3;
      data.nodes.push_back( node );
      data.active_index.push_back ( 4 ); 
      data.best_index = 0;

      basis->compute_basis_coefficients ( 
        data.get_scaled_active_nodes( delta ) );
     
      int reference_node = 0;
      improve_poisedness ( reference_node, data );

      /*
      std::cout << data.nodes.size() << std::endl;
      for ( unsigned i = 0; i < data.nodes.size(); ++i) {
        std::cout << "x[" << i << "] = [";
        for ( unsigned j = 0; j < dim-1; ++j )
          std::cout << data.nodes[i][j] << ", ";
        std::cout << data.nodes[i][dim-1] << "]"<< std::endl;
      }
      for ( unsigned i = 0; i < data.active_index.size(); ++i) 
        std::cout << "active node: " << data.active_index[i] << std::endl;
      */

      if ( fabs( data.nodes.at(5).at(0) - 0.504080300855 ) > 1e-6 ) return 0;
      if ( fabs( data.nodes.at(5).at(1) - 0.863659668544 ) > 1e-6 ) return 0;
      if ( data.active_index.at(0) != 0 ) return 0;
      if ( data.active_index.at(1) != 2 ) return 0;
      if ( data.active_index.at(2) != 3 ) return 0;
      if ( data.active_index.at(3) != 4 ) return 0;
      if ( data.active_index.at(4) != 5 ) return 0;

      return 1;
      
    }

    int improvepoisedness_test2 ( ) {

      int dim = 2;
      BlackBoxData data;
      data.initialize( 1, dim );
      std::vector<double> node( dim);
      node.at(0) = 0.0; node.at(1) = 0e0;
      data.nodes.push_back( node );
      data.active_index.push_back ( 0 ); 
      node.at(0) = 0.9; node.at(1) = 0.01;
      data.nodes.push_back( node );
      data.active_index.push_back ( 1 ); 
      node.at(0) = 0.9; node.at(1) = 0.005;
      data.nodes.push_back( node );
      data.active_index.push_back ( 2 ); 
      node.at(0) = -0.9; node.at(1) = 0.1;
      data.nodes.push_back( node );
      data.active_index.push_back ( 3 ); 
      node.at(0) = -0.9; node.at(1) = -0.3;
      data.nodes.push_back( node );
      data.active_index.push_back ( 4 ); 
      data.best_index = 0;

      basis->compute_basis_coefficients ( 
        data.get_scaled_active_nodes( delta ) );
     
      int reference_node = 0;
  
      improve_poisedness ( reference_node, data );

      /*
      std::cout << data.nodes.size() << std::endl;
      for ( unsigned i = 0; i < data.nodes.size(); ++i) {
        std::cout << "x[" << i << "] = [";
        for ( unsigned j = 0; j < dim-1; ++j )
          std::cout << std::setprecision(12) << data.nodes[i][j] << ", ";
        std::cout << data.nodes[i][dim-1] << "]"<< std::endl;
      }
      for ( unsigned i = 0; i < data.active_index.size(); ++i) 
        std::cout << "active node: " << data.active_index[i] << std::endl;
      */

      if ( data.nodes.size() < 7 ) return 0;
      if ( fabs( data.nodes.at(5).at(0) - 0.0102300802149 ) > 1e-6 ) return -1;
      if ( fabs( data.nodes.at(5).at(1) - 0.0994753510344 ) > 1e-6 ) return -2;
      if ( fabs( data.nodes.at(6).at(0) - 0.0995591752728 ) > 1e-6 ) return -3;
      if ( fabs( data.nodes.at(6).at(1) + 0.00937926537677 ) > 1e-6 ) return -4;
      if ( data.active_index.at(0) != 0 ) return -5;
      if ( data.active_index.at(1) != 3 ) return -6;
      if ( data.active_index.at(2) != 4 ) return -7;
      if ( data.active_index.at(3) != 5 ) return -8;
      if ( data.active_index.at(4) != 6 ) return -9;

      return 1;
      
    }

    int improvepoisedness_test3 ( ) {

      int dim = 2;
      BlackBoxData data;
      data.initialize( 1, dim );
      std::vector<double> node( dim );
      node.at(0) = 0.0; node.at(1) = 0e0;
      data.nodes.push_back( node );
      data.active_index.push_back ( 0 ); 
      node.at(0) = 0.09; node.at(1) = 0.01;
      data.nodes.push_back( node );
      data.active_index.push_back ( 3 ); 
      node.at(0) = 0.09; node.at(1) = 0.005;
      data.nodes.push_back( node );
      data.active_index.push_back ( 4 ); 
      node.at(0) = -0.09; node.at(1) = 0.1;
      data.nodes.push_back( node );
      data.active_index.push_back ( 1 ); 
      node.at(0) = -0.09; node.at(1) = -0.03;
      data.nodes.push_back( node );
      data.active_index.push_back ( 2 ); 
      data.best_index = 0;

      basis->compute_basis_coefficients ( 
        data.get_scaled_active_nodes( delta ) );
     
      int reference_node = 0;

      improve_poisedness ( reference_node, data );

      /*
      std::cout << data.nodes.size() << std::endl;
      for ( unsigned i = 0; i < data.nodes.size(); ++i) {
        std::cout << "x[" << i << "] = [";
        for ( unsigned j = 0; j < dim-1; ++j )
          std::cout << std::setprecision(12) << data.nodes[i][j] << ", ";
        std::cout << data.nodes[i][dim-1] << "]"<< std::endl;
      }
      for ( unsigned i = 0; i < data.active_index.size(); ++i) 
        std::cout << "active node: " << data.active_index[i] << std::endl;
      */

      if ( data.nodes.size() < 7 ) return 0;
      if ( fabs( data.nodes.at(5).at(0) - 0.333873132343 ) > 1e-6 ) return 0;
      if ( fabs( data.nodes.at(5).at(1) + 0.372194481131 ) > 1e-6 ) return 0;
      if ( fabs( data.nodes.at(6).at(0) - 0.457553326487 ) > 1e-6 ) return 0;
      if ( fabs( data.nodes.at(6).at(1) - 0.201606060457 ) > 1e-6 ) return 0;
      if ( data.active_index.at(0) != 0 ) return 0;
      if ( data.active_index.at(1) != 3 ) return 0;
      if ( data.active_index.at(2) != 4 ) return 0;
      if ( data.active_index.at(3) != 5 ) return 0;
      if ( data.active_index.at(4) != 6 ) return 0;
      return 1;
      
    }

    int improvepoisedness_test4 ( ) {

      int dim = 2;
      BlackBoxData data;
      data.initialize( 1, dim );
      std::vector<double> node( dim );
      node.at(0) = -0.98; node.at(1) = -0.96;
      data.nodes.push_back( node );
      data.active_index.push_back ( 0 ); 
      node.at(0) = -0.96; node.at(1) = -0.98;
      data.nodes.push_back( node );
      data.active_index.push_back ( 1 ); 
      node.at(0) = 0.0; node.at(1) = 0.0;
      data.nodes.push_back( node );
      data.active_index.push_back ( 2 ); 
      node.at(0) = 0.98; node.at(1) = 0.96;
      data.nodes.push_back( node );
      data.active_index.push_back ( 3 ); 
      node.at(0) = 0.96; node.at(1) = 0.98;
      data.nodes.push_back( node );
      data.active_index.push_back ( 4 ); 
      node.at(0) = 0.94; node.at(1) = 0.94;
      data.nodes.push_back( node );
      data.active_index.push_back ( 5 ); 
      data.best_index = 2;

      basis->compute_basis_coefficients ( 
        data.get_scaled_active_nodes( delta ) );
     
      int reference_node = 2;
      improve_poisedness ( reference_node, data );

/*
      std::cout << std::setprecision(14) << data.nodes.size() << std::endl;
      for ( unsigned i = 0; i < data.nodes.size(); ++i) {
        std::cout << "x[" << i << "] = [";
        for ( unsigned j = 0; j < dim-1; ++j )
          std::cout << data.nodes[i][j] << ", ";
        std::cout << data.nodes[i][dim-1] << "]"<< std::endl;
      }
      for ( unsigned i = 0; i < data.active_index.size(); ++i) 
        std::cout << "active node: " << data.active_index[i] << std::endl;
*/
      
      std::vector< std::vector<double> > active_nodes;
      active_nodes = data.get_scaled_active_nodes( delta);

      if ( fabs( threshold_poisedness - 1e4 ) < 1e-6 ) {
        if ( fabs( active_nodes.at(0).at(0) + 0.98 ) > 1e-6 ) return -1;
        if ( fabs( active_nodes.at(0).at(1) + 0.96 ) > 1e-6 ) return -2;
        if ( fabs( active_nodes.at(1).at(0) + 0.96 ) > 1e-6 ) return -3;
        if ( fabs( active_nodes.at(1).at(1) + 0.98 ) > 1e-6 ) return -4;
        if ( fabs( active_nodes.at(2).at(0) + 0.00 ) > 1e-6 ) return -5;
        if ( fabs( active_nodes.at(2).at(1) + 0.00 ) > 1e-6 ) return -6;
        if ( fabs( active_nodes.at(3).at(0) - 0.98 ) > 1e-6 ) return -7;
        if ( fabs( active_nodes.at(3).at(1) - 0.96 ) > 1e-6 ) return -8;
        if ( fabs( active_nodes.at(4).at(0) - 0.96 ) > 1e-6 ) return -9;
        if ( fabs( active_nodes.at(4).at(1) - 0.98 ) > 1e-6 ) return -10;
        if ( fabs( active_nodes.at(5).at(0) - 0.94 ) > 1e-6 ) return -11;
        if ( fabs( active_nodes.at(5).at(1) - 0.94 ) > 1e-6 ) return -12;
      }

      if ( fabs( threshold_poisedness - 4e3 ) < 1e-6 ) {
        if ( fabs( active_nodes.at(0).at(0) + 0.98 ) > 1e-6 ) return -1;
        if ( fabs( active_nodes.at(0).at(1) + 0.96 ) > 1e-6 ) return -2;
        if ( fabs( active_nodes.at(1).at(0) + 0.96 ) > 1e-6 ) return -3;
        if ( fabs( active_nodes.at(1).at(1) + 0.98 ) > 1e-6 ) return -4;
        if ( fabs( active_nodes.at(2).at(0) + 0.00 ) > 1e-6 ) return -5;
        if ( fabs( active_nodes.at(2).at(1) + 0.00 ) > 1e-6 ) return -6;
        if ( fabs( active_nodes.at(3).at(0) - 0.98 ) > 1e-6 ) return -7;
        if ( fabs( active_nodes.at(3).at(1) - 0.96 ) > 1e-6 ) return -8;
        if ( fabs( active_nodes.at(4).at(0) - 0.96 ) > 1e-6 ) return -9;
        if ( fabs( active_nodes.at(4).at(1) - 0.98 ) > 1e-6 ) return -10;
        if ( fabs( active_nodes.at(5).at(0) - 0.70710678119654 ) > 1e-6 ) return -11;
        if ( fabs( active_nodes.at(5).at(1) + 0.70710678117656 ) > 1e-6 ) return -12;
      }

      if ( fabs( threshold_poisedness - 2e1 ) < 1e-6 ) {
        if ( fabs( active_nodes.at(0).at(0) + 0.98 ) > 1e-6 ) return -1;
        if ( fabs( active_nodes.at(0).at(1) + 0.96 ) > 1e-6 ) return -2;
        if ( fabs( active_nodes.at(1).at(0) + 0.96 ) > 1e-6 ) return -3;
        if ( fabs( active_nodes.at(1).at(1) + 0.98 ) > 1e-6 ) return -4;
        if ( fabs( active_nodes.at(2).at(0) + 0.00 ) > 1e-6 ) return -5;
        if ( fabs( active_nodes.at(2).at(1) + 0.00 ) > 1e-6 ) return -6;
        if ( fabs( active_nodes.at(3).at(0) - 0.98 ) > 1e-6 ) return -7;
        if ( fabs( active_nodes.at(3).at(1) - 0.96 ) > 1e-6 ) return -8;
        if ( fabs( active_nodes.at(4).at(0) - 0.70710678119654 ) > 1e-6 ) return -11;
        if ( fabs( active_nodes.at(4).at(1) + 0.70710678117656 ) > 1e-6 ) return -12;
        if ( fabs( active_nodes.at(5).at(0) + 0.52862933551908 ) > 1e-6 ) return -9;
        if ( fabs( active_nodes.at(5).at(1) - 0.84885279116596 ) > 1e-6 ) return -10;
      }

      if ( fabs( threshold_poisedness - 2e0 ) < 1e-6 ) {
        if ( fabs( active_nodes.at(0).at(0) + 0.98 ) > 1e-6 ) return -1;
        if ( fabs( active_nodes.at(0).at(1) + 0.96 ) > 1e-6 ) return -2;
        if ( fabs( active_nodes.at(1).at(0) + 0.96 ) > 1e-6 ) return -3;
        if ( fabs( active_nodes.at(1).at(1) + 0.98 ) > 1e-6 ) return -4;
        if ( fabs( active_nodes.at(2).at(0) + 0.00 ) > 1e-6 ) return -5;
        if ( fabs( active_nodes.at(2).at(1) + 0.00 ) > 1e-6 ) return -6;
        if ( fabs( active_nodes.at(3).at(0) - 0.98 ) > 1e-6 ) return -7;
        if ( fabs( active_nodes.at(3).at(1) - 0.96 ) > 1e-6 ) return -8;
        if ( fabs( active_nodes.at(4).at(0) - 0.70710678119654 ) > 1e-6 ) return -11;
        if ( fabs( active_nodes.at(4).at(1) + 0.70710678117656 ) > 1e-6 ) return -12;
        if ( fabs( active_nodes.at(5).at(0) + 0.52862933551908 ) > 1e-6 ) return -9;
        if ( fabs( active_nodes.at(5).at(1) - 0.84885279116596 ) > 1e-6 ) return -10;
      }

      return 1;
      
    }

};
//--------------------------------------------------------------------------------


//--------------------------------------------------------------------------------
TEST ( ImprovePoisednessTest, improvepoisedness_test1 ) 
{
  double threshold_for_poisedness_constant = 10.0;
  int max_nb_nodes = 5;
  double delta_loc = 1.0;
  int print_output = 0;
  int dim = 2;
  MonomialBasisForMinimumFrobeniusNormModel basis( dim );

  Wrapper_ImprovePoisedness W( threshold_for_poisedness_constant, max_nb_nodes,
                              delta_loc, basis, print_output );
  EXPECT_EQ( 1, W.improvepoisedness_test1() );
}
//--------------------------------------------------------------------------------

//--------------------------------------------------------------------------------
TEST ( ImprovePoisednessTest, improvepoisedness_test2 ) 
{
  double threshold_for_poisedness_constant = 10.0;
  int max_nb_nodes = 5;
  double delta_loc = 0.1;
  int print_output = 0;
  int dim = 2;
  MonomialBasisForMinimumFrobeniusNormModel basis( dim );

  Wrapper_ImprovePoisedness W( threshold_for_poisedness_constant, max_nb_nodes,
                              delta_loc, basis, print_output );
  EXPECT_EQ( 1, W.improvepoisedness_test2() );
}
//--------------------------------------------------------------------------------

//--------------------------------------------------------------------------------
TEST ( ImprovePoisednessTest, improvepoisedness_test3 ) 
{
  double threshold_for_poisedness_constant = 10.0;
  int max_nb_nodes = 5;
  double delta_loc = 0.5;
  int print_output = 0;
  int dim = 2;
  MonomialBasisForMinimumFrobeniusNormModel basis( dim );

  Wrapper_ImprovePoisedness W( threshold_for_poisedness_constant, max_nb_nodes,
                              delta_loc, basis, print_output );
  EXPECT_EQ( 1, W.improvepoisedness_test3() );
}
//--------------------------------------------------------------------------------

//--------------------------------------------------------------------------------
TEST ( ImprovePoisednessTest, improvepoisedness_test4_1 ) 
{
  double threshold_for_poisedness_constant = 2e4;
  int max_nb_nodes = 6;
  double delta_loc = 1.0;
  int print_output = 0;
  int dim = 2;
  MonomialBasisForMinimumFrobeniusNormModel basis( dim );

  Wrapper_ImprovePoisedness W( threshold_for_poisedness_constant, max_nb_nodes,
                              delta_loc, basis, print_output );
  EXPECT_EQ( 1, W.improvepoisedness_test4() );
}
//--------------------------------------------------------------------------------

//--------------------------------------------------------------------------------
TEST ( ImprovePoisednessTest, improvepoisedness_test4_2 ) 
{
  double threshold_for_poisedness_constant = 1e2;
  int max_nb_nodes = 6;
  double delta_loc = 1.0;
  int print_output = 0;
  int dim = 2;
  MonomialBasisForMinimumFrobeniusNormModel basis( dim );

  Wrapper_ImprovePoisedness W( threshold_for_poisedness_constant, max_nb_nodes,
                              delta_loc, basis, print_output );
  EXPECT_EQ( 1, W.improvepoisedness_test4() );
}
//--------------------------------------------------------------------------------
