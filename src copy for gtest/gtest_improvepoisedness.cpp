#include "ImprovePoisedness.hpp"
#include "BasisForMinimumFrobeniusNormModel.hpp"
#include "BlackBoxData.hpp"
#include <iomanip>

//--------------------------------------------------------------------------------
class Wrapper_ImprovePoisedness : public ImprovePoisedness {
  public:
    Wrapper_ImprovePoisedness ( double threshold_for_poisedness_constant_input,
                                int max_nb_nodes_input, double &delta_input,
                                BasisForMinimumFrobeniusNormModel &basis_input,
                                int print_output_input ) :
      ImprovePoisedness ( basis_input, threshold_for_poisedness_constant_input, 
                          max_nb_nodes_input, delta_input,
                          print_output_input ) { }

    int improvepoisedness_test1 ( ) {

      BlackBoxData data;
      std::vector<double> node(2);
      node.at(0) = 0.0; node.at(1) = 0e0;
      data.nodes.push_back( node );
      data.surrogate_nodes_index.push_back ( 0 ); 
      node.at(0) = 0.9; node.at(1) = 0.01;
      data.nodes.push_back( node );
      data.surrogate_nodes_index.push_back ( 1 ); 
      node.at(0) = 0.9; node.at(1) = 0.005;
      data.nodes.push_back( node );
      data.surrogate_nodes_index.push_back ( 2 ); 
      node.at(0) = -0.9; node.at(1) = 0.1;
      data.nodes.push_back( node );
      data.surrogate_nodes_index.push_back ( 3 ); 
      node.at(0) = -0.9; node.at(1) = -0.3;
      data.nodes.push_back( node );
      data.surrogate_nodes_index.push_back ( 4 ); 
      data.best_index = 0;
      basis->compute_basis_coefficients ( data );
     
      int reference_node = 0;
  

      improve_poisedness ( reference_node, data );

      if ( fabs( data.nodes.at(5).at(0) - 0.504080300855 ) > 1e-6 ) return 0;
      if ( fabs( data.nodes.at(5).at(1) - 0.863659668544 ) > 1e-6 ) return 0;
      if ( data.surrogate_nodes_index.at(0) != 0 ) return 0;
      if ( data.surrogate_nodes_index.at(1) != 2 ) return 0;
      if ( data.surrogate_nodes_index.at(2) != 3 ) return 0;
      if ( data.surrogate_nodes_index.at(3) != 4 ) return 0;
      if ( data.surrogate_nodes_index.at(4) != 5 ) return 0;

      return 1;
      
    }

    int improvepoisedness_test2 ( ) {

      BlackBoxData data;
      std::vector<double> node(2);
      node.at(0) = 0.0; node.at(1) = 0e0;
      data.nodes.push_back( node );
      data.surrogate_nodes_index.push_back ( 0 ); 
      node.at(0) = 0.9; node.at(1) = 0.01;
      data.nodes.push_back( node );
      data.surrogate_nodes_index.push_back ( 1 ); 
      node.at(0) = 0.9; node.at(1) = 0.005;
      data.nodes.push_back( node );
      data.surrogate_nodes_index.push_back ( 2 ); 
      node.at(0) = -0.9; node.at(1) = 0.1;
      data.nodes.push_back( node );
      data.surrogate_nodes_index.push_back ( 3 ); 
      node.at(0) = -0.9; node.at(1) = -0.3;
      data.nodes.push_back( node );
      data.surrogate_nodes_index.push_back ( 4 ); 
      data.best_index = 0;
      basis->compute_basis_coefficients ( data );
     
      int reference_node = 0;
  
      improve_poisedness ( reference_node, data );

      if ( data.nodes.size() < 7 ) return 0;
      if ( fabs( data.nodes.at(5).at(0) - 0.0102300802149 ) > 1e-6 ) return 0;
      if ( fabs( data.nodes.at(5).at(1) - 0.0994753510344 ) > 1e-6 ) return 0;
      if ( fabs( data.nodes.at(6).at(0) - 0.0843297324385 ) > 1e-6 ) return 0;
      if ( fabs( data.nodes.at(6).at(1) + 0.0537447507783 ) > 1e-6 ) return 0;
      if ( data.surrogate_nodes_index.at(0) != 0 ) return 0;
      if ( data.surrogate_nodes_index.at(1) != 2 ) return 0;
      if ( data.surrogate_nodes_index.at(2) != 3 ) return 0;
      if ( data.surrogate_nodes_index.at(3) != 5 ) return 0;
      if ( data.surrogate_nodes_index.at(4) != 6 ) return 0;

      return 1;
      
    }

    int improvepoisedness_test3 ( ) {

      BlackBoxData data;
      std::vector<double> node(2);
      node.at(0) = 0.0; node.at(1) = 0e0;
      data.nodes.push_back( node );
      data.surrogate_nodes_index.push_back ( 0 ); 
      node.at(0) = 0.09; node.at(1) = 0.01;
      data.nodes.push_back( node );
      data.surrogate_nodes_index.push_back ( 3 ); 
      node.at(0) = 0.09; node.at(1) = 0.005;
      data.nodes.push_back( node );
      data.surrogate_nodes_index.push_back ( 4 ); 
      node.at(0) = -0.09; node.at(1) = 0.1;
      data.nodes.push_back( node );
      data.surrogate_nodes_index.push_back ( 1 ); 
      node.at(0) = -0.09; node.at(1) = -0.03;
      data.nodes.push_back( node );
      data.surrogate_nodes_index.push_back ( 2 ); 
      data.best_index = 0;
      basis->compute_basis_coefficients ( data );
     
      int reference_node = 0;


      improve_poisedness ( reference_node, data );

      if ( data.nodes.size() < 7 ) return 0;
      if ( fabs( data.nodes.at(5).at(0) - 0.333873132343 ) > 1e-6 ) return 0;
      if ( fabs( data.nodes.at(5).at(1) + 0.372194481131 ) > 1e-6 ) return 0;
      if ( fabs( data.nodes.at(6).at(0) - 0.457553326487 ) > 1e-6 ) return 0;
      if ( fabs( data.nodes.at(6).at(1) - 0.201606060457 ) > 1e-6 ) return 0;
      if ( data.surrogate_nodes_index.at(0) != 0 ) return 0;
      if ( data.surrogate_nodes_index.at(1) != 3 ) return 0;
      if ( data.surrogate_nodes_index.at(2) != 4 ) return 0;
      if ( data.surrogate_nodes_index.at(3) != 5 ) return 0;
      if ( data.surrogate_nodes_index.at(4) != 6 ) return 0;
      return 1;
      
    }


};
//--------------------------------------------------------------------------------


//--------------------------------------------------------------------------------
TEST ( ImprovePoisednessTest, improvepoisedness_test1 ) 
{
  double threshold_for_poisedness_constant = 10.0;
  int max_nb_nodes = 5;
  double delta = 1.0;
  int print_output = 0;
  int dim = 2;
  BasisForMinimumFrobeniusNormModel basis( dim, delta );

  Wrapper_ImprovePoisedness W( threshold_for_poisedness_constant, max_nb_nodes,
                              delta, basis, print_output );
  EXPECT_EQ( 1, W.improvepoisedness_test1() );
}
//--------------------------------------------------------------------------------

//--------------------------------------------------------------------------------
TEST ( ImprovePoisednessTest, improvepoisedness_test2 ) 
{
  double threshold_for_poisedness_constant = 10.0;
  int max_nb_nodes = 5;
  double delta = 0.1;
  int print_output = 0;
  int dim = 2;
  BasisForMinimumFrobeniusNormModel basis( dim, delta );

  Wrapper_ImprovePoisedness W( threshold_for_poisedness_constant, max_nb_nodes,
                              delta, basis, print_output );
  EXPECT_EQ( 1, W.improvepoisedness_test2() );
}
//--------------------------------------------------------------------------------

//--------------------------------------------------------------------------------
TEST ( ImprovePoisednessTest, improvepoisedness_test3 ) 
{
  double threshold_for_poisedness_constant = 10.0;
  int max_nb_nodes = 5;
  double delta = 0.5;
  int print_output = 0;
  int dim = 2;
  BasisForMinimumFrobeniusNormModel basis( dim, delta );

  Wrapper_ImprovePoisedness W( threshold_for_poisedness_constant, max_nb_nodes,
                              delta, basis, print_output );
  EXPECT_EQ( 1, W.improvepoisedness_test3() );
}
//--------------------------------------------------------------------------------
