#ifndef HBlackBoxData
#define HBlackBoxData

#include "VectorOperations.hpp"
#include <vector>
#include <cassert>
#include <iostream>

class BlackBoxData : public VectorOperations{
  public:
  int max_nb_nodes;
  int best_index;
  std::vector< std::vector<double> > scaled_active_nodes;
  std::vector<double> active_values;
  std::vector<double> active_noise;  
  std::vector< std::vector<double> > nodes;
  std::vector< std::vector<double> > values;
  std::vector< std::vector<double> > noise;
  std::vector<int> active_index;
  std::vector<double> scaled_node;
  std::vector<double> center_node;
  double scaling;
  void initialize ( int n, int dim ) { 
    scaling = 1e0;
    scaled_node.resize( dim );
    for ( int i = 0; i < dim; ++i )
      scaled_node[i] = 0e0;
    values.resize( n );
    noise.resize( n ); 
  }
  std::vector<double> &transform( std::vector<double> const& x ) {
    rescale( 1e0 / scaling, x, center_node, scaled_node);
    return scaled_node;
  }
  std::vector< std::vector<double> > &get_scaled_active_nodes( 
    std::vector<double> const &center_node_input, double scaling_input) {
    scaling = scaling_input;
    center_node = center_node_input;
    int scaled_active_nodes_size = scaled_active_nodes.size();
    int active_index_size = active_index.size();
    if ( scaled_active_nodes_size > active_index_size ) {
      scaled_active_nodes.resize( active_index_size );
      scaled_active_nodes_size = active_index_size;
    }
    for ( int i = 0; i < scaled_active_nodes_size; ++i ) {
      rescale( 1e0 / scaling,  nodes[ active_index[i] ], center_node, scaled_active_nodes[i]);  
    }
    for ( int i = scaled_active_nodes_size; i < active_index_size; ++i ) {
      rescale( 1e0 / scaling,  nodes[ active_index[i] ], center_node, scaled_node);
      scaled_active_nodes.push_back( scaled_node );
    }
    return scaled_active_nodes;
  }
  std::vector<double> &get_active_values( int i ) {
    int active_values_size = active_values.size();
    int active_index_size = active_index.size();
    if ( active_values_size > active_index_size ) {
      active_values.resize( active_index_size );
      active_values_size = active_index_size;
    }
    for ( int j = 0; j < active_values_size; ++j ) {
      active_values.at(j) = values.at( i ).at( active_index.at(j) );
//      std::cout << active_values.at(j) << std::endl;
    }
    for ( int j = active_values_size; j < active_index_size; ++j ) {
      active_values.push_back( values.at( i ).at(  active_index.at(j) ) );
//      std::cout << active_values.at(j) << std::endl;
    }
    return active_values;
  }
  std::vector<double> &get_active_noise( int i ) {
    int active_noise_size = active_noise.size();
    int active_index_size = active_index.size();
    if ( active_noise_size > active_index_size ) {
      active_noise.resize( active_index_size );
      active_noise_size = active_index_size;
    }
    for (int j = 0; j < active_noise_size; ++j )
      active_noise[j] = noise[i][ active_index[j] ];
    for (int j = active_noise_size; j < active_index_size; ++j )
      active_noise.push_back( noise[ i ][ active_index[j] ] );
    return active_noise;
  }
};

#endif
