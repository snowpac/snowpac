#include "BlackBoxData.hpp"

//--------------------------------------------------------------------------------
void BlackBoxData::initialize ( int n, int dim ) { 
  scaling = 1e0;
  scaled_node.resize( dim );
  for ( int i = 0; i < dim; ++i )
    scaled_node[i] = 0e0;
  values.resize( n );
  noise.resize( n ); 
  return;
}
//--------------------------------------------------------------------------------

//--------------------------------------------------------------------------------
void BlackBoxData::delete_history () {
  int nb_active_nodes = active_index.size();
  int found_point;
  int nb_nodes = nodes.size();
  int n = values.size();
  
  for ( int j = 0; j < nb_active_nodes; ++j ) {  
    if ( best_index == active_index.at(j) ) {
      best_index = j;
      break;
    }
  }

  for ( int i = nb_nodes-1; i >=0; --i ) {
    found_point = -1;
    for ( int j = 0; j < nb_active_nodes; ++j ) {  
      if ( i == active_index.at(j) ) {
        found_point = j;
        break;
      }
    }
    if ( found_point >= 0 ) {
      active_index.erase( active_index.begin() + found_point );
      nb_active_nodes--;
    } else {
      nodes.erase ( nodes.begin() + i );
      for ( int j = 0; j < n; ++j ) {
        values[j].erase( values[j].begin() + i );
        if ( noise[0].size() > 0 )
          noise[j].erase ( noise[j].begin() + i );        
      }
    }
  } 
  nb_nodes = nodes.size();

  assert( active_index.size() == 0 );
  for ( int i = 0; i < nb_nodes; ++i ) 
    active_index.push_back( i );
}
//--------------------------------------------------------------------------------

//--------------------------------------------------------------------------------
std::vector<double> &BlackBoxData::transform( std::vector<double> const& x ) {
  //center_node = nodes[ best_index ];
  //rescale( 1e0 / scaling, x, center_node, scaled_node);
  rescale( 1e0 / scaling, x, nodes[ best_index ], scaled_node);
  return scaled_node;
}
//--------------------------------------------------------------------------------

//--------------------------------------------------------------------------------
std::vector< std::vector<double> > &BlackBoxData::get_scaled_active_nodes ( 
  double scaling_input) {
//  std::vector<double> const &center_node_input, double scaling_input) {
  scaling = scaling_input;
//  center_node = center_node_input;
  int scaled_active_nodes_size = scaled_active_nodes.size();
  int active_index_size = active_index.size();
  if ( scaled_active_nodes_size > active_index_size ) {
    scaled_active_nodes.resize( active_index_size );
    scaled_active_nodes_size = active_index_size;
  }
  for ( int i = 0; i < scaled_active_nodes_size; ++i ) {
    rescale( 1e0 / scaling,  nodes[ active_index[i] ], 
             nodes[ best_index], scaled_active_nodes[i]);  
  }
  for ( int i = scaled_active_nodes_size; i < active_index_size; ++i ) {
    rescale( 1e0 / scaling,  nodes[ active_index[i] ], 
             nodes[ best_index ], scaled_node);
    scaled_active_nodes.push_back( scaled_node );
  }
  return scaled_active_nodes;
}
//--------------------------------------------------------------------------------

//--------------------------------------------------------------------------------
std::vector<double> &BlackBoxData::get_active_values( int i ) {
  int active_values_size = active_values.size();
  int active_index_size = active_index.size();
  if ( active_values_size > active_index_size ) {
    active_values.resize( active_index_size );
    active_values_size = active_index_size;
  }
  for ( int j = 0; j < active_values_size; ++j ) {
    active_values.at(j) = values.at( i ).at( active_index.at(j) );
//        std::cout << active_values.at(j) << std::endl;
  }
  for ( int j = active_values_size; j < active_index_size; ++j ) {
    active_values.push_back( values.at( i ).at(  active_index.at(j) ) );
//        std::cout << active_values.at(j) << std::endl;
  }
  return active_values;
}
//--------------------------------------------------------------------------------

//--------------------------------------------------------------------------------
std::vector<double> &BlackBoxData::get_active_noise( int i ) {
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
//--------------------------------------------------------------------------------



